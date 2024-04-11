import torch
import torch.nn as nn
from functools import partial
import clip
import random
from tqdm import tqdm
import os
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from torch.optim.lr_scheduler import LambdaLR
import numpy as np 
from einops import rearrange
from torchvision.utils import make_grid
from einops import rearrange, repeat
import transformers
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

new_dir = '/home/reddyanugu/IN/stable-diffusion/scripts/latent-diffusion'

os.chdir(new_dir)

import sys
sys.path.append('/home/reddyanugu/IN/stable-diffusion/scripts/latent-diffusion/ldm/models/diffusion')

first_stage_config = {
    "target": "ldm.models.autoencoder.VQModelInterface",
    "params": {
        "embed_dim": 3,
        "n_embed": 8192,
        "ddconfig": {
            "double_z": False,
            "z_channels": 3,
            "resolution": 256,
            "in_channels": 3,
            "out_ch": 3,
            "ch": 128,
            "ch_mult": [1, 2, 4],
            "num_res_blocks": 2,
            "attn_resolutions": [],
            "dropout": 0.0
        },
        "lossconfig": {
            "target": "torch.nn.Identity"
        }
    }
}

cond_stage_config = {
    "target": "ldm.modules.encoders.modules.ClassEmbedder",
    "params": {
        "n_classes": 1001,
        "embed_dim": 512,
        "key": "class_label"
    }
}

unet_config = {
    "target": "ldm.modules.diffusionmodules.openaimodel.UNetModel",
    "params": {
        "image_size": 64,
        "in_channels": 3,
        "out_channels": 3,
        "model_channels": 192,
        "attention_resolutions": [8, 4, 2],
        "num_res_blocks": 2,
        "channel_mult": [1, 2, 3, 5],
        "num_heads": 1,
        "use_spatial_transformer": True,
        "transformer_depth": 1,
        "context_dim": 512
    }
}

params_list = torch.load('./scripts/params_list.pth')

unet_model = instantiate_from_config(unet_config).to(device)
vq_model = instantiate_from_config(first_stage_config).to(device)
cond_model = instantiate_from_config(cond_stage_config).to(device)

def enc_dec(path):
    img = Image.open(path)
    img = img.resize((256, 256))
    img = np.array(img)
    img = ((img/255.0)-0.5)*2
    img = torch.tensor(img).permute(2,0,1).unsqueeze(0).to(device).float()
    enc = vq_model.encode(img)
    dec = vq_model.decode(enc)
    return dec

os.chdir('/home/reddyanugu/IN/stable-diffusion/scripts/latent-diffusion/scripts')

unet_params = []
for name , param in unet_model.named_parameters():
    unet_params.append(name)

vq_params = []
for name , param in vq_model.named_parameters():
    vq_params.append(name)



unet_params = params_list[:688]
unet_params = [(name[22:], param) for name, param in unet_params]
vq_params = params_list[688: -1]
vq_params = [(name[18:], param) for name, param in vq_params]
cond_params = params_list[-1:]
cond_params = [(name[17:], param) for name, param in cond_params]

dec = enc_dec('./image.JPEG')
dec_image = dec.squeeze(0).permute(1,2,0).cpu().detach().numpy()
(Image.fromarray((((dec_image+1)/2)*255).astype(np.uint8))).save('before_load.png')

def load_parameters(model, parameters_list):
    # Create a state_dict from the list of tuples
    state_dict = {name: tensor for name, tensor in parameters_list}

    # Load the state_dict into the model
    model.load_state_dict(state_dict)

load_parameters(unet_model, unet_params)
load_parameters(vq_model, vq_params)
load_parameters(cond_model, cond_params)


class LinearNoiseScheduler:
    r"""
    Class for the linear noise scheduler that is used in DDPM.
    """
    def __init__(self, num_timesteps, beta_start, beta_end):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0).to(device)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod).to(device)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod).to(device)
        
    def add_noise(self, original, noise, t):
        r"""
        Forward method for diffusion
        :param original: Image on which noise is to be applied
        :param noise: Random Noise Tensor (from normal dist)
        :param t: timestep of the forward process of shape -> (B,)
        :return:
        """
        original_shape = original.shape
        batch_size = original_shape[0]
        
        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod[t].reshape(batch_size)
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod[t].reshape(batch_size)
        
        # Reshape till (B,) becomes (B,1,1,1) if image is (B,C,H,W)
        for _ in range(len(original_shape)-1):
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
        for _ in range(len(original_shape)-1):
            sqrt_one_minus_alpha_cum_prod = sqrt_one_minus_alpha_cum_prod.unsqueeze(-1)

        # Apply and Return Forward process equation
        return (sqrt_alpha_cum_prod.to(original.device) * original
                + sqrt_one_minus_alpha_cum_prod.to(original.device) * noise)
        
    def sample_prev_timestep(self, xt, noise_pred, t):
        r"""
            Use the noise prediction by model to get
            xt-1 using xt and the nosie predicted
        :param xt: current timestep sample
        :param noise_pred: model noise prediction
        :param t: current timestep we are at
        :return:
        """
        x0 = (xt - (self.sqrt_one_minus_alpha_cum_prod[t] * noise_pred)) / torch.sqrt(self.alpha_cum_prod[t])
        x0 = torch.clamp(x0, -1., 1.)

        mean = xt - ((self.betas[t])*noise_pred)/(self.sqrt_one_minus_alpha_cum_prod[t])
        mean = mean / torch.sqrt(self.alphas[t])
        
        if t == 0:
            return mean, mean
        else:
            variance = (1-self.alpha_cum_prod[t-1]) / (1.0 - self.alpha_cum_prod[t])
            variance = variance * self.betas[t]
            sigma = variance ** 0.5
            z = torch.randn(xt.shape).to(xt.device)
            
            # OR
            # variance = self.betas[t]
            # sigma = variance ** 0.5
            # z = torch.randn(xt.shape).to(xt.device)
            return mean + sigma*z, x0

    def sample_prev_timestep_ddim (self, xt, noise_pred, t, prev_t):

        x0 = (xt - self.sqrt_one_minus_alpha_cum_prod[t]*noise_pred) / torch.sqrt(self.alpha_cum_prod[t])
        x0 = torch.clamp(x0, -1., 1.)
        xt_prev = self.sqrt_alpha_cum_prod[prev_t] * x0 + self.sqrt_one_minus_alpha_cum_prod[prev_t]*noise_pred
        return xt_prev, x0


ls = LinearNoiseScheduler(1000, 1e-4, 0.02)


dictionary = {}

dictionary['n02106662'] = 235
dictionary['n02124075'] = 285
dictionary['n02281787'] = 326
dictionary['n02389026'] = 339
dictionary['n02492035'] = 378
dictionary['n02504458'] = 386
dictionary['n02510455'] = 388
dictionary['n02607072'] = 393
dictionary['n02690373'] = 404
dictionary['n02906734'] = 462

dictionary['n02951358'] = 472
dictionary['n02992529'] = 487
dictionary['n03063599'] = 504
dictionary['n03100240'] = 511
dictionary['n03180011'] = 527
dictionary['n03197337'] = 531
dictionary['n03272010'] = 402
dictionary['n03272562'] = 547
dictionary['n03297495'] = 550
dictionary['n03376595'] = 559

dictionary['n03445777'] = 574
dictionary['n03452741'] = 579
dictionary['n03584829'] = 606
dictionary['n03590841'] = 607
dictionary['n03709823'] = 636
dictionary['n03773504'] = 657
dictionary['n03775071'] = 658
dictionary['n03792782'] = 671
dictionary['n03792972'] = 672
dictionary['n03877472'] = 697

dictionary['n03888257'] = 701
dictionary['n03982430'] = 736
dictionary['n04044716'] = 755
dictionary['n04069434'] = 759
dictionary['n04086273'] = 763
dictionary['n04120489'] = 770
dictionary['n07753592'] = 954
dictionary['n07873807'] = 963
dictionary['n11939491'] = 985
dictionary['n13054560'] = 997


eeg_signals_raw_with_mean_std = torch.load('task/data/data_set.pth')

def get_image_name_given_id(id):
    return list(eeg_signals_raw_with_mean_std.items())[2][1][id]

eeg_image_list = []
for sample in (list(eeg_signals_raw_with_mean_std.items()))[0][1]:
    eeg = sample['eeg'].float()
    image_id = sample['image']
    image = get_image_name_given_id(image_id)
    eeg_image_list.append((eeg,image))

def get_array_from_jpeg(id):
    path = 'task/all_images_resized/' + id+ '.JPEG'
    # Load the JPEG image
    image = Image.open(path)

    # Resize the image to 256x256 pixels
    image = image.resize((256, 256))

    # Convert the image to a NumPy array
    image_array = np.array(image)

    image_array = torch.from_numpy(image_array)

    id_string = id
    
    return image_array, id_string

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.pca_data = torch.load('pca_data.pth')
        self.normal_data = torch.load('data.pth')


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        eeg, image_id = self.data[idx]
        eeg = (eeg[:,20:460]-(-32768.))/(32767.-(-32768.)) # (128, 440) numpy array
        image, id_string = get_array_from_jpeg(image_id)
        class_id = dictionary[id_string[:9]]
        eeg = eeg.reshape(-1).numpy()
        # print(self.normal_data.shape, self.pca_data.shape, eeg.shape, 'shape')
        loc = np.where(np.all(self.normal_data == eeg, axis=1))[0][0]
        eeg_pca = self.pca_data[loc]
        eeg = torch.tensor(eeg_pca).float()
        image = ((image/255.0)-0.5)*2
        if (image.shape != (256,256,3)):
            image = image.unsqueeze(-1).repeat(1,1,3)
        image = image.permute(2,0,1)
        eeg = eeg.to(device)
        image = image.to(device)
        return eeg, image, class_id

dataset = CustomDataset(eeg_image_list)

def plot_image(img):
    image_normalized = torch.clamp((img+1.0)/2.0, min=0.0, max=1.0)*255.0
    image_array = image_normalized.permute(0,1,2).cpu().detach().numpy().astype(np.uint8)
    return Image.fromarray(image_array)

# Initialize the model
input_dim = 440  # Dimensionality of the input tensor
d_model = 512  # Dimensionality of the encoder output
nhead =  2 # Number of attention heads
num_layers = 2  # Number of transformer encoder layers

def fix_model(model):
    for param in model.parameters():
        param.requires_grad = False

fix_model(vq_model)

# Define the sizes of your training and testing subsets
# train_size = int(0.8 * len(dataset)) # 80% for training
# test_size = len(dataset) - train_size # 20% for testing

train_size = int(0.8 * len(dataset))
test_size = len(dataset)-train_size

# Split the dataset into training and testing subsets
random.seed(42)
torch.manual_seed(42)
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Define DataLoader for training and testing subsets
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print("Train Dataset Indices:", train_dataset.indices, file=open('params_1.txt', 'a'))
print("Test Dataset Indices:", test_dataset.indices, file=open('params_1.txt', 'a'))


class Combined_U_Net(nn.Module):
    def __init__(self, unet_model):
        super(Combined_U_Net, self).__init__()
        self.unet_model = unet_model

    def forward(self, xt, t, eeg):
        noise_pred = self.unet_model(xt, t, eeg)
        return noise_pred

combined_u_net = Combined_U_Net(unet_model).to(device)

num_epochs = 1000
# optimizer = torch.optim.Adam(combined_u_net.parameters(), lr=3e-5)
optimizer = torch.optim.Adam(combined_u_net.parameters(), lr=3e-5)
p_uncond = 0.3

def generate_image (cond, guide_factor, end_timestep=100000, skip=1):

    xt = torch.randn(16, 3, 64, 64).to(device)
    un_cond = torch.zeros(16, 1, 512).to(device)

    
    timestamps = torch.arange(end_timestep, 0, -1*skip)[:, None].to(device)

    for t in tqdm(timestamps):
        index = torch.where(timestamps == t)[0]
        with torch.no_grad():
            
            noise_pred = (guide_factor+1)*unet_model(xt, t, cond) - (guide_factor)*unet_model(xt, t, un_cond)
            if False:
                if (index + 1) != len(timestamps):
                    xt, x0 = ls.sample_prev_timestep_ddim(xt, noise_pred, t, timestamps[index+1])
            else:
                xt, x0 = ls.sample_prev_timestep(xt, noise_pred, t)
    return xt


def validate(batch):
    eeg, actual_image, ids = batch
    cond_1 = (eeg[0].unsqueeze(0)).unsqueeze(1).repeat(4,1,1)
    cond_2 = (eeg[1].unsqueeze(0)).unsqueeze(1).repeat(4,1,1)
    cond_3 = (eeg[2].unsqueeze(0)).unsqueeze(1).repeat(4,1,1)
    cond_4 = (eeg[3].unsqueeze(0)).unsqueeze(1).repeat(4,1,1)
    cond = torch.cat([cond_1, cond_2, cond_3, cond_4], dim=0) # (16, 1, 512)


    img_gen = generate_image(cond, 1.0, 999, 1)

    img_dec_1 = vq_model.decode(img_gen)
    img_dec_1 = torch.clamp((img_dec_1+1.0)/2.0, min=0.0, max=1.0)
    img_dec_1 = img_dec_1.permute(0,2,3,1).cpu().detach().numpy()

    # plot all imaes in a row
    fig, axs = plt.subplots(4, 5, figsize=(15, 15))
    # axs.title('Generated Images')
    # plot the generated images
    for i in range(4):
        for j in range(4):
            axs[i, j].imshow(img_dec_1[i*4+j])
            axs[i, j].axis('off')
    for i in range(4):
        axs[i, 4].imshow(actual_image[i].permute(1,2,0).cpu().detach().numpy())
        axs[i, 4].axis('off')


def find_loss(model, image, class_id):
    cond = eeg
    noise = torch.randn(1, 3, 64, 64).to(device)
    t = torch.tensor([500]).to(device)
    image = image.to(device)
    with torch.no_grad():
        image = vq_model.encode(image)
    xt = ls.add_noise(image, noise, t).to(device)
    noise_pred = unet_model(xt, t, cond)
    loss = F.mse_loss(noise_pred, noise)
    return loss

print('training started')
losses = []
for epoch in tqdm(range(num_epochs)):
    loss_epoch = []
    for batch in ((train_loader)):
        eeg, image, ids = batch
        eeg = eeg.unsqueeze(1)
        with torch.no_grad():
            image = vq_model.encode(image)
        batch_size = eeg.size(0)
        t = torch.randint(0, 1000, (batch_size,)).to(device)
        noise = torch.randn(batch_size, 3, 64, 64).to(device)
        p = torch.rand(1).item()
        if (p<p_uncond):
            eeg = torch.zeros(batch_size, 1, 512).to(device)
        xt = ls.add_noise(image, noise, t).to(device)
        noise_pred = combined_u_net(xt, t, eeg)
        loss = F.mse_loss(noise_pred, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_epoch.append(loss.item())
        

    image = train_dataset[0][1].unsqueeze(0)
    eeg = train_dataset[0][0].unsqueeze(0).unsqueeze(1)
    print('the actual loss in train', find_loss(combined_u_net, image, eeg), file=open('params_1.txt', 'a'))
    losses.append(np.mean(loss_epoch))
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss in train: {np.mean(loss_epoch)}', file=open('loss_1.txt', 'a'))
    # save the model
    if epoch%1 == 0 or epoch == num_epochs-1:
        torch.save(combined_u_net.state_dict(), f'./params_train/train_model_latest_new{epoch}.pth')
        validate(batch)
        plt.savefig(f'./images_1/image_{epoch}.png')
        plt.close()


# validation
print('validation started')