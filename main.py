import os
import sys
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.decomposition import IncrementalPCA

from ldm.util import instantiate_from_config


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ROOT_LDM = "/home/reddyanugu/IN/stable-diffusion/scripts/latent-diffusion"
SCRIPTS_DIR = os.path.join(ROOT_LDM, "scripts")
PARAMS_LIST_PATH = os.path.join(SCRIPTS_DIR, "params_list.pth")

DATASET_PTH = os.path.join(SCRIPTS_DIR, "task/data/data_set.pth")
IMAGES_DIR = os.path.join(SCRIPTS_DIR, "task/all_images_resized")  # contains <id>.JPEG
DEBUG_LOG = os.path.join(SCRIPTS_DIR, "params_1.txt")
LOSS_LOG = os.path.join(SCRIPTS_DIR, "loss_1.txt")

SAVE_MODEL_DIR = os.path.join(SCRIPTS_DIR, "params_train")
SAVE_IMAGE_DIR = os.path.join(SCRIPTS_DIR, "images_1")

os.makedirs(SAVE_MODEL_DIR, exist_ok=True)
os.makedirs(SAVE_IMAGE_DIR, exist_ok=True)

os.chdir(ROOT_LDM)
sys.path.append(os.path.join(ROOT_LDM, "ldm/models/diffusion"))

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
        "lossconfig": {"target": "torch.nn.Identity"}
    }
}

# You can keep cond_stage_config if you want, but we won't use it
cond_stage_config = {
    "target": "ldm.modules.encoders.modules.ClassEmbedder",
    "params": {"n_classes": 1001, "embed_dim": 512, "key": "class_label"}
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


params_list = torch.load(PARAMS_LIST_PATH, map_location="cpu")

unet_model = instantiate_from_config(unet_config).to(device)
vq_model = instantiate_from_config(first_stage_config).to(device)
cond_model = instantiate_from_config(cond_stage_config).to(device)

def load_parameters(model, parameters_list):
    state_dict = {name: tensor for name, tensor in parameters_list}
    model.load_state_dict(state_dict, strict=True)

# split params_list like your code
unet_params = params_list[:688]
unet_params = [(name[22:], param) for name, param in unet_params]

vq_params = params_list[688:-1]
vq_params = [(name[18:], param) for name, param in vq_params]

cond_params = params_list[-1:]
cond_params = [(name[17:], param) for name, param in cond_params]

load_parameters(unet_model, unet_params)
load_parameters(vq_model, vq_params)
load_parameters(cond_model, cond_params)


for p in vq_model.parameters():
    p.requires_grad = False
vq_model.eval()


class LinearNoiseScheduler:
    def __init__(self, num_timesteps, beta_start, beta_end):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1.0 - self.alpha_cum_prod)

    def add_noise(self, original, noise, t):
        # original: (B,C,H,W), t: (B,)
        b = original.shape[0]
        a = self.sqrt_alpha_cum_prod[t].reshape(b, 1, 1, 1).to(original.device)
        om = self.sqrt_one_minus_alpha_cum_prod[t].reshape(b, 1, 1, 1).to(original.device)
        return a * original + om * noise

    def sample_prev_timestep(self, xt, noise_pred, t_scalar):
        # t_scalar is an int or 0-d tensor (single timestep)
        t = int(t_scalar)

        x0 = (xt - self.sqrt_one_minus_alpha_cum_prod[t] * noise_pred) / torch.sqrt(self.alpha_cum_prod[t])
        x0 = torch.clamp(x0, -1.0, 1.0)

        mean = xt - (self.betas[t] * noise_pred) / (self.sqrt_one_minus_alpha_cum_prod[t])
        mean = mean / torch.sqrt(self.alphas[t])

        if t == 0:
            return mean, mean

        variance = (1 - self.alpha_cum_prod[t - 1]) / (1.0 - self.alpha_cum_prod[t])
        variance = variance * self.betas[t]
        sigma = torch.sqrt(variance)
        z = torch.randn_like(xt)
        return mean + sigma * z, x0

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


os.chdir(SCRIPTS_DIR)
eeg_signals_raw_with_mean_std = torch.load(DATASET_PTH, map_location="cpu")

def get_image_name_given_id(image_index: int) -> str:
    return list(eeg_signals_raw_with_mean_std.items())[2][1][image_index]

eeg_image_list = []
for sample in list(eeg_signals_raw_with_mean_std.items())[0][1]:
    eeg = sample["eeg"].float()     # (128, T)
    image_id = sample["image"]      # int index
    image_name = get_image_name_given_id(image_id)  # string id
    eeg_image_list.append((eeg, image_name))

def load_image_tensor_from_id(image_name: str):
    # expects "n0xxxxxxx_...." and file is "<image_name>.JPEG"
    path = os.path.join(IMAGES_DIR, image_name + ".JPEG")
    img = Image.open(path).convert("RGB")
    img = img.resize((256, 256))
    arr = np.array(img)  # (256,256,3) uint8
    x = torch.from_numpy(arr)  # HWC
    return x, image_name


def preprocess_eeg_to_flat(eeg_tensor: torch.Tensor) -> np.ndarray:
    # eeg_tensor: (128, T)
    eeg = eeg_tensor[:, 20:460]  # (128, 440)
    eeg = (eeg - (-32768.0)) / (32767.0 - (-32768.0))  # normalize to ~[0,1]
    flat = eeg.reshape(-1).cpu().numpy().astype(np.float32)  # (56320,)
    return flat

def fit_pca_incremental(pairs, n_components=256, batch_size=32, seed=42) -> IncrementalPCA:
    rng = np.random.default_rng(seed)
    idxs = np.arange(len(pairs))
    rng.shuffle(idxs)

    ipca = IncrementalPCA(n_components=n_components)

    buf = []
    for k in tqdm(idxs, desc="Fitting IncrementalPCA"):
        eeg_raw, _ = pairs[int(k)]
        buf.append(preprocess_eeg_to_flat(eeg_raw))
        if len(buf) >= batch_size:
            X = np.stack(buf, axis=0)  # (B, 56320)
            ipca.partial_fit(X)
            buf = []
    if len(buf) > 0:
        X = np.stack(buf, axis=0)
        ipca.partial_fit(X)
    return ipca

# choose PCA output dim (then encoder maps -> 512)
PCA_DIM = 256
pca = fit_pca_incremental(eeg_image_list, n_components=PCA_DIM, batch_size=16)


class EEGImagePCADataset(Dataset):
    def __init__(self, data_pairs, pca_model: IncrementalPCA):
        self.data = data_pairs
        self.pca = pca_model

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        eeg_raw, image_name = self.data[idx]

        flat = preprocess_eeg_to_flat(eeg_raw)                 # (56320,)
        eeg_pca = self.pca.transform(flat[None, :])[0]         # (PCA_DIM,)
        eeg_pca = torch.from_numpy(eeg_pca).float()            # torch float

        img_u8, name = load_image_tensor_from_id(image_name)
        class_id = dictionary[name[:9]]

        img = img_u8.float() / 255.0
        img = (img - 0.5) * 2.0                                # [-1,1]
        img = img.permute(2, 0, 1)                             # CHW

        return eeg_pca, img, class_id

dataset = EEGImagePCADataset(eeg_image_list, pca)


class EEGEncoder(nn.Module):
    def __init__(self, in_dim=PCA_DIM, out_dim=512, hidden=1024, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        # x: (B, PCA_DIM) -> (B, 512)
        return self.net(x)

eeg_encoder = EEGEncoder().to(device)


class CombinedUNet(nn.Module):
    def __init__(self, unet, eeg_enc):
        super().__init__()
        self.unet = unet
        self.eeg_enc = eeg_enc

    def forward(self, xt, t, eeg_pca):
        # eeg_pca: (B, PCA_DIM)
        cond = self.eeg_enc(eeg_pca)          # (B, 512)
        cond = cond.unsqueeze(1)              # (B, 1, 512)
        return self.unet(xt, t, cond)

model = CombinedUNet(unet_model, eeg_encoder).to(device)


train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

random.seed(42)
torch.manual_seed(42)

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

print("Train Dataset Indices:", train_dataset.indices, file=open(DEBUG_LOG, "a"))
print("Test Dataset Indices:", test_dataset.indices, file=open(DEBUG_LOG, "a"))

@torch.no_grad()
def generate_image_from_eeg_pca(eeg_pca_batch, guide_factor=1.0, end_timestep=999, skip=1):
    """
    eeg_pca_batch: (B, PCA_DIM) but you want 16 samples total for your grid
    We'll assume eeg_pca_batch already has shape (16, PCA_DIM).
    """
    xt = torch.randn(16, 3, 64, 64).to(device)

    # unconditional condition => zeros in PCA space (encoder will map it)
    uncond_pca = torch.zeros(16, PCA_DIM, device=device)

    # time loop
    timesteps = list(range(end_timestep, -1, -skip))
    for t in tqdm(timesteps, desc="Sampling"):
        t_tensor = torch.tensor([t], device=device, dtype=torch.long)

        # pred cond/uncond
        noise_c = model(xt, t_tensor, eeg_pca_batch)
        noise_u = model(xt, t_tensor, uncond_pca)
        noise = (guide_factor + 1.0) * noise_c - guide_factor * noise_u

        xt, _ = ls.sample_prev_timestep(xt, noise, t)
    return xt

@torch.no_grad()
def validate(batch, epoch_idx):
    eeg_pca, actual_image, _ = batch  # eeg_pca: (B,PCA_DIM)

    # take first 4 in batch, repeat each 4 times => 16
    eeg_pca = eeg_pca[:4].to(device)
    actual_image = actual_image[:4]

    cond = torch.cat([eeg_pca[i].unsqueeze(0).repeat(4, 1) for i in range(4)], dim=0)  # (16,PCA_DIM)
    latents = generate_image_from_eeg_pca(cond, guide_factor=1.0, end_timestep=999, skip=1)

    imgs = vq_model.decode(latents)  # (16,3,256,256) approx
    imgs = torch.clamp((imgs + 1.0) / 2.0, 0.0, 1.0).permute(0, 2, 3, 1).cpu().numpy()

    fig, axs = plt.subplots(4, 5, figsize=(15, 15))
    for i in range(4):
        for j in range(4):
            axs[i, j].imshow(imgs[i * 4 + j])
            axs[i, j].axis("off")
        axs[i, 4].imshow(actual_image[i].permute(1, 2, 0).cpu().numpy())
        axs[i, 4].axis("off")

    out_path = os.path.join(SAVE_IMAGE_DIR, f"image_{epoch_idx}.png")
    plt.savefig(out_path)
    plt.close()


num_epochs = 1000
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
p_uncond = 0.3

def train_step(eeg_pca, image):
    # eeg_pca: (B,PCA_DIM), image: (B,3,256,256)
    eeg_pca = eeg_pca.to(device)
    image = image.to(device)

    with torch.no_grad():
        z0 = vq_model.encode(image)  # (B,3,64,64)

    b = z0.shape[0]
    t = torch.randint(0, 1000, (b,), device=device)
    noise = torch.randn(b, 3, 64, 64, device=device)
    xt = ls.add_noise(z0, noise, t)

    # CFG dropout on conditioning
    if random.random() < p_uncond:
        eeg_pca = torch.zeros_like(eeg_pca)

    noise_pred = model(xt, t, eeg_pca)
    return F.mse_loss(noise_pred, noise)

print("training started")
losses = []

for epoch in tqdm(range(num_epochs), desc="Epochs"):
    model.train()
    loss_epoch = []

    for batch in train_loader:
        eeg_pca, image, _ = batch

        loss = train_step(eeg_pca, image)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_epoch.append(loss.item())

    mean_loss = float(np.mean(loss_epoch)) if len(loss_epoch) else 0.0
    losses.append(mean_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {mean_loss}", file=open(LOSS_LOG, "a"))

    # save + validate each epoch (same behavior as you)
    torch.save(model.state_dict(), os.path.join(SAVE_MODEL_DIR, f"train_model_latest_new{epoch}.pth"))

    model.eval()
    with torch.no_grad():
        # validate on a single batch from train loader
        for batch in train_loader:
            validate(batch, epoch)
            break
