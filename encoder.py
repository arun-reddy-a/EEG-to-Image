import os
import sys
import random
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import clip
from sklearn.decomposition import IncrementalPCA

# ------------------------------------------------------------
# Device
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------
# Paths (adjust if needed)
# ------------------------------------------------------------
ROOT = "/home/reddyanugu/IN/stable-diffusion/scripts/latent-diffusion/scripts"
DATA_PTH = os.path.join(ROOT, "task/data/data_set.pth")
IMG_DIR = os.path.join(ROOT, "task/all_images_resized")
SAVE_DIR = os.path.join(ROOT, "eeg_clip")
os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------------------------------------------------
# Load CLIP (ViT encoder)
# ------------------------------------------------------------
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

# freeze CLIP
for p in clip_model.parameters():
    p.requires_grad = False

CLIP_EMB_DIM = clip_model.visual.output_dim  # usually 512

# ------------------------------------------------------------
# Load EEG–image raw dataset
# ------------------------------------------------------------
eeg_signals_raw_with_mean_std = torch.load(DATA_PTH, map_location="cpu")

def get_image_name_given_id(idx):
    return list(eeg_signals_raw_with_mean_std.items())[2][1][idx]

pairs = []
for sample in list(eeg_signals_raw_with_mean_std.items())[0][1]:
    eeg = sample["eeg"].float()     # (128, T)
    img_id = sample["image"]        # int
    img_name = get_image_name_given_id(img_id)
    pairs.append((eeg, img_name))

# ------------------------------------------------------------
# EEG preprocessing + PCA
# ------------------------------------------------------------
def preprocess_eeg(eeg):
    # eeg: (128, T)
    eeg = eeg[:, 20:460]  # (128, 440)
    eeg = (eeg - (-32768.0)) / (32767.0 - (-32768.0))
    return eeg.reshape(-1).cpu().numpy().astype(np.float32)

PCA_DIM = 256

def fit_pca(data, n_components=256, batch_size=16):
    ipca = IncrementalPCA(n_components=n_components)
    buf = []
    for eeg, _ in tqdm(data, desc="Fitting PCA"):
        buf.append(preprocess_eeg(eeg))
        if len(buf) == batch_size:
            ipca.partial_fit(np.stack(buf))
            buf = []
    if buf:
        ipca.partial_fit(np.stack(buf))
    return ipca

pca = fit_pca(pairs, PCA_DIM)

# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------
class EEGImageDataset(Dataset):
    def __init__(self, data, pca):
        self.data = data
        self.pca = pca

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        eeg, img_name = self.data[idx]

        eeg_flat = preprocess_eeg(eeg)
        eeg_pca = self.pca.transform(eeg_flat[None, :])[0]
        eeg_pca = torch.from_numpy(eeg_pca).float()

        img_path = os.path.join(IMG_DIR, img_name + ".JPEG")
        img = Image.open(img_path).convert("RGB")
        img = clip_preprocess(img)   # CLIP normalization

        return eeg_pca, img

dataset = EEGImageDataset(pairs, pca)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=4)

# ------------------------------------------------------------
# EEG Encoder (to CLIP embedding space)
# ------------------------------------------------------------
class EEGEncoder(nn.Module):
    def __init__(self, in_dim=PCA_DIM, out_dim=CLIP_EMB_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, out_dim)
        )

    def forward(self, x):
        return self.net(x)

eeg_encoder = EEGEncoder().to(device)

# ------------------------------------------------------------
# CLIP-style contrastive loss
# ------------------------------------------------------------
class CLIPLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / temperature)))

    def forward(self, eeg_emb, img_emb):
        eeg_emb = F.normalize(eeg_emb, dim=-1)
        img_emb = F.normalize(img_emb, dim=-1)

        logits = torch.matmul(eeg_emb, img_emb.T) * self.logit_scale.exp()

        labels = torch.arange(eeg_emb.size(0), device=eeg_emb.device)

        loss_eeg = F.cross_entropy(logits, labels)
        loss_img = F.cross_entropy(logits.T, labels)

        return (loss_eeg + loss_img) / 2

criterion = CLIPLoss()
optimizer = torch.optim.AdamW(
    list(eeg_encoder.parameters()) + list(criterion.parameters()),
    lr=3e-4,
    weight_decay=1e-4
)

# ------------------------------------------------------------
# Training loop
# ------------------------------------------------------------
def train_epoch(loader):
    eeg_encoder.train()
    total_loss = 0.0

    for eeg_pca, img in tqdm(loader, leave=False):
        eeg_pca = eeg_pca.to(device)
        img = img.to(device)

        with torch.no_grad():
            img_emb = clip_model.encode_image(img)

        eeg_emb = eeg_encoder(eeg_pca)

        loss = criterion(eeg_emb, img_emb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

@torch.no_grad()
def eval_epoch(loader):
    eeg_encoder.eval()
    total_loss = 0.0

    for eeg_pca, img in loader:
        eeg_pca = eeg_pca.to(device)
        img = img.to(device)

        img_emb = clip_model.encode_image(img)
        eeg_emb = eeg_encoder(eeg_pca)

        loss = criterion(eeg_emb, img_emb)
        total_loss += loss.item()

    return total_loss / len(loader)

# ------------------------------------------------------------
# Run training
# ------------------------------------------------------------
epochs = 50
for epoch in range(epochs):
    train_loss = train_epoch(train_loader)
    val_loss = eval_epoch(val_loader)

    print(f"Epoch [{epoch+1}/{epochs}] | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    torch.save(
        eeg_encoder.state_dict(),
        os.path.join(SAVE_DIR, f"eeg_encoder_epoch_{epoch}.pth")
    )
