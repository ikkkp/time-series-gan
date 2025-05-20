#!/usr/bin/env python3
"""
@File  : timegradGANc.py
@Author: Ezra Zephyr (WGAN-GP with latent z_dim)
@Date  : 2025/04/25
@Desc  : PyTorch one-step GAN on time-series windows using WGAN-GP
          - latent-to-window generator (z_dim)
          - no MSE pretraining
          - gradient penalty for discriminator
          - outputs PCA & t-SNE plots and metrics
"""
import os
import json
import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from torch.utils.data import DataLoader, Dataset

from sklearn.preprocessing import StandardScaler
from tqdm import trange

from metrics.visualization_metrics import visualization
from metrics.discriminative_metrics import discriminative_score_metrics
from metrics.predictive_metrics import predictive_score_metrics
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

# ---------------------------------------------------------------------------
# Model Components
# ---------------------------------------------------------------------------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = torch.exp(
            -torch.arange(half, device=t.device) * (np.log(10000) / (half - 1))
        )
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_ch))
        self.res_conv = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.conv1(x) + self.time_mlp(t_emb).unsqueeze(-1)
        h = self.bn1(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        h = self.bn2(h)
        h = h + self.res_conv(x)
        return self.act(h)

class Generator(nn.Module):
    def __init__(self, feat_dim, z_dim=16, time_emb_dim=128, hidden_dim=128, num_blocks=6):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.time_emb = SinusoidalPosEmb(time_emb_dim)
        self.input_proj = nn.Conv1d(z_dim, hidden_dim, 1)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim, time_emb_dim)
            for _ in range(num_blocks)
        ])
        self.output_proj = nn.Conv1d(hidden_dim, feat_dim, 1)

    def forward(self, z, t):
        # z: (batch, z_dim, seq_len)
        t_emb = self.time_emb(t)
        h = self.input_proj(z)  # (batch, hidden_dim, seq_len)
        for blk in self.blocks:
            h = blk(h, t_emb)
        out = self.output_proj(h)
        # back to (batch, seq_len, feat_dim)
        return out.permute(0, 2, 1)

class Discriminator(nn.Module):
    def __init__(self, feat_dim, hidden_dim=128, num_layers=4, dropout=0.2):
        super(Discriminator, self).__init__()
        layers = []
        in_ch = feat_dim
        for _ in range(num_layers):
            layers += [
                nn.Conv1d(in_ch, hidden_dim, 3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout)
            ]
            in_ch = hidden_dim
        self.cnn = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len, feat_dim)
        h = self.cnn(x.permute(0, 2, 1))
        h = h.mean(dim=2)
        return self.fc(h).squeeze(1)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class TimeSeriesWindowDataset(Dataset):
    def __init__(self, data, seq_len):
        n = len(data) - seq_len + 1
        if n <= 0:
            raise ValueError(f"Data length {len(data)} < seq_len {seq_len}")
        self.windows = np.stack([data[i:i+seq_len] for i in range(n)], axis=0)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return torch.tensor(self.windows[idx], dtype=torch.float32)

# ---------------------------------------------------------------------------
# WGAN-GP Training
# ---------------------------------------------------------------------------
def gradient_penalty(D, real, fake, device, lambda_gp=10):
    bs = real.size(0)
    alpha = torch.rand(bs, 1, 1, device=device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interp = D(interp)
    grads = grad(
        outputs=d_interp,
        inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True
    )[0]
    grads = grads.contiguous().view(bs, -1)
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    return gp

def train_gan(G, D, loader, device, epochs, opt_G, opt_D):
    for ep in range(1, epochs + 1):
        for real in loader:
            real = real.to(device)
            bs = real.size(0)
            seq_len = loader.dataset.windows.shape[1]

            # Discriminator step
            noise = torch.randn(bs, G.z_dim, seq_len, device=device)
            fake = G(noise, torch.zeros(bs, device=device, dtype=torch.long))
            d_real = D(real)
            d_fake = D(fake.detach())
            gp = gradient_penalty(D, real, fake, device)
            loss_D = d_fake.mean() - d_real.mean() + gp
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # Generator step
            noise2 = torch.randn(bs, G.z_dim, seq_len, device=device)
            fake2 = G(noise2, torch.zeros(bs, device=device, dtype=torch.long))
            loss_G = -D(fake2).mean()
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

        print(f"Epoch {ep}/{epochs} | D Loss: {loss_D.item():.4f} | G Loss: {loss_G.item():.4f}")

# ---------------------------------------------------------------------------
# Sampling & Main
# ---------------------------------------------------------------------------
def sample_one_step(G, loader, device):
    G.eval()
    synth = []
    with torch.no_grad():
        for real in loader:
            bs = real.size(0)
            seq_len = loader.dataset.windows.shape[1]
            noise = torch.randn(bs, G.z_dim, seq_len, device=device)
            fake = G(noise, torch.zeros(bs, device=device, dtype=torch.long))
            synth.append(fake.cpu().numpy())
    return np.concatenate(synth, axis=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',        type=str,   default='../stock_data/stock_data.csv')
    parser.add_argument('--z_dim',       type=int,   default=16)
    parser.add_argument('--epochs',      type=int,   default=300)
    parser.add_argument('--batch',       type=int,   default=32)
    parser.add_argument('--seq_len',     type=int,   default=24)
    parser.add_argument('--metric_iter', type=int,   default=10)
    parser.add_argument('--lr_g',        type=float, default=1e-4)
    parser.add_argument('--lr_d',        type=float, default=1e-4)
    args = parser.parse_args()

    df = pd.read_csv(args.data).dropna()
    vals = df.values.astype(np.float32)
    data = StandardScaler().fit_transform(vals)

    loader = DataLoader(
        TimeSeriesWindowDataset(data, args.seq_len),
        batch_size=args.batch, shuffle=True
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = Generator(vals.shape[1], z_dim=args.z_dim).to(device)
    D = Discriminator(vals.shape[1]).to(device)

    opt_G = optim.Adam(G.parameters(), lr=args.lr_g)
    opt_D = optim.Adam(D.parameters(), lr=args.lr_d)

    train_gan(G, D, loader, device, args.epochs, opt_G, opt_D)

    ori_w = TimeSeriesWindowDataset(vals, args.seq_len).windows
    fake_w = sample_one_step(G, loader, device)

    os.makedirs('out', exist_ok=True)
    visualization(ori_w, fake_w, 'pca', 'out/pca.png')
    visualization(ori_w, fake_w, 'tsne', 'out/tsne.png')

    disc_scores = [
        discriminative_score_metrics(ori_w, fake_w)
        for _ in range(args.metric_iter)
    ]
    _ = [
        predictive_score_metrics(ori_w, fake_w)
        for _ in range(args.metric_iter)
    ]

    idx = df.columns.get_loc('Close')
    real_last = ori_w[:, -1, idx]
    gen_last = fake_w[:, -1, idx]
    res = {
        'DiscriminativeScore': float(np.mean(disc_scores)),
        'MAE': float(mean_absolute_error(real_last, gen_last)),
        'RMSE': float(mean_squared_error(real_last, gen_last, squared=False))
    }
    with open('out/metrics.json', 'w') as f:
        json.dump(res, f, indent=4)

    print("Done. Metrics and plots saved in 'out/'.")

if __name__ == '__main__':
    main()
