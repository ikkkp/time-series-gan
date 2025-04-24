#!/usr/bin/env python3
"""
@File  :timegradGANc.py
@Author:Ezra Zephyr
@Date  :2025/4/23 17:00
@Desc  :Hybrid GAN model integrating diffusion-inspired generator,
    with improved data preprocessing (smoothing, scaling),
    reconstruction loss, learning rate scheduling,
    PCA/t-SNE visualization, extended JSON metrics output,
    with checkpoint loading/saving to 'a' subfolder
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.optim.lr_scheduler import StepLR
from tqdm import trange
from metrics.visualization_metrics import visualization
from metrics.discriminative_metrics import discriminative_score_metrics
from metrics.predictive_metrics import predictive_score_metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error
)

# ---------------------------------------------------------------------------
#  Utility functions
# ---------------------------------------------------------------------------
def dtw_distance(s1, s2):
    n, m = len(s1), len(s2)
    dtw = [[float('inf')] * (m+1) for _ in range(n+1)]
    dtw[0][0] = 0.0
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(s1[i-1] - s2[j-1])
            dtw[i][j] = cost + min(dtw[i-1][j], dtw[i][j-1], dtw[i-1][j-1])
    return dtw[n][m]

def acf_similarity(x, y, max_lag=20):
    def autocorr(arr, lag):
        if lag == 0:
            return 1.0
        return np.corrcoef(arr[:-lag], arr[lag:])[0,1]
    real_acf = [autocorr(x, k) for k in range(max_lag+1)]
    gen_acf  = [autocorr(y, k) for k in range(max_lag+1)]
    return float(np.mean(np.abs(np.array(real_acf) - np.array(gen_acf))))

# ---------------------------------------------------------------------------
#  Model Components
# ---------------------------------------------------------------------------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        emb = torch.exp(-torch.arange(half, device=t.device) * (np.log(10000) / (half - 1)))
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_ch))
        self.res_conv = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()
    def forward(self, x, t_emb):
        h = self.conv1(x) + self.time_mlp(t_emb).unsqueeze(-1)
        h = self.act(h)
        h = self.conv2(h)
        return self.act(h + self.res_conv(x))

class Generator(nn.Module):
    def __init__(self, feat_dim, time_emb_dim=128, hidden_dim=64, num_blocks=4):
        super().__init__()
        self.time_emb    = SinusoidalPosEmb(time_emb_dim)
        self.input_proj  = nn.Conv1d(feat_dim, hidden_dim, 1)
        self.blocks      = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim, time_emb_dim)
            for _ in range(num_blocks)
        ])
        self.output_proj = nn.Conv1d(hidden_dim, feat_dim, 1)
    def forward(self, x, t):
        # x: (batch, seq_len, feat_dim)
        t_emb = self.time_emb(t)
        h = self.input_proj(x.permute(0,2,1))
        for blk in self.blocks:
            h = blk(h, t_emb)
        out = self.output_proj(h)
        return out.permute(0,2,1)

class Discriminator(nn.Module):
    def __init__(self, feat_dim, hidden_dim=64, num_layers=3):
        super().__init__()
        layers = []
        in_ch = feat_dim
        for _ in range(num_layers):
            layers += [
                nn.Conv1d(in_ch, hidden_dim, 3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            in_ch = hidden_dim
        self.cnn = nn.Sequential(*layers)
        self.fc  = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        h = self.cnn(x.permute(0,2,1))
        h = h.mean(dim=2)
        return self.fc(h).squeeze(1)

# ---------------------------------------------------------------------------
#  Utilities & Dataset
# ---------------------------------------------------------------------------
def linear_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, T)

def make_windows(data, seq_len):
    n = len(data) - seq_len + 1
    if n <= 0:
        raise ValueError(f"Data length {len(data)} < seq_len {seq_len}")
    return np.stack([data[i:i+seq_len] for i in range(n)], axis=0)

class TimeSeriesWindowDataset(Dataset):
    def __init__(self, data, seq_len):
        self.windows = make_windows(data, seq_len)
    def __len__(self):
        return len(self.windows)
    def __getitem__(self, idx):
        return torch.tensor(self.windows[idx], dtype=torch.float32)

# ---------------------------------------------------------------------------
#  Training Loop
# ---------------------------------------------------------------------------
def train_diffusion_gan(G, D, loader, device, betas, acp, epochs,
                        gamma_rec, opt_G, opt_D, scheduler_G=None):
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    for ep in range(1, epochs+1):
        for real in loader:
            real = real.to(device)
            noise = torch.randn_like(real)
            t = torch.full((real.size(0),), betas.size(0)-1,
                           device=device, dtype=torch.long)
            fake = G(noise, t)
            # Discriminator
            pred_r = D(real)
            pred_f = D(fake.detach())
            loss_D = bce(pred_r, torch.ones_like(pred_r)) + \
                     bce(pred_f, torch.zeros_like(pred_f))
            opt_D.zero_grad(); loss_D.backward(); opt_D.step()
            # Generator
            pred_f2 = D(fake)
            adv_loss = bce(pred_f2, torch.ones_like(pred_f2))
            rec_loss = mse(fake, real)
            noise_pred = G(fake, t)
            mse_loss = mse(noise_pred, noise)
            loss_G = adv_loss + gamma_rec * rec_loss + 0.1 * mse_loss
            opt_G.zero_grad(); loss_G.backward()
            nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
            opt_G.step()
        if scheduler_G: scheduler_G.step()
        print(f"Epoch {ep}/{epochs} | D Loss: {loss_D.item():.4f} | G Loss: {loss_G.item():.4f}")

# ---------------------------------------------------------------------------
#  Main and Evaluation
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',        type=str, default='../stock_data/stock_data.csv')
    parser.add_argument('--epochs',      type=int, default=50)
    parser.add_argument('--batch',       type=int, default=32)
    parser.add_argument('--T',           type=int, default=100)
    parser.add_argument('--seq_len',     type=int, default=24)
    parser.add_argument('--metric_iter', type=int, default=10)
    parser.add_argument('--out_dir',     type=str,
                        default=r'C:\Users\24964\Desktop\visible_data\time_gradc')
    parser.add_argument('--smooth',      type=int, choices=range(0,11), default=0)
    parser.add_argument('--scale',       choices=['minmax','standard','none'], default='minmax')
    parser.add_argument('--lr_g',        type=float, default=1e-3)
    parser.add_argument('--lr_d',        type=float, default=1e-4)
    parser.add_argument('--step_size',   type=int, default=10)
    parser.add_argument('--gamma_step',  type=float, default=0.5)
    parser.add_argument('--gamma_rec',   type=float, default=0.5)
    args = parser.parse_args()

    # Create output and checkpoint directories
    os.makedirs(args.out_dir, exist_ok=True)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = os.path.join(script_dir, 'time')
    os.makedirs(ckpt_dir, exist_ok=True)
    gen_ckpt = os.path.join(ckpt_dir, 'generator.pth')
    disc_ckpt = os.path.join(ckpt_dir, 'discriminator.pth')

    # Load and preprocess data
    df = pd.read_csv(args.data).dropna()
    vals = df.values.astype(np.float32)
    if args.smooth > 1:
        vals = pd.DataFrame(vals).rolling(window=args.smooth,
                                          min_periods=1,
                                          center=True).mean().values
    if args.scale == 'minmax':
        scaler = MinMaxScaler(feature_range=(-1,1)); data = scaler.fit_transform(vals)
    elif args.scale == 'standard':
        scaler = StandardScaler(); data = scaler.fit_transform(vals)
    else:
        data = vals

    # DataLoader
    loader = DataLoader(TimeSeriesWindowDataset(data, args.seq_len),
                        batch_size=args.batch, shuffle=True)

    # Device & noise schedule
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    betas  = linear_beta_schedule(args.T).to(device)
    acp    = torch.cumprod(1 - betas, dim=0)

    # Models, optimizers, scheduler
    G = Generator(feat_dim=data.shape[1]).to(device)
    D = Discriminator(feat_dim=data.shape[1]).to(device)
    opt_G = optim.Adam(G.parameters(), lr=args.lr_g)
    opt_D = optim.Adam(D.parameters(), lr=args.lr_d)
    scheduler_G = StepLR(opt_G, step_size=args.step_size, gamma=args.gamma_step)

    # Load existing or train new
    if os.path.isfile(gen_ckpt) and os.path.isfile(disc_ckpt):
        print("Loading checkpoints from 'a' directory...")
        G.load_state_dict(torch.load(gen_ckpt, map_location=device))
        D.load_state_dict(torch.load(disc_ckpt, map_location=device))
    else:
        train_diffusion_gan(G, D, loader, device,
                            betas, acp,
                            args.epochs, args.gamma_rec,
                            opt_G, opt_D, scheduler_G)
        torch.save(G.state_dict(), gen_ckpt)
        torch.save(D.state_dict(), disc_ckpt)
        print(f"Saved checkpoints to {ckpt_dir}")

    # Generate windows for evaluation
    G.eval()
    ori_w = make_windows(data, args.seq_len)
    fake_w = []
    for real in loader:
        real = real.to(device)
        noise = torch.randn_like(real)
        t = torch.full((noise.size(0),), betas.size(0)-1,
                       dtype=torch.long, device=device)
        fake_w.append(G(noise, t).detach().cpu().numpy())
    gen_w = np.concatenate(fake_w, axis=0)

    # Visualization
    visualization(ori_w, gen_w, 'pca', os.path.join(args.out_dir, 'pca_plot.png'))
    visualization(ori_w, gen_w, 'tsne', os.path.join(args.out_dir, 'tsne_plot.png'))
    print("Plots saved to", args.out_dir)

    # Compute metrics
    print("Calculating TimeGAN metrics...")
    disc = [
        discriminative_score_metrics(ori_w, gen_w)
        for _ in trange(args.metric_iter, desc="DiscriminativeScore")
    ]
    _ = [
        predictive_score_metrics(ori_w, gen_w)
        for _ in trange(args.metric_iter, desc="PredictiveScore")
    ]

    # Last-step 'Close' metrics
    close_idx = df.columns.tolist().index('Close')
    real_last = ori_w[:, -1, close_idx]
    gen_last  = gen_w[:, -1, close_idx]

    pred_mae  = mean_absolute_error(real_last, gen_last)
    pred_rmse = mean_squared_error(real_last, gen_last, squared=False)
    dtw_val   = dtw_distance(real_last.tolist(), gen_last.tolist())
    acf_val   = acf_similarity(real_last, gen_last)

    real_move = (real_last[1:] > real_last[:-1]).astype(int)
    gen_move  = (gen_last[1:]  > gen_last[:-1]).astype(int)
    cls_metrics = {
        "Accuracy":  accuracy_score(real_move, gen_move),
        "Precision": precision_score(real_move, gen_move, zero_division=0),
        "Recall":    recall_score(real_move, gen_move, zero_division=0),
        "F1Score":   f1_score(real_move, gen_move, zero_division=0)
    }

    mape = np.mean(np.abs((real_last - gen_last) / (real_last + 1e-8))) * 100
    reg_metrics = {
        "MAE":  float(pred_mae),
        "RMSE": float(pred_rmse),
        "MAPE": f"{mape:.2f}%"
    }

    results = {
        "general_metrics": {
            "IS":  0.0,
            "FID": 0.0,
            "KID": 0.0
        },
        "time_series_metrics": {
            "DiscriminativeScore": float(np.mean(disc)),
            "PredictiveMAE":       float(pred_mae),
            "PredictiveRMSE":      float(pred_rmse),
            "DTW":                 float(dtw_val),
            "ACFSimilarity":       float(acf_val)
        },
        "classification_metrics": cls_metrics,
        "regression_metrics":     reg_metrics
    }

    json_path = os.path.join(args.out_dir, 'metrics.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print("Metrics saved to", json_path)

if __name__ == '__main__':
    main()
