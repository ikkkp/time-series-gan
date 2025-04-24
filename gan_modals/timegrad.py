#!/usr/bin/env python3
"""
TimeGrad-like Diffusion Model for Multivariate Time Series Generation
    + sequence sampling with tqdm progress bar
    + classification & regression metrics (incl. F1)
    + PCA & t-SNE 可视化
    + TimeGAN 指标计算
    * Fixed: use torch.no_grad() + detach() before .numpy()
    * Added: model checkpointing and JSON metrics output to a custom folder
"""
import os
import json
import math
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import trange
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error
)
from metrics.visualization_metrics import visualization
from metrics.discriminative_metrics import discriminative_score_metrics
from metrics.predictive_metrics import predictive_score_metrics

# *** Helper to create 3D windows for visualization ***
def make_windows(data: np.ndarray, seq_len: int) -> np.ndarray:
    n = len(data) - seq_len + 1
    if n <= 0:
        raise ValueError(f"Data length {len(data)} < seq_len {seq_len}")
    return np.stack([data[i:i+seq_len] for i in range(n)], axis=0)

# Diffusion utilities
def linear_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, T)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        emb = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
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

class TimeSeriesDenoise(nn.Module):
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
        x = x.permute(0,2,1)
        t_emb = self.time_emb(t)
        h = self.input_proj(x)
        for blk in self.blocks:
            h = blk(h, t_emb)
        out = self.output_proj(h)
        return out.permute(0,2,1)

class TimeSeriesDataset(Dataset):
    def __init__(self, data_arr):
        self.seq = data_arr.astype(np.float32)
    def __len__(self):
        return len(self.seq)
    def __getitem__(self, idx):
        return torch.tensor(self.seq[idx], dtype=torch.float32)

# Training loop
def train(model, loader, optimizer, device, betas, alphas_cumprod, epochs):
    mse = nn.MSELoss()
    T = len(betas)
    for ep in range(1, epochs+1):
        for x0 in loader:
            x0 = x0.to(device)
            B, feat = x0.shape
            t = torch.randint(0, T, (B,), device=device)
            a_cum = alphas_cumprod[t].unsqueeze(-1)
            sqrt_acp = torch.sqrt(a_cum)
            sqrt_1_acp = torch.sqrt(1 - a_cum)
            noise = torch.randn_like(x0)
            x_noisy = sqrt_acp * x0 + sqrt_1_acp * noise
            pred_noise = model(x_noisy.unsqueeze(1), t).squeeze(1)
            loss = mse(pred_noise, noise)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        print(f"Epoch {ep}/{epochs}  loss={loss.item():.4f}")

# Sampling
def sample_sequence(model, device, betas, alphas_cumprod, seq_len, init_vec):
    model.eval()
    seq = [init_vec]
    x_prev = torch.tensor(init_vec, dtype=torch.float32, device=device)
    x_prev = x_prev.unsqueeze(0).unsqueeze(1)
    with torch.no_grad():
        for _ in trange(seq_len-1, desc="Sampling sequence", unit="step"):
            x = x_prev
            for t in reversed(range(len(betas))):
                t_batch = torch.tensor([t], device=device)
                pred = model(x, t_batch)
                alpha_t = 1 - betas[t]
                acp_t   = alphas_cumprod[t]
                coef1   = 1/math.sqrt(alpha_t.item())
                coef2   = betas[t].item()/math.sqrt((1-acp_t).item())
                x = coef1*(x - coef2*pred)
                if t>0:
                    x = x + math.sqrt(betas[t].item()) * torch.randn_like(x)
            vec = x.squeeze(0).squeeze(0).cpu().numpy()
            seq.append(vec); x_prev = x
    return np.stack(seq, axis=0)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',        type=str,   default='../stock_data/stock_data.csv')
    parser.add_argument('--epochs',      type=int,   default=1200)
    parser.add_argument('--batch',       type=int,   default=64)
    parser.add_argument('--beta_start',  type=float, default=1e-4)
    parser.add_argument('--beta_end',    type=float, default=0.02)
    parser.add_argument('--T',           type=int,   default=1000)
    parser.add_argument('--metric_iter', type=int,   default=10)
    parser.add_argument('--seq_len',     type=int,   default=24)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    df = pd.read_csv(args.data)
    data_arr = df.values
    N, feat_dim = data_arr.shape

    ds = TimeSeriesDataset(data_arr)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True)
    betas = linear_beta_schedule(args.T, args.beta_start, args.beta_end).to(device)
    acp = torch.cumprod(1 - betas, dim=0)

    # Model & checkpoint
    ckpt_dir = 'checkpoints_diffusion'
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_file = os.path.join(ckpt_dir, 'diffusion_model.pth')
    model = TimeSeriesDenoise(feat_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    if os.path.isfile(ckpt_file):
        print("Loading saved diffusion model...")
        model.load_state_dict(torch.load(ckpt_file, map_location=device))
    else:
        train(model, loader, opt, device, betas, acp, args.epochs)
        torch.save(model.state_dict(), ckpt_file)
        print(f"Saved diffusion model checkpoint to {ckpt_file}")

    # Sampling & CSV
    synth_seq = sample_sequence(model, device, betas, acp, N, data_arr[0])
    pd.DataFrame(synth_seq, columns=df.columns).to_csv('synthetic_diffusion.csv', index=False)
    print("Synthetic CSV saved as 'synthetic_diffusion.csv'")

    # Visualization: windowed
    ori_w = make_windows(data_arr, args.seq_len)
    gen_w = make_windows(synth_seq, args.seq_len)
    visualization(ori_w, gen_w, 'pca', r'C:\Users\24964\Desktop\visible_data\time_grad\pca_plot.png')
    visualization(ori_w, gen_w, 'tsne', r'C:\Users\24964\Desktop\visible_data\time_grad\tsne_plot.png')
    print("PCA & t-SNE plots saved.")

    # TimeGAN metrics
    disc_scores = [discriminative_score_metrics(ori_w, gen_w) for _ in range(args.metric_iter)]
    pred_scores = [predictive_score_metrics(ori_w, gen_w) for _ in range(args.metric_iter)]
    ts_metrics = {
        'DiscriminativeScore': float(np.mean(disc_scores)),
        'PredictiveScore':     float(np.mean(pred_scores))
    }
    print("TimeGAN metrics:", ts_metrics)

    # Classification & regression
    close_idx = list(df.columns).index('Close')
    real_c = data_arr[:, close_idx]
    syn_c = synth_seq[:, close_idx]
    r_lbl = (real_c[1:] > real_c[:-1]).astype(int)
    s_lbl = (syn_c[1:] > syn_c[:-1]).astype(int)
    cls_m = {
        'Accuracy':  accuracy_score(r_lbl, s_lbl),
        'Precision': precision_score(r_lbl, s_lbl, zero_division=0),
        'Recall':    recall_score(r_lbl, s_lbl, zero_division=0),
        'F1Score':   f1_score(r_lbl, s_lbl, zero_division=0)
    }
    mae_v = mean_absolute_error(real_c, syn_c)
    rmse_v = mean_squared_error(real_c, syn_c, squared=False)
    mape_v = np.mean(np.abs((real_c - syn_c) / (real_c + 1e-8))) * 100
    print("Classification metrics:", cls_m)
    print("Regression metrics:", {'MAE': mae_v, 'RMSE': rmse_v, 'MAPE': f"{mape_v:.2f}%"})

    # JSON dump aligned with TimeGAN format
    out_dir = r'C:\Users\24964\Desktop\visible_data\time_grad'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'diffusion_metrics.json')
    results = {
        'general_metrics':      {'IS':0.0, 'FID':0.0, 'KID':0.0},
        'time_series_metrics':  ts_metrics,
        'classification_metrics': cls_m,
        'regression_metrics': {
            'MAE':  mae_v,
            'RMSE': rmse_v,
            'MAPE': f"{mape_v:.2f}%"
        }
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"Metrics saved to '{out_path}'")
