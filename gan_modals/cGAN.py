#!/usr/bin/env python3
import os
import json
import datetime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error
)
import argparse
from tqdm import tqdm, trange

# TimeGAN-style metrics
from metrics.discriminative_metrics import discriminative_score_metrics
from metrics.predictive_metrics import predictive_score_metrics
# 可视化函数
from metrics.visualization_metrics import visualization

# Reconstruction loss weight
ALPHA = 0.5

# ---------------------------------------------------------------------------
#  Utilities
# ---------------------------------------------------------------------------
def weights_init(m):
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def make_windows(data: np.ndarray, seq_len: int) -> np.ndarray:
    n = len(data) - seq_len + 1
    if n <= 0:
        raise ValueError(f"Data length {len(data)} < seq_len {seq_len}")
    return np.stack([data[i:i+seq_len] for i in range(n)], axis=0)

class TimeSeriesDataset(Dataset):
    """
    Dataset that returns (prev_step, next_step) pairs for one-step prediction.
    """
    def __init__(self, data: np.ndarray):
        self.x = data[:-1]
        self.y = data[1:]
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return (
            torch.tensor(self.x[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32)
        )

# ---------------------------------------------------------------------------
#  Models
# ---------------------------------------------------------------------------
class Generator(nn.Module):
    def __init__(self, noise_dim, feature_dim):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=noise_dim + feature_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        self.norm = nn.LayerNorm(128)
        self.fc = nn.Sequential(
            nn.Linear(128, feature_dim),
            nn.Tanh()
        )
    def forward(self, noise, prev):
        # noise: (B, noise_dim), prev: (B, feature_dim)
        x = torch.cat([noise, prev], dim=1).unsqueeze(1)
        h, _ = self.rnn(x)
        return self.fc(self.norm(h).squeeze(1))

class Discriminator(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=feature_dim * 2,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        self.norm = nn.LayerNorm(128)
        self.fc_adv = nn.Linear(128, 1)
    def forward(self, candidate, prev):
        # candidate & prev: (B, feature_dim)
        x = torch.cat([candidate, prev], dim=1).unsqueeze(1)
        h, _ = self.rnn(x)
        h = self.norm(h.squeeze(1))
        return self.fc_adv(h)

# ---------------------------------------------------------------------------
#  Training loops
# ---------------------------------------------------------------------------
def pretrain_discriminator(D, G, dataloader, opt_D, adv_criterion,
                           device, epochs, noise_dim):
    for ep in range(epochs):
        loop = tqdm(dataloader, desc=f"Pre-D [epoch {ep+1}/{epochs}]", leave=False)
        for prev, real_next in loop:
            prev, real_next = prev.to(device), real_next.to(device)
            B = real_next.size(0)
            real_lbl = torch.full((B,1), 0.9, device=device)
            fake_lbl = torch.zeros((B,1), device=device)

            D.zero_grad()
            # real
            adv_r = D(real_next, prev)
            loss_r = adv_criterion(adv_r, real_lbl)
            # fake
            with torch.no_grad():
                fake_next = G(torch.randn(B, noise_dim, device=device), prev)
            adv_f = D(fake_next, prev)
            loss_f = adv_criterion(adv_f, fake_lbl)

            loss = 0.5 * (loss_r + loss_f)
            loss.backward()
            opt_D.step()
            loop.set_postfix(loss=loss.item())

def pretrain_generator(G, D, dataloader, opt_G, adv_criterion,
                       device, epochs, noise_dim):
    for ep in range(epochs):
        loop = tqdm(dataloader, desc=f"Pre-G [epoch {ep+1}/{epochs}]", leave=False)
        for prev, _ in loop:
            prev = prev.to(device)
            B = prev.size(0)
            real_lbl = torch.full((B,1), 0.9, device=device)

            G.zero_grad()
            gen_next = G(torch.randn(B, noise_dim, device=device), prev)
            adv_g = D(gen_next, prev)
            loss = adv_criterion(adv_g, real_lbl)
            loss.backward()
            opt_G.step()
            loop.set_postfix(loss=loss.item())

def train_and_evaluate(
    data_path, epochs, batch_size, noise_dim,
    lr, metric_iter, seq_len,
    pretrain_d, pretrain_g, device,
    out_dir="output"
):
    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    gen_ckpt  = os.path.join(ckpt_dir, "generator.pth")
    disc_ckpt = os.path.join(ckpt_dir, "discriminator.pth")

    # Load and scale data
    df     = pd.read_csv(data_path)
    cols   = df.columns.tolist()
    values = df.values.astype(np.float32)
    scaler = MinMaxScaler(feature_range=(-1,1))
    normed = scaler.fit_transform(values)

    # Dataset & Dataloader
    ds = TimeSeriesDataset(normed)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    # Models
    feat_dim = values.shape[1]
    G = Generator(noise_dim, feat_dim).to(device)
    D = Discriminator(feat_dim).to(device)
    G.apply(weights_init); D.apply(weights_init)

    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5,0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5,0.999))
    adv_criterion   = nn.BCEWithLogitsLoss()
    recon_criterion = nn.MSELoss()

    # Pretrain
    if pretrain_d > 0:
        pretrain_discriminator(D, G, dl, opt_D, adv_criterion, device, pretrain_d, noise_dim)
    if pretrain_g > 0:
        pretrain_generator(G, D, dl, opt_G, adv_criterion, device, pretrain_g, noise_dim)

    # Adversarial training
    for ep in range(epochs):
        loop = tqdm(dl, desc=f"Adv [epoch {ep+1}/{epochs}]", leave=False)
        for prev, real_next in loop:
            prev, real_next = prev.to(device), real_next.to(device)
            B = real_next.size(0)
            real_lbl = torch.full((B,1), 0.9, device=device)
            fake_lbl = torch.zeros((B,1), device=device)

            # Train D
            D.zero_grad()
            adv_r = D(real_next, prev)
            loss_dr = adv_criterion(adv_r, real_lbl)
            with torch.no_grad():
                fake_next = G(torch.randn(B, noise_dim, device=device), prev)
            adv_f = D(fake_next, prev)
            loss_df = adv_criterion(adv_f, fake_lbl)
            loss_D = 0.5 * (loss_dr + loss_df)
            loss_D.backward()
            opt_D.step()

            # Train G
            G.zero_grad()
            gen_next = G(torch.randn(B, noise_dim, device=device), prev)
            adv_g = D(gen_next, prev)
            loss_adv = adv_criterion(adv_g, real_lbl)
            loss_recon = recon_criterion(gen_next, real_next)
            loss_G = loss_adv + ALPHA * loss_recon
            loss_G.backward()
            opt_G.step()

            loop.set_postfix(loss_D=loss_D.item(), loss_G=loss_G.item())

    # Save checkpoints
    torch.save(G.state_dict(), gen_ckpt)
    torch.save(D.state_dict(), disc_ckpt)

    # Generate synthetic sequence
    synth, cur = [], normed[0]
    synth.append(cur)
    for _ in range(len(normed) - 1):
        cur = G(torch.randn(1, noise_dim, device=device),
                torch.tensor(cur, device=device).unsqueeze(0)
               ).detach().cpu().numpy()[0]
        synth.append(cur)
    synth = np.vstack(synth)
    denorm = scaler.inverse_transform(synth)

    # Save CSV
    out_csv = os.path.join(out_dir, 'synthetic_cgan.csv')
    pd.DataFrame(denorm, columns=cols).to_csv(out_csv, index=False)
    print(f"Synthetic CSV saved as '{out_csv}'")

    # Windowed data for metrics & plots
    ori_w = make_windows(values, seq_len)
    gen_w = make_windows(denorm, seq_len)

    # Visualizations
    vis_pca = os.path.join(out_dir, 'pca_plot.png')
    vis_tsne = os.path.join(out_dir, 'tsne_plot.png')
    visualization(ori_w, gen_w, 'pca', vis_pca)
    visualization(ori_w, gen_w, 'tsne', vis_tsne)
    print("PCA & t-SNE plots saved.")

    # Metrics computation
    disc_scores = [
        discriminative_score_metrics(ori_w, gen_w)
        for _ in trange(metric_iter, desc="DiscriminativeScore", leave=False)
    ]
    pred_scores = [
        predictive_score_metrics(ori_w, gen_w)
        for _ in trange(metric_iter, desc="PredictiveScore", leave=False)
    ]
    ts_metrics = {
        "DiscriminativeScore": float(np.mean(disc_scores)),
        "PredictiveScore":     float(np.mean(pred_scores))
    }

    # DTW & ACF
    close_i = cols.index('Close') if 'Close' in cols else 0
    real_c = values[:, close_i]
    syn_c  = denorm[:, close_i]
    from scipy.spatial.distance import euclidean
    dtw_dist = None  # keep your custom DTW if needed
    acf_sim  = None  # keep your custom ACF if needed
    ts_metrics.update({"DTW": dtw_dist, "ACFSimilarity": acf_sim})

    # Classification & regression
    r_lbl = (real_c[1:] > real_c[:-1]).astype(int)
    s_lbl = (syn_c[1:] > syn_c[:-1]).astype(int)
    cls_m = {
        "Accuracy":  float(accuracy_score(r_lbl, s_lbl)),
        "Precision": float(precision_score(r_lbl, s_lbl, zero_division=0)),
        "Recall":    float(recall_score(r_lbl, s_lbl, zero_division=0)),
        "F1Score":   float(f1_score(r_lbl, s_lbl, zero_division=0))
    }
    mae_val  = float(mean_absolute_error(real_c, syn_c))
    rmse_val = float(mean_squared_error(real_c, syn_c, squared=False))
    eps = 1e-8
    mape_val = float(np.mean(np.abs((real_c - syn_c) / (real_c + eps))) * 100)
    reg_m = {"MAE": mae_val, "RMSE": rmse_val, "MAPE": f"{mape_val:.2f}%"}

    # Save metrics JSON
    out_json = os.path.join(out_dir, "cgan_metrics.json")
    results = {
        "time_series_metrics":     ts_metrics,
        "classification_metrics":  cls_m,
        "regression_metrics":      reg_m
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"Metrics saved to '{out_json}'")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--data',       type=str,   default='../stock_data/stock_data.csv', help="Path to CSV data file")
    p.add_argument('--pretrain_d', type=int,   default=20,  help="epochs to pretrain D")
    p.add_argument('--pretrain_g', type=int,   default=20,  help="epochs to pretrain G")
    p.add_argument('--epochs',     type=int,   default=200, help="adversarial training epochs")
    p.add_argument('--batch',      type=int,   default=64)
    p.add_argument('--noise',      type=int,   default=10)
    p.add_argument('--lr',         type=float, default=1e-5)
    p.add_argument('--metric_iter',type=int,   default=10)
    p.add_argument('--seq_len',    type=int,   default=24)
    p.add_argument('--out_dir',    type=str,   default=r"C:\Users\24964\Desktop\visible_data\c_gan", help="Directory for all outputs")
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_and_evaluate(
        args.data, args.epochs, args.batch, args.noise, args.lr,
        args.metric_iter, args.seq_len, args.pretrain_d, args.pretrain_g,
        device, args.out_dir
    )
