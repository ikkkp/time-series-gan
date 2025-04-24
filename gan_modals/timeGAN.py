import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from tqdm import trange

# For reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# ─── 1. 网络定义 ────────────────────────────────────────────────────────────
class Embedder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc  = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, x):
        h, _ = self.rnn(x)
        return torch.tanh(self.fc(h))

class Recovery(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc  = nn.Linear(hidden_dim, output_dim)
    def forward(self, h):
        h2, _ = self.rnn(h)
        # Linear output for arbitrary-scale features
        return self.fc(h2)

class Generator(nn.Module):
    def __init__(self, noise_dim, hidden_dim):
        super().__init__()
        self.rnn = nn.GRU(noise_dim, hidden_dim, batch_first=True)
        self.fc  = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, z):
        h, _ = self.rnn(z)
        return torch.tanh(self.fc(h))

class Supervisor(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc  = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, h):
        h2, _ = self.rnn(h)
        return torch.tanh(self.fc(h2))

class Discriminator(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc  = nn.Linear(hidden_dim, 1)
    def forward(self, h):
        h2, _ = self.rnn(h)
        last = h2[:, -1, :]
        return self.fc(last)

# ─── 2. 损失函数 ───────────────────────────────────────────────────────────
def reconstruction_loss(x, x_tilde):
    return nn.MSELoss()(x_tilde, x)

def supervised_loss(h_real, h_pred):
    # Teach supervisor to predict next-step in hidden space
    return nn.MSELoss()(h_pred[:, :-1, :], h_real[:, 1:, :])

# ─── 3. 训练流程 ───────────────────────────────────────────────────────────
def train_timegan(data_windows,
                  E, R, G, S, D,
                  hidden_dim, noise_dim,
                  batch_size,
                  epochs_pretrain,
                  epochs_supervise,
                  epochs_adversarial,
                  lr,
                  gamma,
                  device):
    optim_ae = optim.Adam(list(E.parameters()) + list(R.parameters()), lr=lr)
    optim_s  = optim.Adam(S.parameters(), lr=lr)
    optim_d  = optim.Adam(D.parameters(), lr=lr)
    optim_g  = optim.Adam(list(G.parameters()) + list(S.parameters()), lr=lr)

    arr = np.stack(data_windows)  # shape (Nw, seq_len, feat_dim)
    ds  = TensorDataset(torch.tensor(arr, dtype=torch.float32))
    dl  = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    # 1) 自编码器预训练
    pbar = trange(epochs_pretrain, desc="AE pretrain")
    for _ in pbar:
        tot = 0.0
        for (batch,) in dl:
            batch = batch.to(device)
            loss  = reconstruction_loss(batch, R(E(batch)))
            optim_ae.zero_grad()
            loss.backward()
            optim_ae.step()
            tot += loss.item()
        pbar.set_postfix(loss=tot/len(dl))

    # 2) Supervisor 预训练
    pbar = trange(epochs_supervise, desc="Sup pretrain")
    for _ in pbar:
        tot = 0.0
        for (batch,) in dl:
            batch = batch.to(device)
            with torch.no_grad():
                h = E(batch)
            loss = supervised_loss(h, S(h))
            optim_s.zero_grad()
            loss.backward()
            optim_s.step()
            tot += loss.item()
        pbar.set_postfix(loss=tot/len(dl))

    # 3) 对抗训练
    bce  = nn.BCEWithLogitsLoss()
    pbar = trange(epochs_adversarial, desc="Adv train")
    for _ in pbar:
        d_tot = g_tot = 0.0
        for (batch,) in dl:
            batch = batch.to(device)

            # 判别器: freeze E
            with torch.no_grad():
                h_real = E(batch)
            log_real = D(h_real)
            z        = torch.randn(batch.size(0), batch.size(1), noise_dim, device=device)
            h_fake   = G(z)
            h_sup    = S(h_fake)
            log_fake = D(h_sup.detach())
            d_loss   = bce(log_real, torch.ones_like(log_real)) \
                       + bce(log_fake, torch.zeros_like(log_fake))
            optim_d.zero_grad()
            d_loss.backward()
            optim_d.step()
            d_tot  += d_loss.item()

            # 生成器 + 监督器
            log_fake2 = D(h_sup)
            g_loss    = bce(log_fake2, torch.ones_like(log_fake2))
            # corrected supervised loss
            g_loss   += gamma * supervised_loss(h_fake, h_sup)
            g_loss   += 100 * reconstruction_loss(batch, R(h_sup))
            optim_g.zero_grad()
            g_loss.backward()
            optim_g.step()
            g_tot   += g_loss.item()

        pbar.set_postfix(D_loss=d_tot/len(dl), G_loss=g_tot/len(dl))

# ─── 4. 主流程 ─────────────────────────────────────────────────────────────
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    # 1) 读原始 CSV，分离 timestamp 和数值
    raw_df        = pd.read_csv('../stock_data/processed_stocks.csv')
    timestamps    = raw_df['timestamp'].values
    feature_names = [c for c in raw_df.columns if c != 'timestamp']
    arr_values    = raw_df[feature_names].values.astype(np.float32)
    num_steps, feat_dim = arr_values.shape
    seq_len = 24

    # 2) 构造滑窗数据列表
    data_windows = [arr_values[i:i+seq_len] for i in range(num_steps - seq_len + 1)]
    print(f"Loaded {len(data_windows)} sliding windows, each of shape ({seq_len}, {feat_dim})")

    # 3) 网络实例化
    hidden_dim = 24
    noise_dim  = 32
    E = Embedder(feat_dim, hidden_dim).to(device)
    R = Recovery(hidden_dim, feat_dim).to(device)
    G = Generator(noise_dim, hidden_dim).to(device)
    S = Supervisor(hidden_dim).to(device)
    D = Discriminator(hidden_dim).to(device)

    # 4) 从零训练
    print("Training TimeGAN from scratch...")
    train_timegan(
        data_windows, E, R, G, S, D,
        hidden_dim=hidden_dim,
        noise_dim=noise_dim,
        batch_size=128,
        epochs_pretrain=1000,
        epochs_supervise=500,
        epochs_adversarial=1000,
        lr=5e-4,
        gamma=1,
        device=device
    )
    print("Training complete.")

    # 5) 分块生成合成窗口
    Nw = len(data_windows)
    with torch.no_grad():
        z       = torch.randn(Nw, seq_len, noise_dim, device=device)
        h_fake  = G(z)
        h_sup   = S(h_fake)
        windows = R(h_sup).cpu().numpy()  # shape (Nw, seq_len, feat_dim)

    # 6) 拼接回完整序列
    full = []
    full.extend(windows[0])            # 第一个窗口全部
    for w in windows[1:]:
        full.append(w[-1])             # 后续只要最后一步
    full = np.array(full)
    assert full.shape[0] == num_steps, \
        f"拼接后长度 {full.shape[0]} != 原始行数 {num_steps}"

    # 7) 导出成 CSV
    out_df = pd.DataFrame(full, columns=feature_names)
    out_df.insert(0, 'timestamp', timestamps)
    out_file = 'timegan_full_length_synth.csv'
    out_df.to_csv(out_file, index=False)
    print(f"Full-length synthetic data saved to {out_file}")

if __name__ == '__main__':
    main()
