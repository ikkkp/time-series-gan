#!/usr/bin/env python3
# coding: utf-8

"""
gen.py

对比真实数据与合成数据的 PCA 和 t-SNE 可视化脚本。

支持两种输入：
1. 原始日度数据 CSV，shape=(T, dim)，需要 --seq_len 参数切分；
2. 已切窗好的 .npy/.npz/.csv，shape=(N, seq_len, dim)。

示例：
  # 原始 CSV，按 60 天窗口切
  python gen.py \
    --real ./stock_data/stock_data.csv \
    --synthetic ./gan_modals/synthetic_cgan_20250422_190346.csv \
    --seq_len 60 \
    --output_dir ./output_plots

  # 已切窗好的 npy 文件
  python gen.py \
    --real real_windows.npy \
    --synthetic synth_windows.npy \
    --output_dir ./output_plots
"""

import argparse
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def load_csv(path, skip_header=True):
    arr = np.loadtxt(path, delimiter=',',
                     skiprows=1 if skip_header else 0).astype(np.float32)
    return arr

def sliding_window(data, seq_len):
    """
    将 2D 原始序列 (T, dim) 切成 (T-seq_len+1, seq_len, dim)
    """
    T, dim = data.shape
    num = T - seq_len + 1
    windows = np.zeros((num, seq_len, dim), dtype=np.float32)
    for i in range(num):
        windows[i] = data[i:i+seq_len]
    return windows

def load_data(path, seq_len=None):
    ext = os.path.splitext(path)[1].lower()
    # 如果是 CSV，先用 load_csv
    if ext == '.csv':
        raw = load_csv(path, skip_header=True)
        # 若已是 3D (N, seq_len, dim) 存为 CSV 展平格式，则需要手动 reshape
        # 但这里假定 CSV 是原始 (T, dim)
        if seq_len is None:
            raise ValueError("输入为原始 CSV 时，必须指定 --seq_len")
        return sliding_window(raw, seq_len)
    # Numpy 二进制格式
    if ext == '.npy':
        arr = np.load(path)
    elif ext == '.npz':
        tmp = np.load(path)
        arr = tmp['arr_0'] if 'arr_0' in tmp else next(iter(tmp.values()))
    else:
        raise ValueError(f"不支持的文件格式：{ext}")
    # 如果已经是 2D，需要 reshape 或滑窗
    if arr.ndim == 2:
        if seq_len is None:
            raise ValueError("输入为 2D 数据时，必须指定 --seq_len")
        return sliding_window(arr, seq_len)
    # 如果是 3D，直接返回
    if arr.ndim == 3:
        return arr.astype(np.float32)
    raise ValueError(f"加载后数组维度不合法：{arr.ndim}")

def visualize(real, synth, output_dir):
    # 计算每条序列的特征均值 -> shape (N, seq_len)
    real_rep  = real.mean(axis=2)
    synth_rep = synth.mean(axis=2)

    combined = np.vstack([real_rep, synth_rep])
    n_real   = real_rep.shape[0]

    os.makedirs(output_dir, exist_ok=True)

    # PCA
    pca = PCA(n_components=2)
    proj = pca.fit_transform(combined)
    pr = proj[:n_real]; ps = proj[n_real:]
    plt.figure()
    plt.scatter(pr[:,0], pr[:,1], label='Real', alpha=0.5, marker='o')
    plt.scatter(ps[:,0], ps[:,1], label='Synthetic', alpha=0.5, marker='x')
    plt.legend(); plt.title('PCA: Real vs. Synthetic')
    plt.xlabel('PC1'); plt.ylabel('PC2')
    plt.savefig(os.path.join(output_dir, 'pca_plot.png'), bbox_inches='tight')
    plt.close()

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
    tsne_proj = tsne.fit_transform(combined)
    trn = tsne_proj[:n_real]; tsn = tsne_proj[n_real:]
    plt.figure()
    plt.scatter(trn[:,0], trn[:,1], label='Real', alpha=0.5, marker='o')
    plt.scatter(tsn[:,0], tsn[:,1], label='Synthetic', alpha=0.5, marker='x')
    plt.legend(); plt.title('t-SNE: Real vs. Synthetic')
    plt.xlabel('Dim 1'); plt.ylabel('Dim 2')
    plt.savefig(os.path.join(output_dir, 'tsne_plot.png'), bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--real',      required=True, help='真实数据文件 (.csv/.npy/.npz)')
    parser.add_argument('--synthetic', required=True, help='合成数据文件 (.csv/.npy/.npz)')
    parser.add_argument('--seq_len',   type=int,   default=None,
                        help='序列长度（滑窗大小），用于原始 2D 输入')
    parser.add_argument('--output_dir', default='results',
                        help='图像输出目录')
    args = parser.parse_args()

    real   = load_data(args.real,   seq_len=args.seq_len)
    synth  = load_data(args.synthetic, seq_len=args.seq_len)
    visualize(real, synth, args.output_dir)
    print(f"PCA 与 t-SNE 图已保存至：{args.output_dir}")

if __name__ == '__main__':
    main()
