"""
@File  :yahoo.py
@Author:Ezra Zephyr
@Date  :2025/4/14 14:32
@Desc  :TimeGAN architecture example file with main function
"""

import os
from os import path
# 如果只想用第 0 号 GPU，可取消下一行的注释
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
print(tf.__version__)
print("GPUs:", tf.config.list_physical_devices('GPU'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from ydata_synthetic.synthesizers import ModelParameters
from ydata_synthetic.preprocessing.timeseries import processed_stock
from ydata_synthetic.synthesizers.timeseries import TimeGAN


def main():
    # 超参数
    seq_len = 24
    n_seq = 7
    hidden_dim = 24
    gamma = 1

    noise_dim = 32
    dim = 128
    batch_size = 128

    learning_rate = 5e-4
    train_steps = 10000

    # 1) 数据预处理：从 CSV 中滑窗切割
    #    processed_stock 会按顺序把每 24 步长切出一条序列，每条序列维度为 (24, 7)
    stock_data = processed_stock(path='../stock_data/processed_stocks.csv',
                                 seq_len=seq_len)
    print("序列数量和每条序列形状:", len(stock_data), stock_data[0].shape)
    # 这里 stock_data[i].shape == (24, 7)

    # 2) 构造模型参数
    gan_args = ModelParameters(batch_size=batch_size,
                               lr=learning_rate,
                               noise_dim=noise_dim,
                               layers_dim=dim)

    # 为了不加载旧模型，我们换一个新文件名
    model_file = 'synthesizer_stock_7feat.pkl'

    # 3) 加载或训练模型
    if path.exists(model_file):
        print(f"检测到已保存模型 '{model_file}'，直接加载 …")
        synth = TimeGAN.load(model_file)
    else:
        print("未检测到已保存模型，开始新一轮训练 …")
        synth = TimeGAN(model_parameters=gan_args,
                        hidden_dim=hidden_dim,
                        seq_len=seq_len,
                        n_seq=n_seq,
                        gamma=gamma)
        synth.train(stock_data, train_steps=train_steps)
        synth.save(model_file)
        print(f"模型训练完毕并已保存到 '{model_file}'")

    # 4) 生成合成样本
    synth_data = synth.sample(len(stock_data))
    print("合成样本数组形状:", np.array(synth_data).shape)
    # e.g. (5401, 24, 7)

    # 5) 可视化若干对比
    cols = ["timestamp", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "PriceRange"]
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 10))
    axes = axes.flatten()
    obs = np.random.randint(len(stock_data))

    for j, col in enumerate(cols):
        df = pd.DataFrame({
            'Real':      stock_data[obs][:, j],
            'Synthetic': synth_data[obs][:, j]
        })
        df.plot(ax=axes[j],
                title=col,
                secondary_y='Synthetic',
                style=['-', '--'])
    fig.tight_layout()
    plt.show()

    # 6) PCA & t-SNE 分布对比
    sample_size = 250
    idx = np.random.permutation(len(stock_data))[:sample_size]
    real_sample      = np.asarray(stock_data)[idx]      # (250, 24, 7)
    synthetic_sample = np.asarray(synth_data)[idx]      # (250, 24, 7)

    # 展平到 (250*24, 7) 维，再做降到 2 维度
    real_flat = real_sample.reshape(-1, seq_len)        # (250*24, 24)
    synth_flat= synthetic_sample.reshape(-1, seq_len)   # (250*24, 24)

    pca  = PCA(n_components=2)
    tsne = TSNE(n_components=2, n_iter=300)

    pca.fit(real_flat)
    pca_real  = pca.transform(real_flat)
    pca_synth = pca.transform(synth_flat)

    concat = np.concatenate([real_flat, synth_flat], axis=0)
    tsne_results = tsne.fit_transform(concat)

    # 绘图
    fig = plt.figure(constrained_layout=True, figsize=(20, 10))
    spec = gridspec.GridSpec(1, 2, figure=fig)

    ax1 = fig.add_subplot(spec[0, 0])
    ax1.set_title('PCA Results', fontsize=20)
    ax1.scatter(pca_real[:,0], pca_real[:,1], alpha=0.2, label='Real')
    ax1.scatter(pca_synth[:,0], pca_synth[:,1], alpha=0.2, label='Synth')
    ax1.legend()

    ax2 = fig.add_subplot(spec[0, 1])
    ax2.set_title('t-SNE Results', fontsize=20)
    ax2.scatter(tsne_results[:len(real_flat),0],
                tsne_results[:len(real_flat),1],
                alpha=0.2, label='Real')
    ax2.scatter(tsne_results[len(real_flat):,0],
                tsne_results[len(real_flat):,1],
                alpha=0.2, label='Synth')
    ax2.legend()

    fig.suptitle('Real vs Synthetic Distributions', fontsize=16)
    plt.show()


if __name__ == '__main__':
    main()
