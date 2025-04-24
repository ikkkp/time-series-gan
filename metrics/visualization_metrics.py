# Necessary packages
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import os


def visualization(ori_data, generated_data, analysis, save_path=None):
    """Using PCA or t-SNE for generated and original data visualization.

    Args:
      - ori_data: original data, shape (num_windows, seq_len, dim)
      - generated_data: generated synthetic data, same shape
      - analysis: 'pca' or 'tsne'
      - save_path: full file path where plot will be saved. If None, defaults to 'results/{analysis}_plot.png'.
    """
    # convert to numpy arrays
    ori_data = np.asarray(ori_data)
    generated_data = np.asarray(generated_data)

    # sample for speed
    anal_sample_no = min(1000, len(ori_data))
    idx = np.random.permutation(len(ori_data))[:anal_sample_no]
    ori_subset = ori_data[idx]
    gen_subset = generated_data[idx]

    # extract dimensions
    no, seq_len, dim = ori_subset.shape

    # compute mean over features for each window
    prep_data = ori_subset.mean(axis=2)
    prep_data_hat = gen_subset.mean(axis=2)

    # —— 新增：删除含 NaN 的窗口 ——
    valid_mask = np.all(np.isfinite(prep_data_hat), axis=1)
    prep_data = prep_data[valid_mask]
    prep_data_hat = prep_data_hat[valid_mask]
    # —————————————————————————————

    # choose file path
    if save_path:
        file_path = save_path
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    else:
        save_dir = "results"
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f"{analysis}_plot.png")

    # PCA branch
    if analysis.lower() == 'pca':
        pca = PCA(n_components=2)
        pts1 = pca.fit_transform(prep_data)
        pts2 = pca.transform(prep_data_hat)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(pts1[:, 0], pts1[:, 1], c='red', alpha=0.2, label="Original")
        ax.scatter(pts2[:, 0], pts2[:, 1], c='blue', alpha=0.2, label="Synthetic")
        ax.legend()
        ax.set_title('PCA plot')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')

        fig.savefig(file_path, bbox_inches='tight')
        print(f"PCA plot saved as {file_path}")
        plt.show()
        plt.close(fig)

    # t-SNE branch
    elif analysis.lower() == 'tsne':
        combined = np.concatenate((prep_data, prep_data_hat), axis=0)
        tsne = TSNE(n_components=2, perplexity=40, n_iter=300, verbose=0)
        tsne_res = tsne.fit_transform(combined)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(tsne_res[:len(prep_data), 0], tsne_res[:len(prep_data), 1],
                   c='red', alpha=0.2, label="Original")
        ax.scatter(tsne_res[len(prep_data):, 0], tsne_res[len(prep_data):, 1],
                   c='blue', alpha=0.2, label="Synthetic")
        ax.legend()
        ax.set_title('t-SNE plot')
        ax.set_xlabel('Dim 1')
        ax.set_ylabel('Dim 2')

        fig.savefig(file_path, bbox_inches='tight')
        print(f"t-SNE plot saved as {file_path}")
        plt.show()
        plt.close(fig)

    else:
        raise ValueError("analysis must be 'pca' or 'tsne'")
