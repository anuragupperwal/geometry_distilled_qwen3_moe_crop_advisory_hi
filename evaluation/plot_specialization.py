import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_cka_heatmap(matrix_path, save_path="expert_specialization.png", title=None):

    cka_matrix = torch.load(matrix_path)

    if isinstance(cka_matrix, torch.Tensor):
        cka_matrix = cka_matrix.cpu().numpy()

    sns.set(font_scale=1.2)

    # --------------------------------------------------
    # CASE 1: Expert vs Expert matrix
    # --------------------------------------------------
    if cka_matrix.ndim == 2:

        # Symmetrize safely (ignore NaNs)
        cka_matrix = np.nanmean(
            np.stack([cka_matrix, cka_matrix.T]), axis=0
        )

        np.fill_diagonal(cka_matrix, 1.0)

        n_experts = cka_matrix.shape[0]

        valid_mask = ~np.isnan(cka_matrix)

        plt.figure(figsize=(14, 12))

        sns.heatmap(
            cka_matrix,
            mask=~valid_mask,   # hide inactive pairs
            annot=True,
            fmt=".6f",
            cmap="YlGnBu",
            vmin=0,
            vmax=1,
            xticklabels=[f"Exp {i}" for i in range(n_experts)],
            yticklabels=[f"Exp {i}" for i in range(n_experts)],
            square=True,
            cbar_kws={"label": "CKA Similarity"}
        )

        plt.xlabel("Expert Index")
        plt.ylabel("Expert Index")

    # --------------------------------------------------
    # CASE 2: Teacher vs Expert vector
    # --------------------------------------------------
    elif cka_matrix.ndim == 1:

        n_experts = len(cka_matrix)

        plt.figure(figsize=(10, 5))

        sns.barplot(
            x=np.arange(n_experts),
            y=cka_matrix,
            hue=np.arange(n_experts),
            palette="YlGnBu",
            legend=False
        )

        plt.ylim(0, 1)
        plt.xlabel("Expert Index")
        plt.ylabel("Teacher Similarity")

    else:
        raise ValueError("Unsupported tensor shape for visualization")

    if title is None:
        title = "CKA Analysis"

    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Visualization saved to {save_path}")

if __name__ == "__main__":
    plot_cka_heatmap("checkpoints/Qwen/Qwen3-0.6B-Agri-Distilled/run_test_E86460/step-500.pth")
    pass