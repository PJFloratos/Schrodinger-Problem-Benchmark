from src.datasets.data_utils import get_dataset

from torch.utils.data import DataLoader
from torchvision.utils import make_grid

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import os
import hydra
from omegaconf import DictConfig


def quantify_pca_bounds(cfg: DictConfig, threshold: float = 0.95) -> None:
    """
    Computes PCA on the dataset to analyze the explained variance and
    the Wasserstein error bound (sum of discarded eigenvalues).
    """
    if "input_dim" in cfg.dataset:
        print(
            f"Skipping PCA: {cfg.dataset.name} is already low-dimensional. "
            "Reducing it destroys the manifold."
        )
        return

    print(f"\n--- Running PCA Analysis on {cfg.dataset.name} ---")

    # get dataset
    _, dataset = get_dataset(cfg)

    # Load the dataset for accurate covariance estimation
    # Bounded to 10,000 to prevent memory explosions on full CIFAR/MNIST sets
    dataloader = DataLoader(dataset, batch_size=min(len(dataset), 10000), shuffle=False)
    batch = next(iter(dataloader))
    images = batch[0] if isinstance(batch, (list, tuple)) else batch

    # Flatten the images from (N, C, H, W) to (N, C*H*W)
    X = images.view(images.size(0), -1).numpy()

    # Fit PCA
    pca = PCA().fit(X)
    eigenvalues = pca.explained_variance_
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # Find the dimension k that captures the threshold % of the total variance
    k = (np.argmax(cumulative_variance >= threshold) + 1).item()

    # Calculate the Wasserstein Error Bound (Module 5)
    # This is the exact sum of the discarded eigenvalues
    discarded_variance = np.sum(eigenvalues[k:])

    print(f"Original dimension (p): {X.shape[1]}")
    print(f"Reduced dimension (k) for {threshold * 100:.1f}% variance: {k}")
    print(f"Compression Ratio: {X.shape[1] / k:.1f}x smaller state space")
    print(
        f"Wasserstein Error Bound (sum of discarded eigenvalues): {discarded_variance:.4f}"
    )

    # Plot the Scree Plot (Explained Variance)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, k + 1), cumulative_variance[:k], linewidth=2, color="indigo")
    plt.axvline(
        x=k,
        color="darkorange",
        linestyle="--",
        label=f"k={k} ({threshold * 100:.1f}% Var)",
    )
    plt.title(f"Cumulative Explained Variance ({cfg.dataset.name})")
    plt.xlabel("Number of Principal Components (k)")
    plt.ylabel("Proportion of Variance Retained")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(
        os.path.dirname(cfg.plots_path), f"pca_variance_plot_{cfg.dataset.name}.png"
    )
    plt.savefig(save_path)
    plt.close()
    print(f"PCA Scree plot saved to {save_path}\n")


def inspect_dataset(cfg: DictConfig) -> None:
    # Step up one directory to save ground truth at the dataset root
    base_plot_dir = os.path.dirname(cfg.plots_path)
    os.makedirs(base_plot_dir, exist_ok=True)
    save_path = os.path.join(base_plot_dir, f"ground_truth_{cfg.dataset.name}.png")

    # Load data
    _, dataset = get_dataset(cfg)

    # Duck-typing: 2D Toy Data vs Images
    if "input_dim" in cfg.dataset:
        points = dataset.data.numpy()

        plt.figure(figsize=(6, 6))
        plt.scatter(points[:, 0], points[:, 1], s=2, alpha=0.8, color="darkorange")
        plt.title(f"Ground Truth Distribution: {cfg.dataset.name}")

        plt.savefig(save_path)
        plt.close()

    elif "input_channels" in cfg.dataset:
        # Use a DataLoader to grab a random batch of 64 images
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        batch = next(iter(dataloader))

        # Safely extract tensor if dataset returns (image, label) tuples
        images = batch[0] if isinstance(batch, (list, tuple)) else batch

        # Re-normalize from [-1, 1] to [0, 1] for plotting
        images = (images + 1) / 2.0

        # Create an 8x8 grid and permute for matplotlib (H, W, C)
        grid = make_grid(images, nrow=8).permute(1, 2, 0).numpy()

        plt.figure(figsize=(8, 8))
        plt.imshow(grid)
        plt.axis("off")
        plt.title(f"Ground Truth Distribution: {cfg.dataset.name}")

        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

    else:
        raise ValueError(
            "Unsupported configuration type: must have 'input_dim' or 'input_channels'."
        )

    print(f"Dataset plot successfully saved to {save_path}")


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg: DictConfig):
    inspect_dataset(cfg)
    # quantify_pca_bounds(cfg, threshold=0.999)


if __name__ == "__main__":
    main()
