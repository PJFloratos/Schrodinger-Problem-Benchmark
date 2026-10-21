from src.datasets.data_utils import get_dataset
from src.configs import BaseConfig, Toy2dConfig, MNISTConfig, CIFAR10Config

from torch.utils.data import DataLoader
from torchvision.utils import make_grid

import numpy as np
from sklearn.decomposition import PCA

import os
import matplotlib.pyplot as plt
from enum import Enum


class DatasetType(str, Enum):
    TOY2D = "toy2d"
    MNIST = "mnist"
    CIFAR10 = "cifar10"


# Pick the dataset to inspect
ACTIVE_DATASET = DatasetType.TOY2D


def quantify_pca_bounds(cfg: BaseConfig, threshold: float = 0.95) -> None:
    """
    Computes PCA on the dataset to analyze the explained variance and
    the Wasserstein error bound (sum of discarded eigenvalues).
    """
    if isinstance(cfg, Toy2dConfig):
        print(
            "Skipping PCA: Toy2D is already dimension 2. Reducing to 1D destroys the manifold."
        )
        return

    print(f"\n--- Running PCA Analysis on {cfg.dataset_name} ---")

    # get dataset
    _, dataset = get_dataset(cfg)

    # Load the entire dataset for accurate covariance estimation
    # (For massive datasets, use a large subset, e.g., 10,000 samples)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    images = next(iter(dataloader))

    # Flatten the images from (N, 1, 28, 28) to (N, 784)
    X = images.view(images.size(0), -1).numpy()

    # Fit PCA
    pca = PCA().fit(X)
    eigenvalues = pca.explained_variance_
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # Find the dimension k that captures 95% of the total variance
    k = (np.argmax(cumulative_variance >= threshold) + 1).item()

    # Calculate the Wasserstein Error Bound (Module 5)
    # This is the exact sum of the discarded eigenvalues
    discarded_variance = np.sum(eigenvalues[k:])

    print(f"Original dimension (p): {X.shape[1]}")
    print(f"Reduced dimension (k) for {threshold}% variance: {k}")
    print(f"Compression Ratio: {X.shape[1] / k:.1f}x smaller state space")
    print(
        f"Wasserstein Error Bound (sum of discarded eigenvalues): {discarded_variance:.4f}"
    )

    # Plot the Scree Plot (Explained Variance)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, k + 1), cumulative_variance[:k], linewidth=2, color="indigo")
    plt.axvline(x=k, color="darkorange", linestyle="--", label=f"k={k} (95% Var)")
    plt.title("Cumulative Explained Variance (MNIST)")
    plt.xlabel("Number of Principal Components (k)")
    plt.ylabel("Proportion of Variance Retained")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(os.path.dirname(cfg.plots_path), "pca_variance_plot.png")
    plt.savefig(save_path)
    plt.close()
    print(f"PCA Scree plot saved to {save_path}\n")


def inspect_dataset(cfg: BaseConfig) -> None:
    # cfg.plots_path includes the model_type (e.g., ./plots/Moons/sde).
    # We step up one directory to save ground truth at the dataset root (e.g., ./plots/Moons).
    base_plot_dir = os.path.dirname(cfg.plots_path)
    os.makedirs(base_plot_dir, exist_ok=True)
    save_path = os.path.join(base_plot_dir, "ground_truth_dataset.png")

    # Load data
    _, dataset = get_dataset(cfg)

    if isinstance(cfg, Toy2dConfig):
        # Extract the raw standardized points
        points = dataset.data.numpy()

        plt.figure(figsize=(6, 6))
        plt.scatter(points[:, 0], points[:, 1], s=2, alpha=0.8, color="darkorange")
        plt.title(f"Ground Truth Distribution: {cfg.dataset_name}")

        plt.savefig(save_path)
        plt.close()

    elif isinstance(cfg, MNISTConfig):
        # Use a DataLoader to grab a random batch of 64 images
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        images = next(iter(dataloader))

        # Re-normalize from [-1, 1] to [0, 1] for plotting
        images = (images + 1) / 2.0

        # Create an 8x8 grid and permute for matplotlib (H, W, C)
        grid = make_grid(images, nrow=8).permute(1, 2, 0).numpy()

        plt.figure(figsize=(8, 8))
        plt.imshow(grid)
        plt.axis("off")
        plt.title(f"Ground Truth Distribution: {cfg.dataset_name}")

        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

    else:
        raise ValueError("Unsupported configuration type.")

    print(f"Dataset plot successfully saved to {save_path}")


if __name__ == "__main__":
    if ACTIVE_DATASET == DatasetType.MNIST:
        config = MNISTConfig()
    elif ACTIVE_DATASET == DatasetType.CIFAR10:
        config = CIFAR10Config()
    else:
        config = Toy2dConfig()

    inspect_dataset(config)
    # quantify_pca_bounds(config, threshold=0.999)
