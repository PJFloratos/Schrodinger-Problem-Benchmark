from src.datasets.data_utils import get_dataset

import torch
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
    train, test = get_dataset(cfg)

    # Duck-typing: 2D Toy Data vs Images
    if "input_dim" in cfg.dataset:
        check_toy(cfg, train, test)
        plot_toy(cfg, train, test, save_path)

    elif "input_channels" in cfg.dataset:
        # Use a DataLoader to grab a random batch of 64 images
        dataloader = DataLoader(test, batch_size=64, shuffle=True)
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


def check_toy(cfg, train, test):
    xtr, xte = train.data, test.data
    d = xtr.shape[1]

    def report(name, x):
        m2 = (x**2).sum(1).mean().item() / d
        print(
            f"{name:6s} n={len(x):6d} mean={x.mean(0).numpy().round(3)} "
            f"E|x|^2/d={m2:.3f} min={x.min().item():.2f} max={x.max().item():.2f}"
        )

    report("train", xtr)
    report("test", xte)
    print(f"scale={train.scale:.4f}  mean_raw={train.mean.round(3)}")

    # sizes and independence
    assert len(train) == cfg.dataset.train_samples
    assert len(test) == cfg.dataset.test_samples
    n = min(len(xtr), len(xte), 100)
    assert not torch.equal(xtr[:n], xte[:n]), "train and test are identical!"

    if cfg.dataset.type == "gaussian":
        cov = torch.cov(xtr.T)
        print("empirical cov:\n", cov.numpy().round(3), "\nanalytic var:", train.var)
        assert torch.allclose(
            xtr.mean(0), torch.tensor(train.mu, dtype=torch.float32), atol=0.1
        )
        assert torch.allclose(cov, train.var * torch.eye(2), atol=0.15)
    else:
        assert xtr.mean(0).abs().max() < 0.05
        assert abs((xtr**2).sum(1).mean().item() / d - 1) < 0.05
        assert abs((xte**2).sum(1).mean().item() / d - 1) < 0.10  # uses train stats


def plot_toy(cfg, train, test, save_path):
    g = torch.Generator().manual_seed(cfg.eval_seed)
    prior = torch.randn(len(train), 2, generator=g)
    panels = [
        ("train (pi_1)", train.data),
        ("test (pi_1)", test.data),
        ("prior N(0,I)", prior),
    ]

    allpts = torch.cat([p for _, p in panels])
    lim = float(allpts.abs().max()) * 1.05

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    for ax, (title, pts) in zip(axes, panels):
        pts = pts.numpy()
        ax.scatter(pts[:, 0], pts[:, 1], s=2, alpha=0.5)
        ax.set_title(title)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    fig.suptitle(f"{cfg.dataset.type}  (scale={train.scale:.3f})")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg: DictConfig):
    inspect_dataset(cfg)
    # quantify_pca_bounds(cfg, threshold=0.999)


if __name__ == "__main__":
    main()
