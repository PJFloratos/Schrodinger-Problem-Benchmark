from src.utils.data import get_dataset
from src.configs import BaseConfig, Toy2dConfig, MNISTConfig

from torch.utils.data import DataLoader
from torchvision.utils import make_grid

import os
import matplotlib.pyplot as plt
from enum import Enum


class DatasetType(str, Enum):
    TOY2D = "toy2d"
    MNIST = "mnist"


# Pick the dataset to inspect
ACTIVE_DATASET = DatasetType.TOY2D


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
    config = Toy2dConfig() if ACTIVE_DATASET == DatasetType.TOY2D else MNISTConfig()

    inspect_dataset(config)
