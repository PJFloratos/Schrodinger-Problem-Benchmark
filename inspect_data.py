from src.dataset.Toy2DDataset import Toy2DDataset
from src.dataset.MNISTDataset import MNISTDataset

from torchvision.utils import make_grid
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import os


def inspect_toy2d(type="swiss_roll", n_samples=5000):
    # Instantiate the dataset to pull the raw scaled points
    dataset = Toy2DDataset(n_samples=n_samples, dataset_type=type)
    points = dataset.data.numpy()

    # Plot the ground truth distribution
    plt.figure(figsize=(6, 6))
    plt.scatter(points[:, 0], points[:, 1], s=2, alpha=0.8, color="darkorange")
    plt.title("Ground Truth Distribution: Two Moons")

    os.makedirs("./plots", exist_ok=True)
    plt.savefig("./plots/SwissRoll/ground_truth_dataset.png")
    print("Dataset plot saved to ./plots/SwissRoll/ground_truth_dataset.png")


def inspect_mnist():
    # Instantiate the dataset
    dataset = MNISTDataset(data_dir="./data", train=True)

    # Use a DataLoader to easily grab a random batch of 64 images
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    images = next(iter(dataloader))

    # The dataset standardizes images to [-1, 1]. Rescale back to [0, 1] for plotting.
    images = (images + 1) / 2.0

    # Create an 8x8 grid of images and permute dimensions for matplotlib (H, W, C)
    grid = make_grid(images, nrow=8).permute(1, 2, 0).numpy()

    # Plot the ground truth image grid
    plt.figure(figsize=(8, 8))
    plt.imshow(grid)
    plt.axis("off")
    plt.title("Ground Truth Distribution: MNIST")

    os.makedirs("./plots/MNIST", exist_ok=True)
    plt.savefig("./plots/MNIST/ground_truth_dataset.png", bbox_inches="tight")
    print("Dataset plot saved to ./plots/MNIST/ground_truth_dataset.png")


if __name__ == "__main__":
    inspect_mnist()
