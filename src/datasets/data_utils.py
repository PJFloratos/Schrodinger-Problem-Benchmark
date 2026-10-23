from omegaconf import DictConfig
from src.datasets import Toy2DDataset, MNISTDataset, CIFAR10Dataset


def get_dataset(cfg: DictConfig):
    """Initialize the correct dataset based on the Hydra configuration."""
    dataset_name = cfg.dataset.name

    if dataset_name == "8Gaussians":
        train = Toy2DDataset(
            n_samples=cfg.dataset.train_samples, dataset_type=cfg.dataset.type
        )
        test = Toy2DDataset(
            n_samples=cfg.dataset.test_samples, dataset_type=cfg.dataset.type
        )
    elif dataset_name == "MNIST":
        train = MNISTDataset(train=True)
        test = MNISTDataset(train=False)
    elif dataset_name == "CIFAR10":
        train = CIFAR10Dataset(train=True)
        test = CIFAR10Dataset(train=False)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    return train, test
