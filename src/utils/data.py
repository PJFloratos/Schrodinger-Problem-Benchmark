from src.datasets import Toy2DDataset, MNISTDataset
from src.configs import BaseConfig, Toy2dConfig, MNISTConfig


def get_dataset(cfg: BaseConfig):
    """Initialize the correct dataset based on config type."""
    if isinstance(cfg, Toy2dConfig):
        train = Toy2DDataset(n_samples=cfg.train_samples, dataset_type=cfg.dataset_type)
        test = Toy2DDataset(n_samples=cfg.test_samples, dataset_type=cfg.dataset_type)
    elif isinstance(cfg, MNISTConfig):
        train = MNISTDataset(train=True)
        test = MNISTDataset(train=False)
    else:
        raise ValueError(f"Unknown configuration type: {type(cfg)}")

    return train, test
