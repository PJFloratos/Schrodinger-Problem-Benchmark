from src.datasets import Toy2DDataset, MNISTDataset, CIFAR10Dataset

from omegaconf import DictConfig


def get_dataset(cfg: DictConfig):
    t = cfg.dataset.type

    # List of all supported 2D toy dataset types from your Toy2DDataset class
    toy2d_types = ["moons", "swiss_roll", "checkerboard", "gaussian", "8gaussians"]

    # Check if the requested dataset is one of the 2D toy datasets
    if t in toy2d_types:
        train = Toy2DDataset(
            n_samples=cfg.dataset.train_samples, dataset_type=t, seed=cfg.seed
        )
        test = Toy2DDataset(
            n_samples=cfg.dataset.test_samples,
            dataset_type=t,
            seed=cfg.eval_seed,
            stats=train.stats,  # same coordinates as train
        )

    elif t == "mnist":
        train, test = MNISTDataset(train=True), MNISTDataset(train=False)

    elif t == "cifar10":
        train, test = CIFAR10Dataset(train=True), CIFAR10Dataset(train=False)

    else:
        raise ValueError(f"Unknown dataset type: {t}")

    return train, test
