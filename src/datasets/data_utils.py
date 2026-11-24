from src.datasets import Toy2DDataset, MNISTDataset, CIFAR10Dataset, SBBenchmarkDataset
from src.dynamics.references import ReferenceProcess

from omegaconf import DictConfig
from typing import Optional


def get_dataset(cfg: DictConfig, ref: Optional[ReferenceProcess] = None):
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

    elif t == "sb_benchmark":
        if ref is None:
            raise ValueError(
                "`ref` must be provided when dataset.type == 'sb_benchmark'."
            )

        sb_cfg = cfg.dataset.sb
        shared_kwargs = dict(
            ref=ref,
            n_components=int(sb_cfg.get("n_components", 5)),
            cov_type=sb_cfg.get("cov_type", "isotropic"),
            var=float(sb_cfg.get("var", 0.25)),
            prior_std=sb_cfg.get("prior_std", None),
            n_samples=cfg.dataset.train_samples,
            seed=cfg.seed,
            bridge_seed=int(sb_cfg.get("bridge_seed", 42)),
            paired=bool(cfg.dataset.get("paired", False)),
            device=cfg.get("device", "cpu"),
        )

        if sb_cfg.mode == "synthetic":
            train = SBBenchmarkDataset.from_synthetic(
                d=int(cfg.dataset.input_dim),
                geometry=sb_cfg.get("geometry", "sphere"),
                radius=float(sb_cfg.get("radius", 5.0)),
                **shared_kwargs,
            )
        elif sb_cfg.mode == "empirical":
            base_cfg = DictConfig(
                {
                    "dataset": sb_cfg.base_dataset,
                    "seed": cfg.seed,
                    "eval_seed": cfg.eval_seed,
                }
            )
            base_train, _ = get_dataset(base_cfg)
            train = SBBenchmarkDataset.from_empirical(
                target_data=base_train,
                **shared_kwargs,
            )
        else:
            raise ValueError(f"Unknown SB benchmark mode: {sb_cfg.mode}")

        test = SBBenchmarkDataset(
            bridge=train.bridge,
            n_samples=cfg.dataset.test_samples,
            seed=cfg.eval_seed,
            paired=bool(cfg.dataset.get("test_paired", True)),
        )

    else:
        raise ValueError(f"Unknown dataset type: {t}")

    return train, test
