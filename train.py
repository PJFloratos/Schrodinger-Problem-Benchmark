# https://github.com/JTT94/diffusion_schrodinger_bridge/
# https://github.com/yuyang-shi/dsbm-pytorch


from src.pipeline import Pipeline
from src.datasets.data_utils import get_dataset
from src.configs import BaseConfig, Toy2dConfig, MNISTConfig
from src.utils.log import text_logger, MetricLogger
from src.utils.seed import set_all_seeds

import torch

import os
from enum import Enum


logger = text_logger(__name__)


class DatasetType(str, Enum):
    TOY2D = "toy2d"
    MNIST = "mnist"


# Pick the dataset to train on
ACTIVE_DATASET = DatasetType.TOY2D


def main(cfg: BaseConfig) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    # --- LOCK DOWN REPRODUCIBILITY FIRST ---
    set_all_seeds(cfg.seed)

    # 1. Setup Directories
    os.makedirs(cfg.models_path, exist_ok=True)
    os.makedirs(cfg.plots_path, exist_ok=True)
    os.makedirs(cfg.logs_path, exist_ok=True)

    # Metric logger for benchmark tracking
    metric_logger = MetricLogger(
        log_dir=cfg.logs_path,
        use_tensorboard=True,
        use_wandb=False,
    )

    # Setup Data
    train_dataset, test_dataset = get_dataset(cfg)

    # Pipeline Orchestration (Train -> Evaluate)
    pipeline = Pipeline(
        cfg=cfg,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        metric_logger=metric_logger,
    )

    generative_model, eval_res = pipeline.execute()

    # Log final results
    metric_logger.log({f"final/{k}": v for k, v in eval_res.items()})
    metric_logger.close()


if __name__ == "__main__":
    cfg = Toy2dConfig() if ACTIVE_DATASET == DatasetType.TOY2D else MNISTConfig()

    main(cfg)
