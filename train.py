# https://github.com/JTT94/diffusion_schrodinger_bridge/

from src.training import TrainingOrchestrator
from src.metrics import Evaluator
from src.utils import get_dataset, generate_and_plot
from src.utils.log import text_logger, MetricLogger
from src.configs import BaseConfig, Toy2dConfig, MNISTConfig

import torch
from torch import nn, optim
from torchvision.utils import make_grid

import matplotlib.pyplot as plt

import os
from enum import Enum
from typing import Dict


class DatasetType(str, Enum):
    TOY2D = "toy2d"
    MNIST = "mnist"


# Pick the dataset to train on
ACTIVE_DATASET = DatasetType.TOY2D


def main(cfg: BaseConfig) -> None:
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

    # 2. Setup Data
    train_dataset, test_dataset = get_dataset(cfg)

    evaluator = Evaluator(
        test_ds=test_dataset,
        device=cfg.device,
        sde_steps=cfg.eval_sim_steps,
    )

    # Define the callback using closure to access the evaluator and config
    def eval_callback(model: nn.Module, direction: str):
        if direction == "b":
            return evaluator.evaluate(model, num_samples=cfg.track_gen_samples)
        elif direction == "f":
            # Placeholder: If you want to evaluate Prior matching MMD later
            return {}

    # 3. Build & Train
    orchestrator = TrainingOrchestrator(
        cfg, train_dataset, eval_callback=eval_callback, metric_logger=metric_logger
    )
    generative_model = orchestrator.build_and_train()

    # 4. Evaluate Metrics
    log_file = f"{cfg.epochs}ep_{cfg.sim_steps}ss_{cfg.num_iter}it.txt"

    # Reuses the exact same evaluator instance, just with the larger sample count
    eval_res = evaluator.evaluate(generative_model)

    # 5. Visualize
    generate_and_plot(generative_model, eval_res, cfg)


if __name__ == "__main__":
    cfg = Toy2dConfig() if ACTIVE_DATASET == DatasetType.TOY2D else MNISTConfig()

    main(cfg)
