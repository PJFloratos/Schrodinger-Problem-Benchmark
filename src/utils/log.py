import os
import logging
import json
from typing import Dict, Any


def text_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Configures the text logger with console and file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # do not pass logs to the default logger

    # Check if the logger already has handlers to prevent adding multiple handlers
    if not logger.handlers:
        # Create the formatter object for the logger
        file_formatter = logging.Formatter(
            "%(asctime)s \t %(filename)s \t %(levelname)s \t %(message)s"
        )
        stdout_formatter = logging.Formatter("%(levelname)s \t %(message)s")

        # Create the console handler and setting its level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        # Create the file handler and setting its level
        log_file = "run.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)

        # Add the formatter to the handlers
        console_handler.setFormatter(stdout_formatter)
        file_handler.setFormatter(file_formatter)

        # Add the handlers to the logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger


class MetricLogger:
    """
    Unified metric tracker. Routes scalar dictionaries to TensorBoard,
    Weights & Biases, and a local JSON-lines backup file.
    """

    def __init__(
        self, log_dir: str, use_tensorboard: bool = False, use_wandb: bool = False
    ):
        self.log_dir = log_dir
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb

        os.makedirs(self.log_dir, exist_ok=True)
        self.metrics_file = os.path.join(self.log_dir, "metrics.jsonl")

        if self.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            self.tb_writer = SummaryWriter(log_dir=self.log_dir)

    def log(self, metrics: Dict[str, Any]):
        """Logs a dictionary of scalar metrics to all active backends."""

        # 1. Local Backup (JSON-lines allows appending without corrupting)
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")

        # 2. TensorBoard
        if self.use_tensorboard:
            # Assume the trainer passed a 'global_step' or 'ipf_iteration' to use as the X-axis
            step = metrics.get("global_step", metrics.get("ipf_iteration", 0))
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and key not in [
                    "global_step",
                    "ipf_iteration",
                ]:
                    self.tb_writer.add_scalar(key, value, step)

        # 3. Weights & Biases
        if self.use_wandb:
            import wandb

            wandb.log(metrics)

    def close(self):
        if self.use_tensorboard:
            self.tb_writer.close()
