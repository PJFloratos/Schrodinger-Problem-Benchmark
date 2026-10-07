import torch
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BaseConfig:
    """Shared parameters across ALL experiments."""

    # --- Reproducibility ---
    seed: int = 999
    eval_seed: int = seed + 3000

    # --- Paths ---
    plots_path: str = "./plots"
    models_path: str = "./models"
    logs_path: str = "./logs"

    # --- Hardware ---
    device: torch.device = field(
        default_factory=lambda: torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )

    # --- Training ---
    model_type: str = "imf"  # in [sde, minibatch, flow_m, ipf, imf, sf2m]
    model_name: str = "IMF"  # in [SDE, ODE, FM, IPF, IMF, SF2M]
    ot_method: str = "minibatch"  # in [sinkhorn, minibatch, greedy]
    epochs: int = 2
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    sim_steps: int = 30
    num_iter: int = 5000
    refresh_every: int = 250
    num_cache_batches: int = 10
    save_interval: int = 500
    eval_per: int = 1
    use_amp: bool = False
    use_ema: bool = False

    # --- Evaluation ---
    eval_sim_steps: int = 30
    track_gen_samples: int = 512  # Small batch for fast mid-training tracking
    eval_gen_samples: int = 4000  # Massive batch for final end-of-training metric

    def __post_init__(self):
        """Append dataset name to paths automatically to prevent overwriting."""
        self.plots_path = f"{self.plots_path}/{self.dataset_name}/{self.model_name}"
        self.models_path = f"{self.models_path}/{self.dataset_name}/{self.model_name}"
        self.logs_path = f"{self.logs_path}/{self.dataset_name}/{self.model_name}"
