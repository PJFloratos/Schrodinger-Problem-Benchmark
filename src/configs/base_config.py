import torch

from dataclasses import dataclass, field
from typing import Optional


"""
TODO:
    - Standarize for argitrary Brownian Motion reference
    - Update for OU reference
    - Standarize configs
"""


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
    model_type: str = (
        "ebm"  # in [sde, minibatch, flow_m, ipf, imf, sf2m, lightsbm, ebm, scone]
    )
    model_name: str = "EBM"  # in [SDE, ODE, FM, IPF, IMF, SF2M, LSBM, EBM, SCONE]
    ot_method: str = "minibatch"  # in [sinkhorn, minibatch, greedy]
    n_potentials: int = 1000
    epochs: int = 100
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    sim_steps: int = 100
    num_iter: int = 5000
    refresh_every: int = 500
    num_cache_batches: int = 10
    save_interval: int = 500
    eval_per: int = 200
    use_compile: bool = True
    use_amp: bool = False
    use_ema: bool = False

    # EBM: Lower eps -> more ula_steps, fewer l2_reg

    # --- EBM Specific ---
    epsilon: float = 0.001  # Entropy penalty scale (reference variance)
    ula_steps: int = 300  # Training Langevin MCMC steps
    ula_step_size: float = 1e-3  # Langevin step size (eta)
    spatial_clip: float = 10.0  # Guardrail: spatial gradient clip
    l2_reg: float = 0.0  # Guardrail: potential magnitude penalty
    cold_start: float = 0.02
    eta_min_infer: float = 1e-5

    # --- Evaluation ---
    eval_sim_steps: int = 100
    track_gen_samples: int = 512  # Small batch for fast mid-training tracking
    eval_gen_samples: int = 4000  # Massive batch for final end-of-training metric

    def __post_init__(self):
        """Append dataset name to paths automatically to prevent overwriting."""
        self.plots_path = f"{self.plots_path}/{self.dataset_name}/{self.model_name}"
        self.models_path = f"{self.models_path}/{self.dataset_name}/{self.model_name}"
        self.logs_path = f"{self.logs_path}/{self.dataset_name}/{self.model_name}"
