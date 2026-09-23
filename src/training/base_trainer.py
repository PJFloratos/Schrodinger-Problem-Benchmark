import torch
import torch.nn as nn

import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple


class BaseTrainer(ABC):
    def __init__(
        self,
        device: torch.device,
        metric_logger: Any,  # e.g., WandB or TensorBoard writer
    ):
        self.device = device
        self.metric_logger = metric_logger

        # Compute Cost: Hardware Footprint & Timing
        self.start_time = time.time()
        self.total_gradient_steps = 0
        self.total_nfes = 0

        # State tracking for Outer-Loop Convergence
        self.previous_model_outputs: Optional[torch.Tensor] = None
        self.fixed_probe_batch: Optional[torch.Tensor] = None

    def log_inner_step(
        self,
        model: nn.Module,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        phase: str,
        ipf_iter: int,
    ) -> None:
        """Inner-Phase Diagnostics."""
        # Calculate pre-clip gradient norm
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                grad_norm += param_norm.item() ** 2
        grad_norm = grad_norm**0.5

        # Extract current learning rate
        current_lr = optimizer.param_groups[0].get("lr", 0.0)

        if hasattr(self.metric_logger, "log"):
            self.metric_logger.log(
                {
                    f"{phase}/inner_loss": loss.item(),
                    f"{phase}/grad_norm_pre_clip": grad_norm,
                    f"{phase}/effective_lr": current_lr,
                    "global_step": self.total_gradient_steps,
                    "ipf_iteration": ipf_iter,
                }
            )
        self.total_gradient_steps += 1

    def track_cache_staleness(
        self, pre_refresh_loss: float, post_refresh_loss: float, step_idx: int
    ):
        """Detects model drift at cache boundaries."""
        if hasattr(self.metric_logger, "log"):
            self.metric_logger.log(
                {
                    "diagnostics/cache_staleness_jump": post_refresh_loss
                    - pre_refresh_loss,
                    "global_step": step_idx,
                }
            )

    def log_phase_end(self, phase: str, ipf_iter: int, metrics: Dict[str, float]):
        """Logs aggregated phase metrics (Loss, MMD, Time, NFEs) received from fit()."""
        logged_metrics = {f"{phase}/{k}": v for k, v in metrics.items()}
        logged_metrics["ipf_iteration"] = ipf_iter

        if hasattr(self.metric_logger, "log"):
            self.metric_logger.log(logged_metrics)

    @torch.no_grad()
    def track_parameter_drift(self, model: nn.Module, ipf_iter: int):
        """Section 2: Evaluates ||f_model_n - f_model_{n-1}||."""
        if self.fixed_probe_batch is None:
            return

        t_probe = torch.full(
            (self.fixed_probe_batch.size(0), 1), 0.5, device=self.device
        )
        current_outputs = model(self.fixed_probe_batch, t_probe)

        if self.previous_model_outputs is not None:
            drift = torch.norm(
                current_outputs - self.previous_model_outputs, p=2
            ).item()
            if hasattr(self.metric_logger, "log"):
                self.metric_logger.log(
                    {"outer_loop/parameter_drift": drift, "ipf_iteration": ipf_iter}
                )

        self.previous_model_outputs = current_outputs.clone()

    @torch.no_grad()
    def track_path_consistency(
        self,
        f_model: nn.Module,
        b_model: nn.Module,
        ipf_iter: int,
        t_points: tuple = (0.25, 0.5, 0.75),
    ):
        """Evaluates forward/backward vector field alignment across t in [0, 1]."""
        if self.fixed_probe_batch is None:
            return

        f_model.eval()
        b_model.eval()
        device = self.device

        total_mse = 0.0
        for t_val in t_points:
            t_tensor = torch.full(
                (self.fixed_probe_batch.size(0), 1), t_val, device=device
            )
            f_vec = f_model(self.fixed_probe_batch, t_tensor)
            b_vec = b_model(self.fixed_probe_batch, t_tensor)

            # Measure field alignment consistency
            total_mse += torch.mean((f_vec + b_vec) ** 2).item()

        consistency_mse = total_mse / len(t_points)

        if hasattr(self.metric_logger, "log"):
            self.metric_logger.log(
                {
                    "outer_loop/path_consistency_mse": consistency_mse,
                    "ipf_iteration": ipf_iter,
                }
            )

    def log_compute_cost(self, models: List[nn.Module]):
        """Section 5: Hardware Footprint & Timing."""
        hw_footprint = sum(
            p.numel() for m in models for p in m.parameters() if p.requires_grad
        )
        metrics = {
            "compute/wall_clock_time_seconds": time.time() - self.start_time,
            "compute/total_gradient_steps": self.total_gradient_steps,
            "compute/total_nfes": self.total_nfes,
            "compute/hardware_footprint_params": hw_footprint,
        }
        if hasattr(self.metric_logger, "log"):
            self.metric_logger.log(metrics)

    @abstractmethod
    def fit(self, *args, **kwargs):
        pass
