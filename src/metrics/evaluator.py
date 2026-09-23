from src.utils.log import text_logger
from src.metrics.distances import (
    get_mmd,
    get_path_consistency_mse,
    get_drift_mse,
    get_generative_quality_metrics,
)

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid

import numpy as np
from scipy.stats import wasserstein_distance
from scipy.optimize import linear_sum_assignment

import time
from tqdm import tqdm
from typing import Dict, Optional, Sequence, Union, Tuple, Callable


class Evaluator:
    logger = text_logger(__name__)

    def __init__(
        self,
        test_ds: Dataset,
        device: torch.device = torch.device("cpu"),
        sde_steps: int = 100,
        n_gen_samples: Optional[int] = None,  # None -> as many as the test set
        ground_truth_v: Optional[Callable] = None,
    ) -> None:
        self.dl = DataLoader(test_ds, batch_size=256, shuffle=False)
        self.device = device
        self.sde_steps = sde_steps
        self.n_gen_samples = n_gen_samples
        self.ground_truth_v = ground_truth_v

        # Sniff dataset shape to decide which metrics to run
        sample_batch = next(iter(self.dl))
        self.is_image_data = sample_batch.ndim == 4

    def evaluate(
        self,
        model: nn.Module,
        backward_model: nn.Module = None,
        num_samples: int = None,  # To overwrite the global one
    ) -> Dict[str, float]:
        Evaluator.logger.info("Starting Evaluation Process...")

        model.eval()
        if backward_model:
            backward_model.eval()

        # Validation & Simulation
        loss, x_true = self._compute_validation_loss(model)

        x_gen, gen_time = self._simulate_paths(
            model, num_samples=num_samples if num_samples else len(x_true)
        )

        # Eval Metrics
        Evaluator.logger.info("Calculating Evaluation Metrics...")
        results = {
            "eval_loss": float(loss),
            "eval_NFEs": self.sde_steps,
            "eval_generation_time": gen_time,
        }

        if self.is_image_data:
            results.update(get_generative_quality_metrics(x_true, x_gen))
        else:
            results["eval_MMD"] = float(get_mmd(x_true, x_gen, device=self.device))

            # Ground Truth
            if self.ground_truth_v:
                results["drift_MSE"] = get_drift_mse(model, self.ground_truth_v, x_gen)

        Evaluator.logger.info("Evaluation Process Completed Successfully.")
        return results

    def _compute_validation_loss(self, model: nn.Module) -> Tuple[float, torch.Tensor]:
        """Calculates regression health against straight-line/SDE paths and extracts targets."""

        def _sde_target(model, z_batch):
            eps = torch.randn_like(z_batch)
            x_t = t_expand * z_batch + torch.sqrt(1.0 - t_expand) * eps
            return x_t, (z_batch - x_t) / (1.0 - t_expand)

        def _minibatch_target(model, z_batch, batch_size):
            x_0 = torch.randn_like(z_batch)
            cost_matrix = torch.cdist(x_0.view(B, -1), z_batch.view(B, -1), p=2).pow(2)
            row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().numpy())

            x_0 = x_0[row_ind]
            z_batch = z_batch[col_ind]
            x_t = t_expand * z_batch + (1.0 - t_expand) * x_0
            return x_t, (z_batch - x_t) / (1.0 - t_expand)

        def _fm_target(model, z_batch):
            x_0 = torch.randn_like(z_batch)
            x_t = t_expand * z_batch + (1.0 - t_expand) * x_0
            return x_t, z_batch - x_0

        total_loss = 0.0
        x_true_list = []

        with torch.inference_mode():
            for z_batch in tqdm(self.dl, ascii=True, desc="    Calculating Loss"):
                z_batch = z_batch.to(self.device, non_blocking=True)
                B = z_batch.shape[0]

                # 1. Sample time uniformly
                t = torch.rand(B, 1, device=self.device) * 0.999

                # Expand time for math
                t_expand = t.view(B, *([1] * (z_batch.ndim - 1)))

                # Construct path formulations based on model type
                if model.model_type == "sde":
                    x_t, target_u = _sde_target(model, z_batch)
                elif model.model_type == "minibatch":
                    x_t, target_u = _minibatch_target(model, z_batch, B)
                elif model.model_type == "flow_m":
                    x_t, target_u = _fm_target(model, z_batch)

                # Model predicts the vector field
                pred_u = model(x_t, t)

                # Calculate Loss based on model dynamics
                if model.model_type in ["sde", "minibatch"]:
                    # Time-Weighted CGM Loss to prevent explosion near t=1
                    batch_loss = torch.mean((1.0 - t_expand) * (pred_u - target_u) ** 2)
                else:
                    # Standard MSE for Flow Matching (constant velocity)
                    batch_loss = torch.mean((pred_u - target_u) ** 2)

                total_loss += batch_loss.item()

                x_true_list.append(z_batch.cpu())

        return total_loss / len(self.dl), torch.cat(x_true_list, dim=0)

    def _simulate_paths(
        self, model: nn.Module, num_samples: int
    ) -> Tuple[torch.Tensor, float]:
        """Handles trajectory simulation and generation timing."""
        gen_start = time.time()

        # Overwrite the global n_gen_samples
        if num_samples:
            n_samples = num_samples
        else:
            n_samples = self.n_gen_samples

        x_gen_tensor = model.generate(
            n_samples=n_samples, steps=self.sde_steps, device=self.device
        )

        return x_gen_tensor, time.time() - gen_start
