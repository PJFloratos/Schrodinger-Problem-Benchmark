from src.utils.log import configure_logger
from src.metrics.distances import get_mmd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid

import numpy as np
from scipy.stats import wasserstein_distance
from scipy.optimize import linear_sum_assignment

import os
from tqdm import tqdm
from typing import Dict, Optional, Sequence, Union


class Evaluator:
    logger = configure_logger(__name__)

    def __init__(
        self,
        model: nn.Module,
        test_ds: Dataset,
        device: torch.device = torch.device("cpu"),
        sde_steps: int = 100,
        log_file: Optional[str] = None,
        n_gen_samples: Optional[int] = None,  # None -> as many as the test set
    ) -> None:
        self.model = model.to(device, non_blocking=True)
        self.test_ds = test_ds
        self.device = device
        self.sde_steps = sde_steps
        self.n_gen_samples = n_gen_samples
        self.log_file = log_file

    def evaluate(self) -> Dict[str, float]:
        Evaluator.logger.info("Starting Evaluation Process...")
        self.model.eval()

        dl = DataLoader(self.test_ds, batch_size=256, shuffle=False)
        total_loss = 0.0
        x_true_list = []

        with torch.inference_mode():
            for z_batch in tqdm(dl, ascii=True, desc="    Evaluating CGM Loss"):
                z_batch = z_batch.to(self.device, non_blocking=True)
                B = z_batch.shape[0]
                x_true_list.append(z_batch.cpu().numpy())

                # 1. Sample time uniformly
                t = torch.rand(B, 1, device=self.device) * 0.999

                # Expand time for math
                t_expand = t.view(B, *([1] * (z_batch.ndim - 1)))

                if self.model.model_type == "minibatch":
                    x_0 = torch.randn_like(z_batch)
                    cost_matrix = torch.cdist(
                        x_0.view(B, -1), z_batch.view(B, -1), p=2
                    ).pow(2)
                    row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().numpy())

                    x_0 = x_0[row_ind]
                    z_batch = z_batch[col_ind]

                    x_t = t_expand * z_batch + (1.0 - t_expand) * x_0
                    target_u = (z_batch - x_t) / (1.0 - t_expand)

                elif self.model.model_type == "sde":
                    eps = torch.randn_like(z_batch)
                    x_t = t_expand * z_batch + torch.sqrt(1.0 - t_expand) * eps
                    target_u = (z_batch - x_t) / (1.0 - t_expand)

                elif self.model.model_type == "flow_m":
                    x_0 = torch.randn_like(z_batch)
                    # cost_matrix = torch.cdist(
                    #     x_0.view(B, -1), z_batch.view(B, -1), p=2
                    # ).pow(2)
                    # row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().numpy())

                    # x_0 = x_0[row_ind]
                    # z_batch = z_batch[col_ind]

                    x_t = t_expand * z_batch + (1.0 - t_expand) * x_0
                    target_u = z_batch - x_0

                # Append the potentially permuted data batches for accurate MMD scoring
                x_true_list.append(z_batch.cpu().numpy())

                # Model predicts the vector field
                pred_u = self.model(x_t, t)

                # Calculate Loss based on model dynamics
                if self.model.model_type in ["sde", "minibatch"]:
                    # Time-Weighted CGM Loss to prevent explosion near t=1
                    loss = torch.mean((1.0 - t_expand) * (pred_u - target_u) ** 2)
                else:
                    # Standard MSE for Flow Matching (constant velocity)
                    loss = torch.mean((pred_u - target_u) ** 2)

                total_loss += loss.item()

        cgm_loss = total_loss / len(dl)
        x_true = np.concatenate(x_true_list, axis=0)

        Evaluator.logger.info("    Simulating Paths...")
        x_gen_tensor = self.model.generate(
            n_samples=len(x_true), steps=self.sde_steps, device=self.device
        )
        x_gen = x_gen_tensor.cpu().numpy()

        Evaluator.logger.info("    Calculating Dist Metrics...")
        results = {
            "CGM_Loss": float(cgm_loss),
            "MMD": float(get_mmd(x_true, x_gen, device=self.device)),
        }

        # Log metrics
        if self.log_file:
            with open(self.log_file, "w") as f:
                f.write(str(results))

        Evaluator.logger.info("Evaluation Process Completed Successfully.")
        return results
