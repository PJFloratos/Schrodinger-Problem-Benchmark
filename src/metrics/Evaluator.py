from src.utils.log import configure_logger

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
from scipy.stats import wasserstein_distance

from tqdm import tqdm
from typing import Dict, Optional, Sequence, Union


# --- Generative Evaluation Metrics ---
def _as_flat_tensor(x, device: torch.device) -> torch.Tensor:
    """(N, ...) array/tensor -> (N, D) float32 tensor on `device`."""
    x = torch.as_tensor(x)
    return x.reshape(x.shape[0], -1).to(device=device, dtype=torch.float32)


@torch.no_grad()
def _median_gamma(a: torch.Tensor, b: torch.Tensor, n: int = 1000) -> float:
    """Median heuristic: gamma = 1 / (2 * median(||x - y||^2)) on a pooled subsample."""
    pool = torch.cat([a[:: max(1, len(a) // n)][:n], b[:: max(1, len(b) // n)][:n]])
    d2 = torch.cdist(pool, pool).square_()
    off_diag = ~torch.eye(len(pool), dtype=torch.bool, device=pool.device)
    med = d2[off_diag].median().clamp_min(1e-12)
    return 1.0 / (2.0 * med.item())


@torch.no_grad()
def _mean_kernel(
    a: torch.Tensor, b: torch.Tensor, gammas: Sequence[float], chunk: int
) -> float:
    """mean_{i,j} mean_g exp(-g * ||a_i - b_j||^2), computed in row chunks so the
    full N x M kernel matrix is never materialized."""
    total = torch.zeros((), dtype=torch.float64, device=a.device)
    for i in range(0, a.shape[0], chunk):
        d2 = torch.cdist(a[i : i + chunk], b).square_()
        for g in gammas:
            total += torch.exp(-g * d2).sum(dtype=torch.float64)
    return (total / (a.shape[0] * b.shape[0] * len(gammas))).item()


def get_mmd(
    x_true: np.ndarray,
    x_gen: np.ndarray,
    gamma: Union[float, Sequence[float], None] = None,
    chunk: int = 2048,
    device: Optional[torch.device] = None,
) -> float:
    """(Biased) squared Maximum Mean Discrepancy with an RBF kernel.

    Works for any sample shape: inputs are flattened to (N, D). If `gamma` is None,
    the bandwidth comes from the median heuristic, averaged over 5 scales
    (0.25x ... 4x). Pass a float (e.g. 1.0) or a list of floats to fix it manually.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = _as_flat_tensor(x_true, device)
    b = _as_flat_tensor(x_gen, device)

    if gamma is None:
        g0 = _median_gamma(a, b)
        gammas = [g0 * m for m in (0.25, 0.5, 1.0, 2.0, 4.0)]
    elif isinstance(gamma, (int, float)):
        gammas = [float(gamma)]
    else:
        gammas = [float(g) for g in gamma]

    xx = _mean_kernel(a, a, gammas, chunk)
    yy = _mean_kernel(b, b, gammas, chunk)
    xy = _mean_kernel(a, b, gammas, chunk)
    return xx + yy - 2.0 * xy


def get_wasserstein_x(x_true: np.ndarray, x_gen: np.ndarray) -> float:
    return wasserstein_distance(x_true[:, 0], x_gen[:, 0])


def get_wasserstein_y(x_true: np.ndarray, x_gen: np.ndarray) -> float:
    return wasserstein_distance(x_true[:, 1], x_gen[:, 1])


class Evaluator:
    logger = configure_logger(__name__)

    def __init__(
        self,
        model: nn.Module,
        test_ds: Dataset,
        criterion: nn.Module,
        device: torch.device = torch.device("cpu"),
        sde_steps: int = 100,
        n_gen_samples: Optional[int] = None,  # None -> as many as the test set
    ) -> None:
        self.model = model.to(device, non_blocking=True)
        self.test_ds = test_ds
        self.criterion = criterion
        self.device = device
        self.sde_steps = sde_steps
        self.n_gen_samples = n_gen_samples

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

                # 2. Prior is standard Gaussian[cite: 1]
                eps = torch.randn_like(z_batch)

                # 3. Construct conditional path
                x_t = t_expand * z_batch + torch.sqrt(1.0 - t_expand) * eps

                # 4. Construct exact closed-form target vector field[cite: 1]
                target_u = (z_batch - x_t) / (1.0 - t_expand)

                # 5. Model predicts the vector field
                pred_u = self.model(x_t, t)

                # 6. Calculate Time-Weighted CGM Loss to prevent explosion near t=1
                loss = torch.mean((1.0 - t_expand) * (pred_u - target_u) ** 2)

                # total_loss += self.criterion(pred_u, target_u).item()
                total_loss += loss

        cgm_loss = total_loss / len(dl)
        x_true = np.concatenate(x_true_list, axis=0)

        Evaluator.logger.info("    Simulating SDE...")
        x_gen_tensor = self.model.generate(
            n_samples=len(x_true), steps=self.sde_steps, device=self.device
        )
        x_gen = x_gen_tensor.cpu().numpy()

        Evaluator.logger.info("    Calculating Dist Metrics...")
        results = {
            "CGM_Loss": float(cgm_loss),
            "MMD": float(get_mmd(x_true, x_gen, device=self.device)),
            # "MMD": float(get_mmd(x_true, x_gen)),
            # "Wasserstein_X": float(get_wasserstein_x(x_true, x_gen)),
            # "Wasserstein_Y": float(get_wasserstein_y(x_true, x_gen)),
        }

        Evaluator.logger.info("Evaluation Process Completed Successfully.")
        return results
