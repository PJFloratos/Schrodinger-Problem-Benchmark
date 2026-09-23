import torch

import numpy as np
from scipy.stats import wasserstein_distance

from typing import Sequence, Union, Optional, Callable, Dict


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


@torch.no_grad()
def get_path_consistency_mse(
    f_model: torch.nn.Module,
    b_model: torch.nn.Module,
    x_probe: torch.Tensor,
    t_points: Sequence[float] = (0.25, 0.5, 0.75),
) -> float:
    """
    Probes alignment of forward and backward chains.
    Evaluates the squared difference between the forward drift and the reversed backward drift.
    """
    f_model.eval()
    b_model.eval()
    device = x_probe.device

    total_mse = 0.0
    for t_val in t_points:
        t_tensor = torch.full((x_probe.shape[0], 1), t_val, device=device)

        # Depending on your exact SB formulation, forward and backward drifts sum to a specific score.
        # This acts as a base divergence metric between the two vector fields.
        f_vec = f_model(x_probe, t_tensor)
        b_vec = b_model(x_probe, t_tensor)

        total_mse += torch.mean((f_vec + b_vec) ** 2).item()

    return total_mse / len(t_points)


@torch.no_grad()
def get_drift_mse(
    model: torch.nn.Module,
    ground_truth_v: Callable,
    x_probe: torch.Tensor,
    t_points: Sequence[float] = (0.1, 0.5, 0.9),
) -> float:
    """
    Compare against analytic v*(x,t) at multiple t-steps.
    """
    model.eval()
    device = x_probe.device

    total_mse = 0.0
    for t_val in t_points:
        t_tensor = torch.full((x_probe.shape[0], 1), t_val, device=device)
        pred_u = model(x_probe, t_tensor)
        true_u = ground_truth_v(x_probe, t_tensor)

        total_mse += torch.mean((pred_u - true_u) ** 2).item()

    return total_mse / len(t_points)


def get_generative_quality_metrics(
    x_true: torch.Tensor, x_gen: torch.Tensor
) -> Dict[str, float]:
    """
    Standard generative metrics for standard datasets.
    For production, wrap torchmetrics.image.fid.FrechetInceptionDistance here.
    """
    # Placeholder for torchmetrics integration
    return {"FID": 0.0, "Precision": 0.0, "Recall": 0.0}
