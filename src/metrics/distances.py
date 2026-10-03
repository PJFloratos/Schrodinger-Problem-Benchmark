import torch
from torchmetrics.image.fid import FrechetInceptionDistance

import numpy as np
from scipy.stats import wasserstein_distance

from typing import Sequence, Union, Optional, Callable, Dict, Tuple


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


@torch.no_grad()
def get_precision_recall(
    real_features: torch.Tensor, fake_features: torch.Tensor, k: int = 3
) -> Tuple[float, float]:
    """
    Computes Generative Precision and Recall using k-NN manifolds.
    - Precision: Fraction of fake samples that fall into the real data manifold (Quality).
    - Recall: Fraction of real samples that fall into the fake data manifold (Diversity).
    """
    device = fake_features.device
    real_features = real_features.to(device)

    # 1. Compute pairwise distance matrices
    D_rr = torch.cdist(real_features, real_features)
    D_ff = torch.cdist(fake_features, fake_features)
    D_rf = torch.cdist(real_features, fake_features)

    # 2. Estimate the manifold radii (distance to k-th nearest neighbor)
    # We use k + 1 because the 0-th nearest neighbor of a point to itself is always 0.
    radius_real = torch.topk(D_rr, k + 1, dim=1, largest=False).values[:, -1]
    radius_fake = torch.topk(D_ff, k + 1, dim=1, largest=False).values[:, -1]

    # 3. Precision: Fake points falling within ANY real point's radius
    # D_rf is (N_real, N_fake). We broadcast radius_real to compare.
    in_real_manifold = D_rf <= radius_real.unsqueeze(1)
    precision = in_real_manifold.any(dim=0).float().mean().item()

    # 4. Recall: Real points falling within ANY fake point's radius
    in_fake_manifold = D_rf <= radius_fake.unsqueeze(0)
    recall = in_fake_manifold.any(dim=1).float().mean().item()

    return precision, recall


def get_generative_quality_metrics(
    x_true: torch.Tensor, x_gen: torch.Tensor, k_nn: int = 3
) -> Dict[str, float]:
    device = x_gen.device  # the GPU (x_true was moved to CPU by the evaluator)

    # ---- 1. Precision & Recall (pixel space) ----
    flat_true = x_true.reshape(x_true.shape[0], -1).float().to(device)
    flat_gen = x_gen.reshape(x_gen.shape[0], -1).float()
    precision, recall = get_precision_recall(flat_true, flat_gen, k=k_nn)

    # ---- 2. FID ----
    if x_true.ndim == 3:  # (N, H, W) -> (N, 1, H, W)
        x_true = x_true.unsqueeze(1)
        x_gen = x_gen.unsqueeze(1)

    def to_uint8(t: torch.Tensor) -> torch.Tensor:
        t = (t.clamp(-1.0, 1.0) + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        return (t * 255.0).round().to(torch.uint8)

    def to_rgb(t: torch.Tensor) -> torch.Tensor:
        # Inception-v3 expects 3 channels
        return t.repeat(1, 3, 1, 1) if t.shape[1] == 1 else t

    x_true_u8 = to_uint8(x_true)
    x_gen_u8 = to_uint8(x_gen)

    fid = FrechetInceptionDistance(feature=64).to(device)
    chunk_size = 256

    for i in range(0, x_true_u8.shape[0], chunk_size):
        batch = to_rgb(x_true_u8[i : i + chunk_size]).to(device)
        fid.update(batch, real=True)

    for i in range(0, x_gen_u8.shape[0], chunk_size):
        batch = to_rgb(x_gen_u8[i : i + chunk_size]).to(device)
        fid.update(batch, real=False)

    fid_score = fid.cpu().compute().item()
    fid.reset()

    return {"FID": fid_score, "Precision": precision, "Recall": recall}
