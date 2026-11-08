import torch
from torch.utils.data import Dataset

import numpy as np
from sklearn.datasets import make_moons, make_swiss_roll
from sklearn.preprocessing import StandardScaler

import math
from typing import List, Tuple, Union, Iterable, Optional


class Toy2DDataset(Dataset):
    """
    2D toy target distribution (pi_1) for Schrödinger-bridge / generative solvers.

    Normalization: x_norm = (x_raw - mean) / scale, with ONE scalar `scale`
    (isotropic, preserves geometry) chosen so E||x_norm||^2 / d = 1, i.e. the
    data has the same second moment as N(0, I).
    Pass `stats=(mean, scale)` from a train set to build a val set consistently.
    """

    def __init__(
        self,
        n_samples: int = 10000,
        dataset_type: str = "moons",
        seed: int = 0,
        normalize: Optional[bool] = None,
        stats: Optional[Tuple[np.ndarray, float]] = None,
        # gaussian_mu: Tuple[float, float] = (5.0, 5.0),
        # gaussian_var: float = 2.0,
        gaussian_mu: Tuple[float, float] = (0.0, 0.0),
        gaussian_var: float = 0.0025,
    ) -> None:
        super().__init__()

        rng = np.random.default_rng(seed)
        self.dataset_type = dataset_type

        self.n_samples = n_samples

        if dataset_type == "moons":
            X, _ = make_moons(n_samples=n_samples, noise=0.05, random_state=seed)

        elif dataset_type == "swiss_roll":
            X, _ = make_swiss_roll(n_samples=n_samples, noise=0.5, random_state=seed)
            X = X[:, [0, 2]]  # drop the height axis -> 2D spiral

        elif dataset_type == "checkerboard":
            # Sample exactly n_samples: pick an allowed unit cell, then a uniform point in it
            cells = np.array(
                [(i, j) for i in range(4) for j in range(4) if (i + j) % 2 == 0]
            )
            idx = rng.integers(0, len(cells), size=n_samples)
            X = cells[idx] + rng.uniform(0.0, 1.0, size=(n_samples, 2))

        elif dataset_type == "gaussian":
            self.mu = np.asarray(gaussian_mu, dtype=np.float64)
            self.var = float(gaussian_var)
            X = self.mu + math.sqrt(self.var) * rng.standard_normal((n_samples, 2))

        elif dataset_type == "8gaussians":
            r = 2.0
            angles = np.arange(8) * (2 * np.pi / 8)
            centers = r * np.stack([np.cos(angles), np.sin(angles)], axis=1)
            idx = rng.integers(0, 8, size=n_samples)
            X = centers[idx] + 0.15 * rng.standard_normal((n_samples, 2))

        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        # Analytical Gaussian keeps its exact parameters unless explicitly asked otherwise
        if normalize is None:
            normalize = dataset_type != "gaussian"

        if normalize:
            if stats is None:
                mean = X.mean(axis=0)
                scale = float(
                    np.sqrt(((X - mean) ** 2).sum(axis=1).mean() / X.shape[1])
                )
            else:
                mean, scale = stats
        else:
            mean, scale = np.zeros(2), 1.0

        self.mean = np.asarray(mean, dtype=np.float64)
        self.scale = float(scale)
        self.data = torch.tensor((X - self.mean) / self.scale, dtype=torch.float32)

    ############################################################
    # Coordinate Mapping
    ############################################################

    @property
    def stats(self):
        return self.mean, self.scale

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + torch.as_tensor(
            self.mean, dtype=x.dtype, device=x.device
        )

    def eps_normalized(self, eps_raw: float) -> float:
        """Reference noise level to use in normalized space for a given raw-space eps."""
        return eps_raw / self.scale**2

    ############################################################
    # Dataset API
    ############################################################

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[index]

    def __len__(self) -> int:
        return self.n_samples

    def __str__(self) -> str:
        return f"Toy2DDataset(type={self.dataset_type}, samples={len(self)}, scale={self.scale:.3f})"
