from src.utils import configure_logger

import torch
from torch.utils.data import Dataset

import numpy as np
from sklearn.datasets import make_moons, make_swiss_roll
from sklearn.preprocessing import StandardScaler

import math
from typing import List, Tuple, Union, Iterable


class Toy2DDataset(Dataset):
    """
    A 2D toy dataset for validating the Generator Matching solver.
    """

    logger = configure_logger(__name__)

    def __init__(self, n_samples: int = 10000, dataset_type: str = "moons") -> None:
        super().__init__()

        self.n_samples = n_samples

        if dataset_type == "moons":
            X, _ = make_moons(n_samples=n_samples, noise=0.05)

        elif dataset_type == "swiss_roll":
            # make_swiss_roll returns 3D data; slice out the height dimension to get a 2D spiral
            X, _ = make_swiss_roll(n_samples=n_samples, noise=0.5)
            X = X[:, [0, 2]]

        elif dataset_type == "checkerboard":
            # Generate uniform points and filter them into alternating disjoint squares
            x1 = np.random.uniform(0, 4, n_samples * 3)
            x2 = np.random.uniform(0, 4, n_samples * 3)
            mask = (np.floor(x1) % 2 + np.floor(x2) % 2) % 2 == 0
            X = np.vstack([x1[mask], x2[mask]]).T[:n_samples]

        elif dataset_type == "gaussian":
            self.mu = torch.tensor([5.0, 5.0], dtype=torch.float32)
            X = torch.randn(self.n_samples, 2) * math.sqrt(2) + self.mu

        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        if dataset_type == "gaussian":
            # Skip scaling to preserve the exact analytical distribution
            self.data = torch.tensor(X, dtype=torch.float32)
        else:
            # Standardize the data so it roughly aligns with the N(0, I) prior
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.data = torch.tensor(X_scaled, dtype=torch.float32)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[index]

    def __len__(self) -> int:
        return self.n_samples

    def __str__(self) -> str:
        return f"Toy2DDataset(samples={self.n_samples})"
