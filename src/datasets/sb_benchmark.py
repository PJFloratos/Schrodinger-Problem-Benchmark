from __future__ import annotations

from src.dynamics.references import ReferenceProcess
from src.dynamics.analytic_sb import AnalyticSchrodingerBridge, CovType

import torch
from torch.utils.data import Dataset

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture

import math
from typing import Optional, Tuple, Union, Literal


GeometryType = Literal["sphere", "manifold"]


class SBBenchmarkDataset(Dataset):
    """
    Thin PyTorch Dataset wrapper around `AnalyticSchrodingerBridge` for standard
    DataLoader training loops. Generates a fixed finite split of (x, y) ~ pi^*.
    """

    def __init__(
        self,
        bridge: AnalyticSchrodingerBridge,
        n_samples: int = 50_000,
        seed: int = 0,
        paired: bool = False,
    ) -> None:
        super().__init__()
        self.bridge = bridge
        self.n_samples = n_samples
        self.paired = paired

        gen = torch.Generator(device=bridge.device).manual_seed(seed)
        self.x, self.y = bridge.sample_joint(n_samples, gen=gen)

    # ------------------------------------------------------------------
    # Internal Helpers & Dataset API
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(
        self, idx: int
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.paired:
            return self.x[idx], self.y[idx]
        return self.y[idx]

    # ------------------------------------------------------------------
    # Regime-Specific Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_synthetic(
        cls,
        ref: ReferenceProcess,
        d: int,
        geometry: GeometryType = "sphere",
        n_components: int = 5,
        cov_type: CovType = "isotropic",
        var: float = 0.25,
        radius: float = 5.0,
        prior_std: Optional[float] = None,
        n_samples: int = 50_000,
        seed: int = 0,
        bridge_seed: int = 42,
        paired: bool = False,
        device: Union[str, torch.device] = "cpu",
    ) -> SBBenchmarkDataset:
        """
        Synthetic SB benchmark:
          - geometry='sphere': K=1 unimodal Gaussian or K>1 spherical GMM.
          - geometry='manifold': K components arranged along a nonlinear curve
            (modulated spirograph spiral in d=2; isometrically embedded Fourier
            manifold in d>2).
        """
        if cov_type == "full" and d > 256:
            raise ValueError(
                f"cov_type='full' is O(K*d^2) and unsupported for d={d} > 256."
            )

        gen = torch.Generator(device="cpu").manual_seed(bridge_seed)
        weights: Optional[torch.Tensor] = None

        if geometry == "sphere":
            if n_components == 1:
                means = torch.full((1, d), radius / math.sqrt(d))
            elif d == 2:
                angles = torch.arange(n_components, dtype=torch.float32) * (
                    2.0 * math.pi / n_components
                )
                means = radius * torch.stack(
                    [torch.cos(angles), torch.sin(angles)], dim=1
                )
            else:
                raw = torch.randn((n_components, d), generator=gen)
                means = radius * (raw / raw.norm(dim=-1, keepdim=True))

        elif geometry == "manifold":
            u = torch.linspace(0.0, 2.0 * math.pi, n_components + 1)[:-1]
            if d == 2:
                # 3.5-turn star-modulated spiral with sharp, well-separated turns
                turns = 3.5
                theta = turns * u
                lobe = 1.0 + 0.18 * torch.sin(6.0 * theta)
                r = radius * (0.28 + 0.72 * (u / (2.0 * math.pi))) * lobe
                means = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)
            else:
                # Multi-frequency twisted Fourier knot in R^m isometrically embedded in R^d
                m = min(d, 8)
                freqs = torch.tensor(
                    [1.0, 2.0, 3.0, 5.0][: max(1, m // 2)], dtype=torch.float32
                )
                phases = u.unsqueeze(1) * freqs.unsqueeze(0)
                latent = torch.cat([torch.cos(phases), torch.sin(phases)], dim=1)
                if latent.shape[1] < m:
                    latent = torch.cat([latent, torch.sin(7.0 * u).unsqueeze(1)], dim=1)
                latent = latent[:, :m]
                latent = radius * (
                    latent / latent.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                )

                Q_embed, _ = torch.linalg.qr(torch.randn((d, m), generator=gen))
                means = latent @ Q_embed.T

            # Compensate weights for the Gaussian reference marginal decay exp(-||mu_k||^2 / (2 * sigma_marg^2))
            r0_var = (
                float(prior_std) ** 2
                if prior_std is not None
                else float(ref.r0_std) ** 2
            )
            sigma_marg_sq = (float(ref.A(0.0, 1.0)) ** 2) * r0_var + float(
                ref.V(0.0, 1.0)
            )
            log_w = 0.5 * means.pow(2).sum(dim=-1) / (var + sigma_marg_sq)
            weights = torch.softmax(log_w, dim=0)
        else:
            raise ValueError(f"Unknown synthetic geometry: {geometry}")

        # 2. Component covariances
        if cov_type == "isotropic":
            covs = torch.full((n_components,), var)
        elif cov_type == "diagonal":
            covs = var * torch.exp(
                torch.empty((n_components, d)).uniform_(-0.7, 0.7, generator=gen)
            )
        elif cov_type == "full":
            Q, _ = torch.linalg.qr(torch.randn((n_components, d, d), generator=gen))
            eigvals = var * torch.exp(
                torch.empty((n_components, d)).uniform_(-0.7, 0.7, generator=gen)
            )
            covs = torch.einsum("kij,kj,klj->kil", Q, eigvals, Q)
        else:
            raise ValueError(f"Unknown cov_type: {cov_type}")

        bridge = AnalyticSchrodingerBridge(
            ref=ref,
            means=means,
            covs=covs,
            weights=weights,
            cov_type=cov_type,
            prior_std=prior_std,
            device=device,
        )
        return cls(bridge=bridge, n_samples=n_samples, seed=seed, paired=paired)

    @classmethod
    def from_empirical(
        cls,
        ref: ReferenceProcess,
        target_data: Union[torch.Tensor, Dataset],
        n_components: int = 100,
        cov_type: CovType = "isotropic",
        var: float = 0.02,
        prior_std: Optional[float] = None,
        n_samples: int = 50_000,
        seed: int = 0,
        bridge_seed: int = 42,
        paired: bool = False,
        device: Union[str, torch.device] = "cpu",
    ) -> SBBenchmarkDataset:
        """Heuristic benchmark fitted to empirical target samples via K-means or GMM."""
        if isinstance(target_data, torch.Tensor):
            raw = target_data
        elif hasattr(target_data, "data") and torch.is_floating_point(
            torch.as_tensor(target_data.data)
        ):
            raw = torch.as_tensor(target_data.data)
        else:
            # Uses __getitem__ so torchvision transforms (e.g. MNIST normalization) are applied
            raw = torch.stack(
                [item[0] if isinstance(item, tuple) else item for item in target_data]
            )

        flat = raw.detach().cpu().float().reshape(raw.shape[0], -1)
        N, d = flat.shape

        if cov_type == "isotropic":
            km = MiniBatchKMeans(
                n_clusters=n_components,
                random_state=bridge_seed,
                batch_size=min(4096, N),
                n_init="auto",
            ).fit(flat.numpy())

            counts = np.bincount(km.labels_, minlength=n_components).astype(np.float32)
            base_weights = torch.from_numpy((counts + 1.0) / (counts + 1.0).sum())
            means = torch.tensor(km.cluster_centers_, dtype=torch.float32)
            covs = torch.full((n_components,), var)

        elif cov_type in ("diagonal", "full"):
            if cov_type == "full" and d > 256:
                raise ValueError(
                    f"cov_type='full' is O(K*d^2) and unsupported for d={d} > 256."
                )
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type="diag" if cov_type == "diagonal" else "full",
                reg_covar=var,
                random_state=bridge_seed,
            ).fit(flat.numpy())

            base_weights = torch.tensor(gmm.weights_, dtype=torch.float32).clamp_min(
                1e-8
            )
            means = torch.tensor(gmm.means_, dtype=torch.float32)
            covs = torch.tensor(gmm.covariances_, dtype=torch.float32)

        else:
            raise ValueError(f"Unknown cov_type: {cov_type}")

        # Cancel Gaussian marginal attenuation N(b_{01}; mu_k, Sigma_k + (V_{01} + A_{01}^2 sigma_0^2) I)
        # so induced marginal nu preserves the empirical cluster frequencies base_weights.
        prior_var = (
            float(prior_std) ** 2 if prior_std is not None else float(ref.r0_std) ** 2
        )
        a01 = float(ref.A(0.0, 1.0))
        b01 = float(ref.B(0.0, 1.0))
        v01 = float(ref.V(0.0, 1.0))
        eff_noise_var = v01 + (a01**2) * prior_var

        diff = means - b01
        if cov_type == "isotropic":
            tot_var = covs + eff_noise_var
            mahal = (diff**2).sum(dim=-1) / tot_var
            log_det = d * torch.log(tot_var)
        elif cov_type == "diagonal":
            tot_var = covs + eff_noise_var
            mahal = ((diff**2) / tot_var).sum(dim=-1)
            log_det = torch.log(tot_var).sum(dim=-1)
        else:
            eye = torch.eye(d, dtype=torch.float32).unsqueeze(0)
            L = torch.linalg.cholesky(covs + eff_noise_var * eye)
            solved = torch.cholesky_solve(diff.unsqueeze(-1), L).squeeze(-1)
            mahal = (diff * solved).sum(dim=-1)
            log_det = 2.0 * torch.diagonal(L, dim1=-2, dim2=-1).log().sum(dim=-1)

        log_w = torch.log(base_weights) + 0.5 * (mahal + log_det)
        weights = torch.softmax(log_w, dim=0)

        bridge = AnalyticSchrodingerBridge(
            ref=ref,
            means=means,
            covs=covs,
            weights=weights,
            cov_type=cov_type,
            prior_std=prior_std,
            device=device,
        )

        return cls(bridge=bridge, n_samples=n_samples, seed=seed, paired=paired)
