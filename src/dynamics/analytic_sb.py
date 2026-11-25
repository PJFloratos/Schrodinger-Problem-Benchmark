from __future__ import annotations

from src.dynamics.references import ReferenceProcess
from src.dynamics.utils import randn_like_with_gen

import torch

import math
from typing import Callable, Literal, Optional, Tuple, Union


CovType = Literal["isotropic", "diagonal", "full"]
PriorSampler = Callable[[int, torch.device, torch.Generator], torch.Tensor]


class AnalyticSchrodingerBridge:
    """
    Ground-truth Schrödinger Bridge benchmark constructed via a log-GMM terminal
    potential paired with an arbitrary affine linear SDE reference measure R:

        dX_t = (alpha(t) X_t + beta(t)) dt + sigma(t) dW_t
        psi^*(y) = log sum_{k=1}^K w_k N(y; mu_k, Sigma_k)

    Provides closed-form sampling from:
    - Prior mu:               x ~ mu
    - Conditional plan:       y ~ pi^*(. | x)
    - Joint coupling:         (x, y) ~ pi^*
    - Target marginal:        y ~ nu
    - Exact bridge paths:     X_t ~ P^*_{t | 0,1}(. | x_0, x_1) = R_{t | 0,1}(. | x_0, x_1)
    - Optimal drift field:    v^*(x, t)
    - Conditional moments:    (mu^*(x), Sigma^*(x)) and cBW_2^2-UVP
    """

    def __init__(
        self,
        ref: ReferenceProcess,
        means: torch.Tensor,
        covs: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        cov_type: CovType = "isotropic",
        prior_std: Optional[float] = None,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.ref = ref
        self.device = torch.device(device)
        self.dtype = dtype
        self.cov_type: CovType = cov_type
        self.prior_std = float(prior_std) if prior_std is not None else None

        if means.ndim != 2:
            raise ValueError(
                f"`means` must have shape (K, d), got {tuple(means.shape)}"
            )
        self.means = means.to(device=self.device, dtype=self.dtype)
        self.K, self.d = self.means.shape

        if weights is None:
            weights = torch.full(
                (self.K,), 1.0 / self.K, device=self.device, dtype=self.dtype
            )
        else:
            weights = weights.to(device=self.device, dtype=self.dtype)
            if weights.shape != (self.K,) or torch.any(weights <= 0):
                raise ValueError(
                    "`weights` must have shape (K,) with strictly positive entries."
                )
            weights = weights / weights.sum()
        self.log_weights = torch.log(weights)

        self.covs = self._validate_and_format_covs(covs)

        if self.cov_type == "full":
            covs_sym = 0.5 * (self.covs + self.covs.transpose(-1, -2))
            eigvals, eigvecs = torch.linalg.eigh(covs_sym)
            if torch.any(eigvals <= 0):
                raise ValueError(
                    "Full covariance matrices must be strictly positive definite."
                )
            self._cov_eigvals: Optional[torch.Tensor] = eigvals  # (K, d)
            self._cov_eigvecs: Optional[torch.Tensor] = eigvecs  # (K, d, d)
        else:
            self._cov_eigvals = None
            self._cov_eigvecs = None

        # Precompute squared centroid norms for matrix-free isotropic Mahalanobis evaluation
        self._means_norm_sq = (self.means**2).sum(dim=-1).unsqueeze(0)  # (1, K)

        # Cache t=0 -> t=1 posterior covariance parameters (S_k) and noise factors
        self._v01 = float(self.ref.V(0.0, 1.0))
        self._post_cov_param = self._compute_posterior_cov(self._v01)
        self._post_noise_factor = self._compute_posterior_noise_factor()

    def _validate_and_format_covs(self, covs: torch.Tensor) -> torch.Tensor:
        """
        Casts covs to the target device/dtype, verifies shape compatibility—(K,)
        for isotropic, (K, d) for diagonal, (K, d, d) for full, and enforces
        strict positive definiteness.
        """
        covs = covs.to(device=self.device, dtype=self.dtype)
        if self.cov_type == "isotropic":
            if covs.ndim == 0:
                covs = covs.expand(self.K).clone()
            elif covs.shape != (self.K,):
                raise ValueError(
                    f"Expected shape (K,) for isotropic covs, got {tuple(covs.shape)}"
                )
            if torch.any(covs <= 0):
                raise ValueError("Isotropic variances must be strictly positive.")

        elif self.cov_type == "diagonal":
            if covs.shape != (self.K, self.d):
                raise ValueError(
                    f"Expected shape (K, d) for diagonal covs, got {tuple(covs.shape)}"
                )
            if torch.any(covs <= 0):
                raise ValueError("Diagonal variances must be strictly positive.")

        elif self.cov_type == "full":
            if covs.shape != (self.K, self.d, self.d):
                raise ValueError(
                    f"Expected shape (K, d, d) for full covs, got {tuple(covs.shape)}"
                )
        else:
            raise ValueError(f"Unknown cov_type: {self.cov_type}")

        return covs

    def to(self, device: Union[str, torch.device]) -> AnalyticSchrodingerBridge:
        """
        Moves all stored parameter tensors and cached buffers to device and returns self.
        """
        self.device = torch.device(device)
        self.means = self.means.to(self.device)
        self.log_weights = self.log_weights.to(self.device)
        self.covs = self.covs.to(self.device)
        self._means_norm_sq = self._means_norm_sq.to(self.device)
        self._post_cov_param = self._post_cov_param.to(self.device)
        self._post_noise_factor = self._post_noise_factor.to(self.device)

        if self._cov_eigvals is not None:
            self._cov_eigvals = self._cov_eigvals.to(self.device)
            self._cov_eigvecs = self._cov_eigvecs.to(self.device)

        return self

    # ------------------------------------------------------------------
    # Internal Linear-Gaussian Mixture Algebra
    # ------------------------------------------------------------------
    def _compute_posterior_cov(self, v01: float) -> torch.Tensor:
        """
        Computes Computes the exact conditional covariance S_k of component k:
         S_k = Sigma_k - Sigma_k (Sigma_k + V_{01} I_d)^{-1} Sigma_k

        Using scalar/elementwise division for isotropic/diagonal types and the
        precomputed eigenspectrum for full covariances (symmetrizing the result).
        """
        if self.cov_type in ("isotropic", "diagonal"):
            return (self.covs * v01) / (self.covs + v01)

        post_eigvals = (self._cov_eigvals * v01) / (self._cov_eigvals + v01)  # (K, d)
        S = (
            self._cov_eigvecs * post_eigvals.unsqueeze(1)
        ) @ self._cov_eigvecs.transpose(-1, -2)

        return 0.5 * (S + S.transpose(-1, -2))

    def _compute_posterior_noise_factor(self) -> torch.Tensor:
        """Precomputes sqrt(S_k) (isotropic/diagonal) or Cholesky L_{S_k} (full)."""
        if self.cov_type in ("isotropic", "diagonal"):
            return torch.sqrt(self._post_cov_param)

        return torch.linalg.cholesky(self._post_cov_param)

    def _posterior_weights(
        self, x_query: torch.Tensor, v_t1: Union[float, torch.Tensor]
    ) -> torch.Tensor:
        """
        Evaluates normalized responsibilities w_hat_k(x, t) without materializing
        (B, K, d) tensors whenever v_t1 is well-conditioned (>= 1e-2).
        """
        B = x_query.shape[0]
        if isinstance(v_t1, torch.Tensor):
            if v_t1.ndim == 0:
                min_v = float(v_t1.item())
                v_t1_t = torch.tensor([[min_v]], device=self.device, dtype=self.dtype)
            else:
                v_t1_t = v_t1.to(device=self.device, dtype=self.dtype).view(B, 1)
                min_v = float(v_t1_t.min().item())
        else:
            min_v = float(v_t1)
            v_t1_t = torch.tensor([[min_v]], device=self.device, dtype=self.dtype)

        if self.cov_type == "isotropic":
            total_var = self.covs.unsqueeze(0) + v_t1_t  # (1, K) or (B, K)
            log_det = self.d * torch.log(total_var)
            if min_v >= 1e-2:
                x_norm_sq = (x_query**2).sum(dim=-1, keepdim=True)  # (B, 1)
                sq_dist = (
                    x_norm_sq + self._means_norm_sq - 2.0 * (x_query @ self.means.T)
                ).clamp_min(0.0)
                sq_mahalanobis = sq_dist / total_var
            else:
                diff = x_query.unsqueeze(1) - self.means.unsqueeze(0)
                sq_mahalanobis = (diff**2).sum(dim=-1) / total_var

        elif self.cov_type == "diagonal":
            if v_t1_t.shape[0] == 1 and min_v >= 1e-2:
                tot_var_kd = self.covs + v_t1_t.item()  # (K, d)
                log_det = torch.log(tot_var_kd).sum(dim=-1).unsqueeze(0)
                inv_var = 1.0 / tot_var_kd  # (K, d)
                inv_var_means = inv_var * self.means  # (K, d)
                quad_means = (inv_var_means * self.means).sum(dim=-1).unsqueeze(0)
                sq_mahalanobis = (
                    (x_query**2) @ inv_var.T
                    + quad_means
                    - 2.0 * (x_query @ inv_var_means.T)
                ).clamp_min(0.0)
            else:
                total_var = self.covs.unsqueeze(0) + v_t1_t.unsqueeze(-1)
                diff = x_query.unsqueeze(1) - self.means.unsqueeze(0)
                sq_mahalanobis = ((diff**2) / total_var).sum(dim=-1)
                log_det = torch.log(total_var).sum(dim=-1)

        else:  # full
            weights, _ = self._marginal_precision_stats(x_query, v_t1)
            return weights

        log_unnorm = self.log_weights.unsqueeze(0) - 0.5 * (log_det + sq_mahalanobis)

        return torch.softmax(log_unnorm, dim=-1)

    def _marginal_precision_stats(
        self, x_query: torch.Tensor, v_t1: Union[float, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates normalized responsibilities w_hat_k(x, t) and precision-scaled residuals:
            scaled_diff_k = (Sigma_k + V_{t1} I_d)^{-1} (mu_{t1}(x) - mu_k)
        """
        B = x_query.shape[0]
        if isinstance(v_t1, torch.Tensor):
            if v_t1.ndim == 0:
                min_v = float(v_t1.item())
                v_t1_t = torch.tensor([[min_v]], device=self.device, dtype=self.dtype)
            else:
                v_t1_t = v_t1.to(device=self.device, dtype=self.dtype).view(B, 1)
                min_v = float(v_t1_t.min().item())
        else:
            min_v = float(v_t1)
            v_t1_t = torch.tensor([[min_v]], device=self.device, dtype=self.dtype)

        diff = x_query.unsqueeze(1) - self.means.unsqueeze(0)

        if self.cov_type == "isotropic":
            total_var = self.covs.unsqueeze(0) + v_t1_t
            scaled_diff = diff / total_var.unsqueeze(-1)
            sq_mahalanobis = (diff * scaled_diff).sum(dim=-1)
            log_det = self.d * torch.log(total_var)

        elif self.cov_type == "diagonal":
            total_var = self.covs.unsqueeze(0) + v_t1_t.unsqueeze(-1)
            scaled_diff = diff / total_var
            sq_mahalanobis = (diff * scaled_diff).sum(dim=-1)
            log_det = torch.log(total_var).sum(dim=-1)

        else:  # full
            eye = torch.eye(self.d, device=self.device, dtype=self.dtype)
            if v_t1_t.shape[0] == 1:
                total_cov = self.covs + v_t1_t.item() * eye.unsqueeze(0)
                L = torch.linalg.cholesky(total_cov)
                log_det = 2.0 * torch.diagonal(L, dim1=-2, dim2=-1).log().sum(
                    dim=-1
                ).unsqueeze(0)
            else:
                total_cov = self.covs.unsqueeze(0) + v_t1_t.view(B, 1, 1, 1) * eye.view(
                    1, 1, self.d, self.d
                )
                L = torch.linalg.cholesky(total_cov)
                log_det = 2.0 * torch.diagonal(L, dim1=-2, dim2=-1).log().sum(dim=-1)

            scaled_diff = torch.cholesky_solve(diff.unsqueeze(-1), L).squeeze(-1)
            sq_mahalanobis = (diff * scaled_diff).sum(dim=-1)

        log_unnorm = self.log_weights.unsqueeze(0) - 0.5 * (log_det + sq_mahalanobis)
        weights = torch.softmax(log_unnorm, dim=-1)

        return weights, scaled_diff

    def _conditional_gmm_params(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes state-dependent weights tilde_w_k(x) and posterior means m_k(x)
        of pi^*(y | x) = sum_{k=1}^K tilde_w_k(x) N(y; m_k(x), S_k).
        """
        mu_01 = self.ref.mu(x, 0.0, 1.0)
        weights, scaled_diff = self._marginal_precision_stats(mu_01, self._v01)

        if self.cov_type == "isotropic":
            shift = self.covs.view(1, self.K, 1) * scaled_diff
        elif self.cov_type == "diagonal":
            shift = self.covs.unsqueeze(0) * scaled_diff
        else:
            shift = torch.einsum("kij,bkj->bki", self.covs, scaled_diff)

        post_means = self.means.unsqueeze(0) + shift

        return weights, post_means

    @torch.no_grad()
    def effective_num_modes(
        self, x: torch.Tensor, chunk_size: int = 512
    ) -> torch.Tensor:
        """
        Computes posterior perplexity K_eff(x) = exp(H(tilde_w(. | x))) in [1, K]
        """
        x = x.to(device=self.device, dtype=self.dtype)
        B = x.shape[0]
        out = torch.empty((B,), device=self.device, dtype=self.dtype)

        for start in range(0, B, chunk_size):
            mu_01 = self.ref.mu(x[start : start + chunk_size], 0.0, 1.0)
            weights = self._posterior_weights(mu_01, self._v01)
            entropy = torch.special.entr(weights).sum(dim=-1)
            out[start : start + chunk_size] = torch.exp(entropy)

        return out

    # ------------------------------------------------------------------
    # Core Benchmark Sampling API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_prior(
        self, n_samples: int, gen: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """Draws initial samples x ~ mu."""
        if self.prior_std is None:
            return self.ref.sample_prior(n_samples, (self.d,), self.device, gen).to(
                dtype=self.dtype
            )
        z = torch.randn(
            (n_samples, self.d), device=self.device, dtype=self.dtype, generator=gen
        )
        return self.prior_std * z

    @torch.no_grad()
    def sample_conditional(
        self,
        x: torch.Tensor,
        gen: Optional[torch.Generator] = None,
        chunk_size: int = 512,
    ) -> torch.Tensor:
        """
        Two-step ancestral sampling from the exact conditional plan pi^*(. | x):
        1. Routing step:    k^* ~ Categorical(tilde_w_1(x), ..., tilde_w_K(x))
        2. Generative step: y ~ N(m_{k^*}(x), S_{k^*})
        """
        x = x.to(device=self.device, dtype=self.dtype)
        B = x.shape[0]
        if B == 0:
            return torch.empty((0, self.d), device=self.device, dtype=self.dtype)

        out = torch.empty((B, self.d), device=self.device, dtype=self.dtype)
        eye = (
            torch.eye(self.d, device=self.device, dtype=self.dtype).unsqueeze(0)
            if self.cov_type == "full"
            else None
        )

        for start in range(0, B, chunk_size):
            x_chunk = x[start : start + chunk_size]
            mu_01 = self.ref.mu(x_chunk, 0.0, 1.0)
            weights = self._posterior_weights(mu_01, self._v01)

            k_star = torch.multinomial(
                weights, num_samples=1, replacement=True, generator=gen
            ).squeeze(-1)

            mu_k = self.means[k_star]  # (chunk, d)
            diff_k = mu_01 - mu_k

            if self.cov_type == "isotropic":
                cov_k = self.covs[k_star].unsqueeze(-1)
                chosen_mean = mu_k + (cov_k / (cov_k + self._v01)) * diff_k
                noise = self._post_noise_factor[k_star].unsqueeze(
                    -1
                ) * randn_like_with_gen(chosen_mean, gen)
            elif self.cov_type == "diagonal":
                cov_k = self.covs[k_star]
                chosen_mean = mu_k + (cov_k / (cov_k + self._v01)) * diff_k
                noise = self._post_noise_factor[k_star] * randn_like_with_gen(
                    chosen_mean, gen
                )
            else:
                Q_k = self._cov_eigvecs[k_star]  # (chunk, d, d)
                lam_k = self._cov_eigvals[k_star]  # (chunk, d)
                proj = torch.bmm(Q_k.transpose(-1, -2), diff_k.unsqueeze(-1)).squeeze(
                    -1
                )
                scaled = (lam_k / (lam_k + self._v01)) * proj
                chosen_mean = mu_k + torch.bmm(Q_k, scaled.unsqueeze(-1)).squeeze(-1)
                L_post = self._post_noise_factor[k_star]
                z = randn_like_with_gen(chosen_mean, gen)
                noise = torch.bmm(L_post, z.unsqueeze(-1)).squeeze(-1)

            out[start : start + chunk_size] = chosen_mean + noise

        return out

    @torch.no_grad()
    def sample_joint(
        self, n_samples: int, gen: Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Draws paired ground-truth coupling samples (x, y) ~ pi^*
        """
        x = self.sample_prior(n_samples, gen=gen)
        y = self.sample_conditional(x, gen=gen)

        return x, y

    @torch.no_grad()
    def sample_target(
        self, n_samples: int, gen: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        Draws marginal target samples y ~ nu = int pi^*(. | x) dmu(x)
        """
        _, y = self.sample_joint(n_samples, gen=gen)

        return y

    @torch.no_grad()
    def sample_bridge_trajectory(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: Union[float, torch.Tensor],
        gen: Optional[torch.Generator] = None,
        return_noise: bool = False,
    ) -> torch.Tensor:
        """
        Exact intermediate state X_t ~  R_{t|0,1}(. | x0, x1),
        enabling discretization-free Forward KL evaluation.
        """
        return self.ref.sample_bridge(x0, x1, t, gen=gen, return_noise=return_noise)

    # ------------------------------------------------------------------
    # Closed-Form Optimal Drift & Evaluation Metrics
    # ------------------------------------------------------------------
    @torch.no_grad()
    def optimal_drift(
        self,
        x: torch.Tensor,
        t: Union[float, torch.Tensor],
        chunk_size: int = 512,
    ) -> torch.Tensor:
        """
        Evaluates the exact optimal Schrödinger Bridge drift vector field (Eq. 5):
            v^*(x, t) = f(x, t) - sigma^2(t) A_{t1} sum_{k=1}^K w_hat_k(x, t) (Sigma_k + V_{t1} I_d)^{-1} (mu_{t1}(x) - mu_k)
        """
        if isinstance(t, torch.Tensor) and t.ndim == 0:
            t = float(t.item())

        x = x.to(device=self.device, dtype=self.dtype)
        B = x.shape[0]
        if B == 0:
            return torch.empty((0, self.d), device=self.device, dtype=self.dtype)

        is_tensor_t = isinstance(t, torch.Tensor) and t.ndim > 0
        out = torch.empty((B, self.d), device=self.device, dtype=self.dtype)

        for start in range(0, B, chunk_size):
            x_chunk = x[start : start + chunk_size]
            t_chunk = t[start : start + chunk_size] if is_tensor_t else t
            chunk_b = x_chunk.shape[0]

            mu_t1 = self.ref.mu(x_chunk, t_chunk, 1.0)
            v_t1 = self.ref.V(t_chunk, 1.0)

            if isinstance(v_t1, torch.Tensor):
                v_t1_t = v_t1.to(device=self.device, dtype=self.dtype).view(chunk_b, 1)
                min_v = float(v_t1_t.min().item())
            else:
                min_v = float(v_t1)
                v_t1_t = torch.tensor([[min_v]], device=self.device, dtype=self.dtype)

            # Fast matrix-free GEMM path when V_{t1} >= 1e-2 (avoids (B, K, d) allocation)
            if self.cov_type == "isotropic" and min_v >= 1e-2:
                total_var = self.covs.unsqueeze(0) + v_t1_t
                x_norm_sq = (mu_t1**2).sum(dim=-1, keepdim=True)
                sq_dist = (
                    x_norm_sq + self._means_norm_sq - 2.0 * (mu_t1 @ self.means.T)
                ).clamp_min(0.0)
                log_unnorm = self.log_weights.unsqueeze(0) - 0.5 * (
                    self.d * torch.log(total_var) + sq_dist / total_var
                )
                weights = torch.softmax(log_unnorm, dim=-1)
                alpha = weights / total_var  # (chunk_b, K)
                expected_score = alpha.sum(dim=-1, keepdim=True) * mu_t1 - (
                    alpha @ self.means
                )
            elif self.cov_type == "diagonal" and v_t1_t.shape[0] == 1 and min_v >= 1e-2:
                tot_var_kd = self.covs + v_t1_t.item()
                inv_var = 1.0 / tot_var_kd
                inv_var_means = inv_var * self.means
                quad_means = (inv_var_means * self.means).sum(dim=-1).unsqueeze(0)
                sq_mahal = (
                    (mu_t1**2) @ inv_var.T
                    + quad_means
                    - 2.0 * (mu_t1 @ inv_var_means.T)
                ).clamp_min(0.0)
                log_unnorm = self.log_weights.unsqueeze(0) - 0.5 * (
                    torch.log(tot_var_kd).sum(dim=-1).unsqueeze(0) + sq_mahal
                )
                weights = torch.softmax(log_unnorm, dim=-1)
                expected_score = (weights @ inv_var) * mu_t1 - (weights @ inv_var_means)
            else:
                weights, scaled_diff = self._marginal_precision_stats(mu_t1, v_t1)
                expected_score = (weights.unsqueeze(-1) * scaled_diff).sum(dim=1)

            f_xt = self.ref.drift(x_chunk, t_chunk)
            sig_sq = self.ref._bcast_val(self.ref.sigma_sq(t_chunk), x_chunk)
            a_t1 = self.ref._bcast_val(self.ref.A(t_chunk, 1.0), x_chunk)

            out[start : start + chunk_size] = f_xt - sig_sq * a_t1 * expected_score

        return out

    @torch.no_grad()
    def conditional_moments(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the exact analytical conditional mean mu^*(x) and covariance Sigma^*(x)
        of pi^*(. | x) for static cBW_2^2-UVP evaluation.
        """
        x = x.to(device=self.device, dtype=self.dtype)
        B = x.shape[0]
        weights, post_means = self._conditional_gmm_params(x)  # (B, K), (B, K, d)

        mean = (weights.unsqueeze(-1) * post_means).sum(dim=1)  # (B, d)
        centered = post_means - mean.unsqueeze(1)  # (B, K, d)
        between_cov = torch.einsum("bk,bki,bkj->bij", weights, centered, centered)

        if self.cov_type == "isotropic":
            within_var = (weights * self._post_cov_param.unsqueeze(0)).sum(dim=1)
            eye = torch.eye(self.d, device=self.device, dtype=self.dtype).unsqueeze(0)
            cov = between_cov + within_var.view(B, 1, 1) * eye
        elif self.cov_type == "diagonal":
            within_diag = (
                weights.unsqueeze(-1) * self._post_cov_param.unsqueeze(0)
            ).sum(dim=1)
            cov = between_cov + torch.diag_embed(within_diag)
        else:
            within_cov = torch.einsum("bk,kij->bij", weights, self._post_cov_param)
            cov = between_cov + within_cov

        return mean, cov

    @torch.no_grad()
    def compute_cbw2_uvp(
        self,
        solver_sampler: Callable[[torch.Tensor, int], torch.Tensor],
        n_test_x: int = 1000,
        n_cond_samples: int = 1000,
        n_target_var_samples: int = 50_000,
        chunk_size: int = 64,
        gen: Optional[torch.Generator] = None,
    ) -> float:
        """
        Computes the Conditional Bures-Wasserstein Unexplained Variance Percentage:
            cBW_2^2-UVP = 100% / (0.5 * Tr(Cov(nu))) * E_{x ~ mu}[ BW_2^2(hat_pi_x, pi^*_x) ]

        Args:
            solver_sampler: callable taking (x_batch: [M, d], n_samples: int) -> [M, n_samples, d]
        """
        y_ref = self.sample_target(n_target_var_samples, gen=gen)
        target_total_var = float(y_ref.var(dim=0, unbiased=True).sum().item())

        x_test = self.sample_prior(n_test_x, gen=gen)
        bw2_total = 0.0
        eff_chunk = min(chunk_size, max(8, 16_384 // self.d))

        for start in range(0, n_test_x, eff_chunk):
            x_chunk = x_test[start : start + eff_chunk]

            mu_true, cov_true = self.conditional_moments(x_chunk)
            y_hat = solver_sampler(x_chunk, n_cond_samples).to(
                device=self.device, dtype=self.dtype
            )  # (M, S, d)

            mu_hat = y_hat.mean(dim=1)
            diff_hat = y_hat - mu_hat.unsqueeze(1)
            cov_hat = torch.einsum("msd,mse->mde", diff_hat, diff_hat) / max(
                n_cond_samples - 1, 1
            )

            mean_sq_err = ((mu_hat - mu_true) ** 2).sum(dim=-1)

            # Cholesky similarity transform: eigvals(L_true^T cov_hat L_true) == eigvals(cov_true^{1/2} cov_hat cov_true^{1/2})
            L_true, info = torch.linalg.cholesky_ex(cov_true, check_errors=False)
            if torch.all(info == 0):
                sandwich = L_true.transpose(-1, -2) @ cov_hat @ L_true
            else:
                eigvals_t, eigvecs_t = torch.linalg.eigh(cov_true)
                sqrt_cov_true = (
                    eigvecs_t * torch.sqrt(eigvals_t.clamp_min(0.0)).unsqueeze(1)
                ) @ eigvecs_t.transpose(-1, -2)
                sandwich = sqrt_cov_true @ cov_hat @ sqrt_cov_true

            sandwich = 0.5 * (sandwich + sandwich.transpose(-1, -2))
            cross_eigvals = torch.linalg.eigvalsh(sandwich).clamp_min(0.0)
            trace_term = (
                torch.diagonal(cov_hat, dim1=-2, dim2=-1).sum(dim=-1)
                + torch.diagonal(cov_true, dim1=-2, dim2=-1).sum(dim=-1)
                - 2.0 * torch.sqrt(cross_eigvals).sum(dim=-1)
            )

            bw2_total += float((mean_sq_err + trace_term).sum().item())

        mean_bw2 = bw2_total / n_test_x

        return 100.0 * mean_bw2 / (0.5 * target_total_var)
