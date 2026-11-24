from src.dynamics.utils import randn_like_with_gen

import torch

import math
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Type, Union


TimeLike = Union[float, torch.Tensor]
ScalarOrTensor = Union[float, torch.Tensor]


class ReferenceProcess(ABC):
    """
    Base class for zero-mean homogeneous linear SDE reference measures R on [0, 1]:
        dX_t = alpha(t) X_t dt + sigma(t) dW_t

    Covers:
      - Scaled Brownian Motion (R_sigma): dX_t = sigma dW_t
      - Variance-Preserving OU (R_gamma): dX_t = -gamma X_t dt + sqrt(2*gamma) dW_t
    """

    name: str = "base"

    @staticmethod
    def _bcast_val(val: ScalarOrTensor, ref: torch.Tensor) -> ScalarOrTensor:
        """
        Broadcasts scalars, (B,) / (B, 1) time tensors, or (D,) / (C, H, W) state tensors
        to align with `ref` of shape (B, ...).
        """
        if not isinstance(val, torch.Tensor):
            return float(val)

        val = val.to(device=ref.device, dtype=ref.dtype)
        B = ref.shape[0]
        bcast_time = (B, *([1] * (ref.ndim - 1)))

        if val.ndim == 0:
            return val
        if val.ndim == 1 and val.shape[0] == B and ref.ndim > 1:
            return val.view(bcast_time)
        if val.ndim == 2 and val.shape == (B, 1) and ref.ndim > 2:
            return val.view(bcast_time)
        if val.ndim == ref.ndim - 1:
            return val.unsqueeze(0)
        return val

    # ------------------------------------------------------------------
    # Elementary Linear SDE Primitives
    # ------------------------------------------------------------------
    @abstractmethod
    def alpha(self, t: TimeLike = 0.0) -> ScalarOrTensor:
        """Linear drift coefficient alpha(t)."""

    @abstractmethod
    def beta(self, t: TimeLike = 0.0) -> ScalarOrTensor:
        """Deterministic drift shift beta(t) in R^d."""

    @abstractmethod
    def sigma(self, t: TimeLike = 0.0) -> ScalarOrTensor:
        """Diffusion volatility sigma(t)."""

    @abstractmethod
    def sigma_sq(self, t: TimeLike = 0.0) -> ScalarOrTensor:
        """Squared diffusion volatility sigma^2(t)."""

    @abstractmethod
    def A(self, s: TimeLike, t: TimeLike) -> ScalarOrTensor:
        """State transition multiplier A_{st} = exp(\\int_s^t alpha(u) du)."""

    @abstractmethod
    def B(self, s: TimeLike, t: TimeLike) -> ScalarOrTensor:
        """Integrated shift B_{st} = \\int_s^t A_{tau, t} beta(tau) dtau."""

    @abstractmethod
    def V(self, s: TimeLike, t: TimeLike) -> ScalarOrTensor:
        """Transition variance V_{st} = \\int_s^t A_{tau, t}^2 sigma^2(tau) dtau."""

    @property
    @abstractmethod
    def r0_mean(self) -> ScalarOrTensor:
        """Mean of the initial/prior isotropic Gaussian R_0 = N(r0_mean, r0_std^2 I)."""

    @property
    @abstractmethod
    def r0_std(self) -> float:
        """Standard deviation of the initial/prior isotropic Gaussian R_0 = N(r0_mean, r0_std^2 I)."""

    # ------------------------------------------------------------------
    # Derived Drift & Transition Moments
    # ------------------------------------------------------------------
    def drift(self, x: torch.Tensor, t: TimeLike = 0.0) -> torch.Tensor:
        """Forward reference drift f_t(x) = alpha(t) * x + beta(t)."""
        a_t = self._bcast_val(self.alpha(t), x)
        b_t = self._bcast_val(self.beta(t), x)

        return a_t * x + b_t

    def mu(self, x_s: torch.Tensor, s: TimeLike, t: TimeLike) -> torch.Tensor:
        """Conditional transition mean mu_{st}(x_s) = A_{st} * x_s + B_{st}."""
        a_st = self._bcast_val(self.A(s, t), x_s)
        b_st = self._bcast_val(self.B(s, t), x_s)

        return a_st * x_s + b_st

    # ------------------------------------------------------------------
    # Transition Kernel & Scores: R_{t|s}(. | x_s)
    # ------------------------------------------------------------------
    def sample_prior(
        self,
        batch_size: int,
        shape: Tuple[int, ...],
        device: torch.device,
        gen: torch.Generator,
        memory_format: torch.memory_format = torch.contiguous_format,
    ) -> torch.Tensor:
        """Samples x ~ R_0 = N(r0_mean, r0_std^2 I_d)."""
        z = torch.randn(batch_size, *shape, device=device, generator=gen).contiguous(
            memory_format=memory_format
        )
        mean = self._bcast_val(self.r0_mean, z)
        std = self._bcast_val(self.r0_std, z)

        return mean + std * z

    def sample_conditional(
        self,
        x_s: torch.Tensor,
        s: TimeLike,
        t: TimeLike,
        gen: torch.Generator,
        memory_format: torch.memory_format = torch.contiguous_format,
    ) -> torch.Tensor:
        """Samples x_t ~ R_{t|s}(. | x_s) = N(mu_{st}(x_s), V_{st} I_d)."""
        mean = self.mu(x_s, s, t)
        v_st = self._bcast_val(self.V(s, t), x_s)
        std = (
            torch.sqrt(v_st.clamp_min(1e-12))
            if isinstance(v_st, torch.Tensor)
            else math.sqrt(max(v_st, 1e-12))
        )
        z = randn_like_with_gen(x_s, gen, memory_format)

        return mean + std * z

    def conditional_score(
        self,
        x_t: torch.Tensor,
        x_s: torch.Tensor,
        s: TimeLike,
        t: TimeLike,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """Computes grad_{x_t} log R_{t|s}(x_t | x_s) = (mu_{st}(x_s) - x_t) / V_{st}."""
        mean = self.mu(x_s, s, t)
        v_st = self._bcast_val(self.V(s, t), x_t)
        v_st = v_st.clamp_min(eps) if isinstance(v_st, torch.Tensor) else max(v_st, eps)

        return (mean - x_t) / v_st

    def sample_joint_01(
        self,
        batch_size: int,
        shape: Tuple[int, ...],
        device: torch.device,
        gen: torch.Generator,
        memory_format: torch.memory_format = torch.contiguous_format,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples joint endpoint pairs (x_0, x_1) ~ R_{01} = R_0 * R_{1|0} (for SCONE)."""
        x0 = self.sample_prior(batch_size, shape, device, gen, memory_format)
        x1 = self.sample_conditional(
            x0, s=0.0, t=1.0, gen=gen, memory_format=memory_format
        )

        return x0, x1

    def joint_scores_01(
        self, x0: torch.Tensor, x1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes (grad_{x0} log R_{01}(x0, x1), grad_{x1} log R_{01}(x0, x1)) for SCONE:
            grad_{x0} log R_{01}(x0, x1) = (A_{01} / V_{01})(x1 - mu_{01}(x0)) + grad_{x0} log R_0(x0)
            grad_{x1} log R_{01}(x0, x1) = -(1 / V_{01})(x1 - mu_{01}(x0))
        """
        a01 = self._bcast_val(self.A(0.0, 1.0), x0)
        v01 = self._bcast_val(self.V(0.0, 1.0), x0)
        v01 = v01.clamp_min(1e-12) if isinstance(v01, torch.Tensor) else max(v01, 1e-12)

        residual = x1 - self.mu(x0, 0.0, 1.0)
        r0_mean = self._bcast_val(self.r0_mean, x0)
        r0_var = self._bcast_val(self.r0_std, x0) ** 2

        grad_r0 = -(x0 - r0_mean) / r0_var
        score_x0 = (a01 / v01) * residual + grad_r0
        score_x1 = -residual / v01

        return score_x0, score_x1

    # ------------------------------------------------------------------
    # Conditional Bridge: R_{t|0,1}(. | x0, x1)
    # ------------------------------------------------------------------
    def bridge_params(self, t: TimeLike) -> Tuple[ScalarOrTensor, ScalarOrTensor]:
        """
        Returns (c1, V_t) where:
            mu_t(x0, x1) = mu_{0t}(x0) + c1 * (x1 - mu_{01}(x0))
            V_t          = (V_{0t} * V_{t1}) / V_{01}
        """
        v01 = self.V(0.0, 1.0)
        v0t = self.V(0.0, t)
        vt1 = self.V(t, 1.0)
        a0t = self.A(0.0, t)
        at1 = self.A(t, 1.0)

        c1 = (at1 * v0t) / v01
        v_t = (v0t * vt1) / v01

        return c1, v_t

    def bridge_mean(
        self, x0: torch.Tensor, x1: torch.Tensor, t: TimeLike
    ) -> torch.Tensor:
        """
        Computes exact bridge mean for general affine linear SDEs:
            mu_t(x0, x1) = mu_{0t}(x0) + (A_{t1} V_{0t} / V_{01}) * (x1 - mu_{01}(x0)).
        """
        c1, _ = self.bridge_params(t)
        c1_b = self._bcast_val(c1, x0)

        return self.mu(x0, 0.0, t) + c1_b * (x1 - self.mu(x0, 0.0, 1.0))

    def sample_bridge(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: TimeLike,
        gen: torch.Generator,
        memory_format: torch.memory_format = torch.contiguous_format,
        return_noise: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Reparameterized bridge sample:
            x_t = mu_t(x0, x1) + sqrt(V_t) * eps,   eps ~ N(0, I_d).
        Set `return_noise=True` for DSBM/IMF and SF2M targets.
        """
        c1, v_t = self.bridge_params(t)
        c1_b = self._bcast_val(c1, x0)
        v_t_b = self._bcast_val(v_t, x0)

        mean = self.bridge_mean(x0, x1, t)
        std = (
            torch.sqrt(v_t_b.clamp_min(1e-12))
            if isinstance(v_t_b, torch.Tensor)
            else math.sqrt(max(v_t_b, 1e-12))
        )

        eps = randn_like_with_gen(x0, gen, memory_format)
        x_t = mean + std * eps
        return (x_t, eps) if return_noise else x_t


class BrownianReference(ReferenceProcess):
    """
    Scaled Brownian Motion (R_sigma):
        dX_t = sigma dW_t
    """

    name = "brownian"

    def __init__(self, sigma: float = 1.0, sigma_0: Optional[float] = None):
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")

        self._sigma = float(sigma)

        # Defaults to `sigma` for standard Brownian prior N(0, sigma^2 I);
        # can be set to a broader sigma_0 > 0 for full-dual SCONE.
        self._sigma_0 = float(sigma_0 if sigma_0 is not None else sigma)
        if self._sigma_0 <= 0:
            raise ValueError(f"sigma_0 must be positive, got {self._sigma_0}")

    def alpha(self, t: TimeLike = 0.0) -> float:
        return 0.0

    def beta(self, t: TimeLike = 0.0) -> float:
        return 0.0

    def sigma(self, t: TimeLike = 0.0) -> float:
        return self._sigma

    def sigma_sq(self, t: TimeLike = 0.0) -> float:
        return self._sigma**2

    def A(self, s: TimeLike, t: TimeLike) -> ScalarOrTensor:
        dt = t - s
        return torch.ones_like(dt) if isinstance(dt, torch.Tensor) else 1.0

    def B(self, s: TimeLike, t: TimeLike) -> ScalarOrTensor:
        dt = t - s
        return torch.zeros_like(dt) if isinstance(dt, torch.Tensor) else 0.0

    def V(self, s: TimeLike, t: TimeLike) -> ScalarOrTensor:
        return (self._sigma**2) * (t - s)

    @property
    def r0_mean(self) -> float:
        return 0.0

    @property
    def r0_std(self) -> float:
        return self._sigma_0


class OUReference(ReferenceProcess):
    """
    Variance-Preserving Ornstein-Uhlenbeck Process (R_gamma):
        dX_t = -gamma X_t dt + sqrt(2*gamma) dW_t
    Admits N(0, I_d) as its unique stationary marginal.
    """

    name = "ou"

    def __init__(self, gamma: float = 1.0):
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}")
        self.gamma = float(gamma)

    def alpha(self, t: TimeLike = 0.0) -> float:
        return -self.gamma

    def beta(self, t: TimeLike = 0.0) -> float:
        return 0.0

    def sigma(self, t: TimeLike = 0.0) -> float:
        return math.sqrt(2.0 * self.gamma)

    def sigma_sq(self, t: TimeLike = 0.0) -> float:
        return 2.0 * self.gamma

    def A(self, s: TimeLike, t: TimeLike) -> ScalarOrTensor:
        dt = t - s
        if isinstance(dt, torch.Tensor):
            return torch.exp(-self.gamma * dt)
        return math.exp(-self.gamma * float(dt))

    def B(self, s: TimeLike, t: TimeLike) -> ScalarOrTensor:
        dt = t - s
        return torch.zeros_like(dt) if isinstance(dt, torch.Tensor) else 0.0

    def V(self, s: TimeLike, t: TimeLike) -> ScalarOrTensor:
        dt = t - s
        if isinstance(dt, torch.Tensor):
            return -torch.expm1(-2.0 * self.gamma * dt)
        return -math.expm1(-2.0 * self.gamma * float(dt))

    @property
    def r0_mean(self) -> float:
        return 0.0

    @property
    def r0_std(self) -> float:
        return 1.0
