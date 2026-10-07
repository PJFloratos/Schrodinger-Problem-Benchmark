import torch

import math
from typing import Tuple, Optional


class ConditionalVectorField:
    """Encapsulates interpolation dynamics and target vector fields."""

    @staticmethod
    def _randn_like(
        ref: torch.Tensor, gen: torch.Generator, memory_format=torch.contiguous_format
    ) -> torch.Tensor:
        # torch.randn_like has no `generator` argument, so draw explicitly and put the
        # result in the memory format randn_like would have preserved.
        return torch.randn(
            ref.shape, device=ref.device, dtype=ref.dtype, generator=gen
        ).contiguous(memory_format=memory_format)

    @staticmethod
    def greedy_assignment_gpu(
        cost_matrix: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pure-GPU greedy approximation of Minibatch Optimal Transport."""
        B = cost_matrix.shape[0]
        row_ind = torch.arange(B, device=cost_matrix.device)
        col_ind = torch.zeros(B, dtype=torch.long, device=cost_matrix.device)

        flat_cost = cost_matrix.clone().flatten()
        for _ in range(B):
            min_idx = torch.argmin(flat_cost)
            r, c = min_idx // B, min_idx % B
            col_ind[r] = c

            flat_cost[r * B : (r + 1) * B] = float("inf")
            flat_cost[c::B] = float("inf")

        return row_ind, col_ind

    # =========================================================================
    # IPF / DSB Specific Math
    # =========================================================================

    @staticmethod
    def get_dsb_target(
        drift_next: torch.Tensor, z: torch.Tensor, h: float, sigma: float = 1.0
    ) -> torch.Tensor:
        """
        Computes the Diffusion Schrodinger Bridge regression target
        for the opposite network in IPF.
        """
        return -drift_next - (sigma * z) / math.sqrt(h)

    # =========================================================================
    # SF2M Specific Math
    # =========================================================================

    @staticmethod
    def compute_sf2m_loss(
        v_pred: torch.Tensor,
        eps_pred: torch.Tensor,
        v_target: torch.Tensor,
        eps_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Separate fp32 MSEs for the flow and noise heads (each has its own optimizer/scaler)."""
        loss_v = torch.mean((v_pred.float() - v_target) ** 2)
        loss_eps = torch.mean((eps_pred.float() - eps_target) ** 2)

        return loss_v, loss_eps

    # =========================================================================
    # Closed-Form Interpolants
    # =========================================================================

    @classmethod
    def get_interpolant_and_target(
        cls,
        model_type: str,
        z_batch: torch.Tensor,
        t: torch.Tensor,
        gen: torch.Generator,
        memory_format=torch.contiguous_format,
        sigma: Optional[int] = None,
        direction: Optional[str] = None,
        x0_batch: Optional[torch.Tensor] = None,
        eps: Optional[float] = 1e-4,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes (x_t, target_u, t_expand) given endpoint batch z_batch (data) and time t.
        """
        B = z_batch.shape[0]
        t_expand = t.view(B, *([1] * (z_batch.ndim - 1)))

        if model_type == "minibatch":
            # Sample from the standard Gaussian prior
            x_0 = cls._randn_like(z_batch, gen, memory_format)

            # Minibatch OT using the pure GPU implementation
            cost_matrix = torch.cdist(x_0.view(B, -1), z_batch.view(B, -1), p=2).pow(2)
            row_ind, col_ind = cls.greedy_assignment_gpu(cost_matrix)

            # Permute the batches to align them according to the optimal transport plan
            x_0 = x_0[row_ind]
            z_batch = z_batch[col_ind]

            x_t = t_expand * z_batch + (1.0 - t_expand) * x_0
            target_u = (z_batch - x_t) / (1.0 - t_expand)

        elif model_type == "sde":
            # Independent endpoints relaxation: Prior is standard Gaussian
            eps = cls._randn_like(z_batch, gen, memory_format)

            x_t = t_expand * z_batch + torch.sqrt(1.0 - t_expand) * eps
            target_u = (z_batch - x_t) / (1.0 - t_expand)

        elif model_type == "flow_m":
            # Prior is standard Gaussian
            x_0 = cls._randn_like(z_batch, gen, memory_format)

            x_t = t_expand * z_batch + (1.0 - t_expand) * x_0
            target_u = z_batch - x_0

        elif model_type == "imf":
            # For IMF, z_batch acts as x1 (target endpoint). x0_batch is the source.
            if x0_batch is None:
                raise ValueError("IMF model requires x0_batch explicitly passed.")

            z = cls._randn_like(z_batch, gen, memory_format)
            x_t = (
                (1.0 - t_expand) * x0_batch
                + t_expand * z_batch
                + sigma * torch.sqrt(t_expand * (1.0 - t_expand)) * z
            )

            if direction == "f":
                target_u = (z_batch - x0_batch) - sigma * torch.sqrt(
                    t_expand / (1.0 - t_expand)
                ) * z
                t_net = t_expand
            else:
                target_u = (
                    -(z_batch - x0_batch)
                    - sigma * torch.sqrt((1.0 - t_expand) / t_expand) * z
                )
                t_net = (
                    1.0 - t_expand
                )  # The backward network processes the state at 1 - t

            return x_t, target_u, t_net

        elif model_type == "sf2m":
            # For SF2M, z_batch = Data, x0_batch = Noise.
            if x0_batch is None:
                raise ValueError("SF2M model requires x0_batch explicitly passed.")
            if sigma is None:
                sigma = 1.0

            eps_noise = cls._randn_like(z_batch, gen, memory_format)

            t_safe = t_expand.clamp(1e-4, 1.0 - 1e-4)
            sigma_t = sigma * torch.sqrt(t_safe * (1.0 - t_safe))

            # Reparameterized conditional sample
            x_t = t_safe * z_batch + (1.0 - t_safe) * x0_batch + sigma_t * eps_noise

            # SF2M models the velocity and the noise components separately
            v_target = z_batch - x0_batch
            eps_target = eps_noise

            # Pack targets together
            target_u = (v_target, eps_target)

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        return x_t, target_u, t_expand

    @staticmethod
    def compute_loss(
        model_type: str,
        pred_u: torch.Tensor,
        target_u: torch.Tensor,
        t_net: torch.Tensor,
        h: Optional[float] = None,
        sigma: float = 1.0,
    ) -> torch.Tensor:
        """Computes matching objective (time-weighted for SDE/OT, standard MSE for Flow Matching)."""

        raw_loss = (pred_u - target_u) ** 2

        if model_type in ["sde", "minibatch"]:
            return torch.mean((1.0 - t_net) * raw_loss)

        elif model_type == "ipf":
            if h is None:
                raise ValueError("IPF loss requires step size 'h' to be passed.")
            return torch.mean(raw_loss) * h

        elif model_type == "imf":
            # The mathematical weight for the Brownian bridge simplifies perfectly
            # to the exact same equation for BOTH directions when mapped to t_net!
            weight = 1.0 / (1.0 + (sigma**2 * t_net) / (1.0 - t_net))
            return torch.mean(weight * raw_loss)

        return torch.mean(raw_loss)
