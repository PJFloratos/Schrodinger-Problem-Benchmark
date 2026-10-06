import torch
from typing import Tuple


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

    @classmethod
    def get_interpolant_and_target(
        cls,
        model_type: str,
        z_batch: torch.Tensor,
        t: torch.Tensor,
        gen: torch.Generator,
        memory_format=torch.contiguous_format,
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

        elif model_type == "sf2m":
            # x_0 = Noise, z_batch = Data. Generation is t=0 (Noise) -> t=1 (Data)
            x_0 = cls._randn_like(z_batch, gen, memory_format)
            eps = cls._randn_like(z_batch, gen, memory_format)
            sigma = 1.0

            t_safe = t_expand.clamp(1e-4, 1.0 - 1e-4)
            sigma_t = sigma * torch.sqrt(t_safe * (1.0 - t_safe))

            # Reparameterized conditional sample
            x_t = t_safe * z_batch + (1.0 - t_safe) * x_0 + sigma_t * eps

            # Evaluator tests the combined forward SDE drift (noise -> data): u_t^o + 0.5 * sigma^2 * s_t
            # With the new parameterization: v - sigma * sqrt(t/(1-t)) * eps
            target_u = (z_batch - x_0) - sigma * torch.sqrt(
                t_safe / (1.0 - t_safe)
            ) * eps

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        return x_t, target_u, t_expand

    @staticmethod
    def compute_loss(
        model_type: str,
        pred_u: torch.Tensor,
        target_u: torch.Tensor,
        t_expand: torch.Tensor,
    ) -> torch.Tensor:
        """Computes matching objective (time-weighted for SDE/OT, standard MSE for Flow Matching)."""
        if model_type in ["sde", "minibatch"]:
            return torch.mean((1.0 - t_expand) * (pred_u - target_u) ** 2)

        return torch.mean((pred_u - target_u) ** 2)
