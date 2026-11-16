import torch

from scipy.optimize import linear_sum_assignment

import math
from typing import Tuple, Optional, Union


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
    def exact_ot_assignment_cpu(
        cost_matrix: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Exact Minibatch Optimal Transport using Scipy's Hungarian algorithm on CPU."""
        # Offload cost matrix to CPU for the exact solver
        cost_np = cost_matrix.detach().cpu().numpy()

        # Solve exact bipartite matching
        row_ind, col_ind = linear_sum_assignment(cost_np)

        # Push indices back to the original device
        device = cost_matrix.device
        return (
            torch.tensor(row_ind, dtype=torch.long, device=device),
            torch.tensor(col_ind, dtype=torch.long, device=device),
        )

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
    # DSB Specific Math
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
    # SB-FBSDE Specific Math
    # =========================================================================

    @staticmethod
    def compute_div_gz(
        z: torch.Tensor,
        x: torch.Tensor,
        g_t: Union[float, torch.Tensor],
        gen: Optional[torch.Generator] = None,
        exact: bool = True,
    ) -> torch.Tensor:
        """
        Computes the Hutchinson trace estimator for the divergence of g(t)Z(t, x).
        Requires x.requires_grad = True before passing through the network.
        """
        if exact:
            div = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
            for i in range(z.shape[1]):
                # Differentiate the i-th output dimension with respect to all inputs
                grad_z = torch.autograd.grad(
                    outputs=z[:, i].sum(),
                    inputs=x,
                    create_graph=True,
                    retain_graph=True,
                )[0]
                # Accumulate the diagonal terms (d z_i / d x_i)
                div += grad_z[:, i]

            return div * g_t

        # Ensure 'e' is instantiated in the exact same precision as 'x' (FP32)
        e = (
            torch.randint(0, 2, x.shape, device=x.device, generator=gen).to(x.dtype)
            * 2.0
            - 1.0
        )

        gz = z * g_t

        e_dzdx = torch.autograd.grad(
            outputs=gz, inputs=x, grad_outputs=e, create_graph=True, retain_graph=True
        )[0]

        div_gz = (e_dzdx * e).view(x.shape[0], -1).sum(dim=1)

        return div_gz

    @staticmethod
    def compute_sbfbsde_loss(z_opt, z_impt, div_gz, h, w=None):
        inner = (z_opt * (0.5 * z_opt + z_impt)).view(z_opt.shape[0], -1).sum(1)
        per = (inner + div_gz) * h
        if w is not None:
            per = per * w.view(-1)
        return per.mean()

    # =========================================================================
    # SCONE Specific Math
    # =========================================================================

    @staticmethod
    def get_reference_scores(
        x: torch.Tensor,
        y: torch.Tensor,
        sigma: float = 1.0,
        reference: str = "brownian",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves analytical joint scores \nabla_x \log R_{01} and \nabla_y \log R_{01}.
        """
        if reference == "brownian":
            score_x = (y - x) / (sigma**2)
            score_y = (x - y) / (sigma**2)
            return score_x, score_y

        raise NotImplementedError(
            f"Analytical scores for reference '{reference}' are not supported. "
            "Requires a pre-trained Denoising Score Matching network (s_omega)."
        )

    # =========================================================================
    # Iterative Action Matching (IAM) Specific Math
    # =========================================================================
    @staticmethod
    def get_iam_reference_params(
        x: torch.Tensor,
        sigma: float = 1.0,
        gamma: float = 1.0,
        reference: str = "brownian",
    ) -> Tuple[torch.Tensor, float]:
        """
        Returns the forward reference drift f_t(x) and squared diffusion sigma_t^2
        for R_sigma ('brownian') or R_gamma ('ou').
        """
        if reference == "brownian":
            return torch.zeros_like(x), float(sigma**2)
        elif reference == "ou":
            return -gamma * x, float(2.0 * gamma)
        raise ValueError(f"Unsupported reference process for IAM: {reference!r}")

    @classmethod
    def sample_iam_bridge(
        cls,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        gen: torch.Generator,
        sigma: float = 1.0,
        gamma: float = 1.0,
        reference: str = "brownian",
        memory_format=torch.contiguous_format,
    ) -> torch.Tensor:
        """
        Samples x_t ~ q_{t|01}(. | x0, x1) = N(mu_t(x0, x1), Sigma_t)
        in closed form for either R_sigma or R_gamma.
        """
        B = x0.shape[0]
        t_exp = t.view(B, *([1] * (x0.ndim - 1)))
        z = cls._randn_like(x0, gen, memory_format)

        if reference == "brownian":
            mu_t = (1.0 - t_exp) * x0 + t_exp * x1
            std_t = sigma * torch.sqrt(t_exp * (1.0 - t_exp))
            return mu_t + std_t * z

        elif reference == "ou":
            # Use -expm1(-u) = 1 - exp(-u) for numerical stability near t=0 and t=1
            denom = -math.expm1(-2.0 * gamma)
            num_0 = -torch.expm1(-2.0 * gamma * (1.0 - t_exp))
            num_1 = -torch.expm1(-2.0 * gamma * t_exp)

            coef_0 = (num_0 / denom) * torch.exp(-gamma * t_exp)
            coef_1 = (num_1 / denom) * torch.exp(-gamma * (1.0 - t_exp))
            mu_t = coef_0 * x0 + coef_1 * x1

            var_t = (num_0 * num_1) / denom
            std_t = torch.sqrt(var_t.clamp_min(1e-12))
            return mu_t + std_t * z

        raise ValueError(f"Unsupported reference process for IAM: {reference!r}")

    @staticmethod
    def compute_action_laplacian(
        grad_x: torch.Tensor,
        x: torch.Tensor,
        exact: bool = True,
        gen: Optional[torch.Generator] = None,
        create_graph: bool = True,
    ) -> torch.Tensor:
        """
        Computes the spatial Laplacian Delta_x s_t(x) = Tr(nabla_x^2 s_t(x))
        either exactly (O(d) backward passes) or via Hutchinson's trace estimator (O(1)).
        """
        B = x.shape[0]
        flat_grad = grad_x.reshape(B, -1)
        d = flat_grad.shape[1]

        if exact:
            lap = torch.zeros(B, device=x.device, dtype=x.dtype)
            for i in range(d):
                grad2_i = torch.autograd.grad(
                    outputs=flat_grad[:, i].sum(),
                    inputs=x,
                    create_graph=create_graph,
                    retain_graph=True,
                )[0]
                lap = lap + grad2_i.reshape(B, -1)[:, i]
            return lap

        # Hutchinson's Rademacher trace estimator: E[z^T nabla_x (grad_x^T z)]
        z = (
            torch.randint(0, 2, x.shape, device=x.device, generator=gen).to(x.dtype)
            * 2.0
            - 1.0
        )
        hvp = torch.autograd.grad(
            outputs=grad_x,
            inputs=x,
            grad_outputs=z,
            create_graph=create_graph,
            retain_graph=True,
        )[0]
        return (hvp * z).reshape(B, -1).sum(dim=1)

    @classmethod
    def compute_iam_loss(
        cls,
        model: torch.nn.Module,
        x0: torch.Tensor,
        x1: torch.Tensor,
        direction: str = "b",
        sigma: float = 1.0,
        gamma: float = 1.0,
        reference: str = "brownian",
        exact_laplacian: bool = True,
        gen: Optional[torch.Generator] = None,
        create_graph: bool = True,
    ) -> torch.Tensor:
        """
        Evaluates the empirical Conditional Entropic Action Matching loss:
          L_SB^F (direction='f'): + 0.5 * sigma_t^2 * Delta_x s_t
          L_SB^B (direction='b'): - 0.5 * sigma_t^2 * Delta_x s_t
        """
        B = x0.shape[0]
        device, dtype = x0.device, x0.dtype

        # 1. Stratified sin^2 importance sampling in time and bridge state x_t
        u = (
            torch.arange(B, device=device, dtype=dtype).unsqueeze(1)
            + torch.rand(B, 1, device=device, generator=gen, dtype=dtype)
        ) / B
        u = 1e-3 + (1.0 - 2e-3) * u
        t = torch.sin(0.5 * math.pi * u).pow(2)
        dt_du = (0.5 * math.pi * torch.sin(math.pi * u)).view(B)

        x_t = cls.sample_iam_bridge(
            x0=x0,
            x1=x1,
            t=t,
            gen=gen,
            sigma=sigma,
            gamma=gamma,
            reference=reference,
            memory_format=torch.contiguous_format,
        )

        # 2. Gauge-fixed boundary term: [s(x0, 0) - s(mu0, 0)] - [s(x1, 1) - s(mu1, 1)]
        mu0 = x0.detach().mean(dim=0, keepdim=True)
        mu1 = x1.detach().mean(dim=0, keepdim=True)

        x_bnd = torch.cat([x0, x1, mu0, mu1], dim=0)
        t_bnd = torch.cat(
            [
                torch.zeros(B, 1, device=device, dtype=dtype),
                torch.ones(B, 1, device=device, dtype=dtype),
                torch.zeros(1, 1, device=device, dtype=dtype),
                torch.ones(1, 1, device=device, dtype=dtype),
            ],
            dim=0,
        )
        s_bnd = model(x_bnd, t_bnd).view(-1)
        boundary_term = (s_bnd[:B] - s_bnd[-2]) - (s_bnd[B : 2 * B] - s_bnd[-1])

        # 3. First-order gradients and spatial Laplacian of s_bar(x_t, t)
        x_t = x_t.detach().requires_grad_(True)
        t_in = t.detach().requires_grad_(True)

        t_bcast = t_in.view(B, *([1] * (x_t.ndim - 1)))
        x_bar_t = (1.0 - t_bcast) * mu0 + t_bcast * mu1

        s_pair = model(
            torch.cat([x_t, x_bar_t], dim=0),
            torch.cat([t_in, t_in], dim=0),
        ).view(2 * B)
        s_t = s_pair[:B] - s_pair[B:]

        # Keep create_graph=True here so lap_x can compute second spatial derivatives
        grad_x, grad_t = torch.autograd.grad(
            outputs=s_t.sum(),
            inputs=(x_t, t_in),
            create_graph=True,
        )
        lap_x = cls.compute_action_laplacian(
            grad_x=grad_x,
            x=x_t,
            exact=exact_laplacian,
            gen=gen,
            create_graph=create_graph,
        )

        # 4. Assemble integrand
        f_t, sigma_t_sq = cls.get_iam_reference_params(
            x_t, sigma=sigma, gamma=gamma, reference=reference
        )
        grad_flat = grad_x.reshape(B, -1)
        f_flat = f_t.reshape(B, -1)

        kinetic_and_drift = (grad_flat * (f_flat + 0.5 * sigma_t_sq * grad_flat)).sum(
            dim=1
        )
        lap_sign = 1.0 if direction == "f" else -1.0
        integrand = (
            kinetic_and_drift + grad_t.view(B) + lap_sign * 0.5 * sigma_t_sq * lap_x
        )

        return (boundary_term + dt_du * integrand).mean()

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
            # row_ind, col_ind = cls.greedy_assignment_gpu(cost_matrix)
            row_ind, col_ind = cls.exact_ot_assignment_cpu(cost_matrix)

            # Permute the batches to align them according to the optimal transport plan
            x_0 = x_0[row_ind]
            z_batch = z_batch[col_ind]

            x_t = t_expand * z_batch + (1.0 - t_expand) * x_0
            target_u = (z_batch - x_t) / (1.0 - t_expand)

        elif model_type == "sde":
            # Independent endpoints relaxation: Prior is standard Gaussian
            eps = cls._randn_like(z_batch, gen, memory_format)

            x_t = t_expand * z_batch + sigma * torch.sqrt(1.0 - t_expand) * eps
            target_u = (z_batch - x_t) / (1.0 - t_expand)

        elif model_type == "flow_m":
            # Prior is standard Gaussian
            x_0 = cls._randn_like(z_batch, gen, memory_format)

            # cost_matrix = torch.cdist(x_0.view(B, -1), z_batch.view(B, -1), p=2).pow(2)
            # row_ind, col_ind = cls.exact_ot_assignment_cpu(cost_matrix)
            # x_0 = x_0[row_ind]
            # z_batch = z_batch[col_ind]

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
                # t_net = t_expand
                t_net = t
            else:
                target_u = (
                    -(z_batch - x0_batch)
                    - sigma * torch.sqrt((1.0 - t_expand) / t_expand) * z
                )
                # t_net = (
                #     1.0 - t_expand
                # )  # The backward network processes the state at 1 - t
                t_net = 1.0 - t

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

        elif model_type == "lightsbm":
            # 1. Sample Prior
            x_0 = cls._randn_like(z_batch, gen, memory_format) * sigma

            # 2. Minibatch OT assignment
            cost_matrix = torch.cdist(x_0.view(B, -1), z_batch.view(B, -1), p=2).pow(2)
            # row_ind, col_ind = cls.greedy_assignment_gpu(cost_matrix)
            row_ind, col_ind = cls.exact_ot_assignment_cpu(cost_matrix)
            x_0 = x_0[row_ind]
            z_batch = z_batch[col_ind]

            # 3. Brownian Bridge Interpolation
            noise = cls._randn_like(z_batch, gen, memory_format)

            # x_t = t * x_1 + (1-t) * x_0 + \sqrt{\epsilon t (1-t)} * z
            x_t = (
                t_expand * z_batch
                + (1.0 - t_expand) * x_0
                + sigma * torch.sqrt(t_expand * (1.0 - t_expand)) * noise
            )

            # 4. Target for matching the drift
            target_u = (z_batch - x_t) / (1.0 - t_expand)

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # return x_t, target_u, t_expand
        return x_t, target_u, t

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

        if model_type in ["sde", "minibatch", "lightsbm"]:
            return torch.mean((1.0 - t_net) * raw_loss)

        elif model_type == "ipf":
            if h is None:
                raise ValueError("IPF loss requires step size 'h' to be passed.")
            return torch.mean(raw_loss) * h

        elif model_type == "imf":
            # The mathematical weight for the Brownian bridge simplifies perfectly
            # to the exact same equation for BOTH directions when mapped to t_net!
            t_net = t_net.unsqueeze(-1)
            weight = 1.0 / (1.0 + (sigma**2 * t_net) / (1.0 - t_net))
            return torch.mean(weight * raw_loss)

        return torch.mean(raw_loss)
