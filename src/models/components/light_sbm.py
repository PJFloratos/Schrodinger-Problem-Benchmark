import torch
import torch.nn as nn
import math


class LightSBM(nn.Module):
    def __init__(
        self,
        dim: int,
        n_potentials: int = 50,
        epsilon: float = 1,
        model_type: str = "lightsbm",
        S_diagonal_init: float = 0.1,
    ):
        super().__init__()
        self.model_type = model_type
        self.sigma = math.sqrt(epsilon)
        self.dim = dim
        self.n_potentials = n_potentials
        self.register_buffer("epsilon", torch.tensor(epsilon, dtype=torch.float32))

        # 1. Weights (alpha): parameterized in log-space
        self.log_alpha = nn.Parameter(
            torch.log(torch.ones(n_potentials) / n_potentials)
        )

        # 2. Means (r_k): standard initialization
        self.r = nn.Parameter(torch.randn(n_potentials, dim))

        # 3. Covariances (S_k): initialized diagonally in log-space for stability
        self.S_log_diagonal_matrix = nn.Parameter(
            torch.log(S_diagonal_init * torch.ones(n_potentials, self.dim))
        )

    def init_r_by_samples(self, samples: torch.Tensor):
        """Optionally initialize means using actual data samples."""
        assert samples.shape[0] == self.r.shape[0]
        self.r.data = samples.clone().to(self.r.device)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Computes the analytical drift g_theta(t, x) exactly.
        """
        # Ensure consistent formatting
        t = t.reshape(-1, 1).float()
        t_1d = t.view(-1)  # Shape: (B,)
        eps = self.epsilon

        # 1. Fetch parameters
        S_diag = torch.exp(self.S_log_diagonal_matrix)  # (K, D)
        r = self.r  # (K, D)
        log_alpha = self.log_alpha  # (K,)

        S_log_det = torch.sum(self.S_log_diagonal_matrix, dim=-1)  # (K,)

        # 2. Compute matrices: A_k^t = t / (eps * (1-t)) I + S_k^{-1} / eps
        # Shape: (B, K, D)
        A_diag = (t_1d / (eps * (1 - t_1d)))[:, None, None] + (1 / (eps * S_diag))[
            None, :, :
        ]
        A_log_det = torch.sum(torch.log(A_diag), dim=-1)  # (B, K)

        S_inv = 1 / S_diag  # (K, D)
        A_inv = 1 / A_diag  # (B, K, D)

        # 3. Compute mean shift: c = x / (eps * (1-t)) + r / (eps * S_k)
        # Shape: (B, K, D)
        c = ((1 / (eps * (1 - t_1d)))[:, None, None] * x[:, None, :]) + (
            r / (eps * S_diag)
        )[None, :, :]

        # 4. Assemble the exponent components
        r_S_inv_r = ((r * S_inv * r) / eps).sum(dim=-1)  # (K,)
        c_A_inv_c = (c * A_inv * c).sum(dim=-1)  # (B, K)

        # compute w_k
        exp_arg = (
            log_alpha[None, :]
            - 0.5 * S_log_det[None, :]
            - 0.5 * A_log_det
            - 0.5 * r_S_inv_r[None, :]
            + 0.5 * c_A_inv_c
        )  # (B, K)

        # Compute Softmax weights: \Omega_k(t, x)
        omega = torch.softmax(exp_arg, dim=-1)  # (B, K)

        # Compute the weighted sum: \sum_k \Omega_k(t, x) * (A_k^t)^{-1} h_k(t, x)
        # Note: A_inv * c has shape (B, K, D), omega has shape (B, K).
        # We unsqueeze omega to (B, K, 1) to broadcast over D.
        grad_term = torch.sum(omega.unsqueeze(-1) * (A_inv * c), dim=1)  # (B, D)

        # Assemble full exact drift
        drift = (-x + grad_term) / (1 - t)

        return drift

    @torch.no_grad()
    def sample_x1(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Samples the exact terminal point x_1 ~ \pi_\theta(x_1 | x0).
        Bypasses SDE integration completely.
        """
        eps = self.epsilon
        S_diag = torch.exp(self.S_log_diagonal_matrix)  # (K, D)
        r = self.r  # (K, D)
        log_alpha = self.log_alpha  # (K,)

        # 1. Compute log conditional weights: log \tilde{\alpha}_k(x_0)
        # Formula: (x_0^T S_k x_0 + 2 r_k^T x_0) / 2\epsilon + \log \alpha_k
        x_S_x = (x0[:, None, :] * S_diag[None, :, :] * x0[:, None, :]).sum(
            dim=-1
        )  # (B, K)
        x_r = (x0[:, None, :] * r[None, :, :]).sum(dim=-1)  # (B, K)

        log_tilde_alpha = (x_S_x + 2 * x_r) / (2 * eps) + log_alpha[None, :]  # (B, K)

        # 2. Normalize weights into probabilities
        probs = torch.softmax(log_tilde_alpha, dim=-1)  # (B, K)

        # 3. Sample a mixture component for each item in the batch
        mix_idx = torch.multinomial(probs, 1).squeeze(-1)  # (B,)

        # 4. Gather the parameters for the chosen components
        S_chosen = S_diag[mix_idx]  # (B, D)
        r_chosen = r[mix_idx]  # (B, D)

        # 5. Compute the conditional mean: r_k(x_0) = r_k + S_k x_0
        mean_chosen = r_chosen + S_chosen * x0  # (B, D)

        # 6. Sample the final Gaussian: x_1 ~ N(mean, eps * S_k)
        noise = torch.randn_like(x0)
        x_1 = mean_chosen + torch.sqrt(eps * S_chosen) * noise

        return x_1
