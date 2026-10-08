import torch
import torch.nn as nn


class SF2MInferenceWrapper(nn.Module):
    """
    Adapter exposing the SDE drift of the learned (u, s) pair to the pipeline.
    t=0 (Prior) -> t=1 (Data).
    """

    def __init__(
        self,
        u_model: nn.Module,
        s_model: nn.Module,
        sigma: float,
        direction: str = "b",
        eps: float = 1e-4,
    ):
        super().__init__()
        self.u = u_model
        self.s = s_model
        self.sigma = sigma
        self.direction = direction
        self.eps = eps
        self.model_type = "sf2m"

    def forward(self, x, t):
        # direction="b" (Prior -> Data), t passed is 0 -> 1.
        # direction="f" (Data -> Prior), t passed is 0 -> 1 by the sampler,
        # but we must treat it internally as 1 -> 0 for the networks.
        t_real = 1.0 - t if self.direction == "f" else t
        t_safe = torch.as_tensor(t_real, device=x.device, dtype=x.dtype).reshape(-1)
        if t_safe.numel() == 1:
            t_safe = t_safe.expand(x.shape[0])
        t_safe = t_safe.reshape(-1, 1).clamp(self.eps, 1.0 - self.eps)

        v_hat = self.u(x, t_safe)
        eps_hat = self.s(x, t_safe)

        t_b = t_safe.view(-1, *([1] * (x.dim() - 1)))

        if self.direction == "b":
            # Forward SDE drift: u_t^o + 0.5 * sigma^2 * s_t
            drift = v_hat - self.sigma * torch.sqrt(t_b / (1.0 - t_b)) * eps_hat
        else:
            # Reverse SDE drift (Anderson): -u_t^o + 0.5 * sigma^2 * s_t
            drift = -v_hat - self.sigma * torch.sqrt((1.0 - t_b) / t_b) * eps_hat

        return drift
