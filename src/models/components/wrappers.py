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


class SCONEWrapper(torch.nn.Module):
    """Packs the dual potentials together so the Evaluator can treat them as a single model."""

    def __init__(
        self,
        phi_model: torch.nn.Module,
        psi_model: torch.nn.Module,
        sigma: float,
        reference: str,
    ):
        super().__init__()
        self.model_type = "scone"
        self.phi = phi_model
        self.psi = psi_model
        self.sigma = sigma
        self.reference = reference


class ENOTWrapper(nn.Module):
    """Packs the drift and potential models together."""

    def __init__(
        self, drift_model: nn.Module, potential_model: nn.Module, sigma: float
    ):
        super().__init__()
        self.drift = drift_model
        self.potential = potential_model
        self.sigma = sigma
        self.model_type = "enot"

    def forward(self, x, t):
        # Evaluator and standard solvers only need the drift
        return self.drift(x, t)


class IterWrapper(nn.Module):
    """
    Adapter packing dual forward/backward policies into a single entity.
    Compatible with IPF, IMF, and SB-FBSDE pipelines.
    """

    def __init__(
        self,
        f_model: nn.Module,
        b_model: nn.Module,
        sigma: float,
        direction: str = "b",
        model_type: str = "fbsde",
    ):
        super().__init__()
        self.f = f_model
        self.b = b_model
        self.sigma = sigma
        self.direction = direction
        self.model_type = model_type

    def forward(self, x, t):
        # The Evaluator's generic validation loss simply calls `model(x, t)`.
        # We proxy to the active policy based on the evaluation direction.
        return self.b(x, t) if self.direction == "b" else self.f(x, t)
