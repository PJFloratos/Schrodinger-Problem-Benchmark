import torch
import torch.nn as nn
import math


class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal time embedding for diffusion/flow models."""

    def __init__(self, dim: int, scale: float = 1000.0, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.scale = scale
        self.max_period = max_period

    def forward(self, t):
        device = t.device

        t = t * self.scale

        half_dim = self.dim // 2
        embeddings = math.log(self.max_period) / max(half_dim - 1, 1)
        embeddings = torch.exp(
            torch.arange(half_dim, device=device, dtype=t.dtype) * -embeddings
        )
        embeddings = t * embeddings[None, :]

        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
