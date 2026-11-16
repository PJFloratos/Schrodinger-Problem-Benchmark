from src.models.components import SinusoidalTimeEmbedding, FourierFeatures

import torch
from torch import nn
from torch.nn.utils import spectral_norm

import math


class VelocityMLP(nn.Module):
    def __init__(
        self,
        d=2,
        hidden=128,
        time_dim=64,
        model_type="sde",
        fourier_scale=0.5,
        use_fourier=True,
    ):
        super().__init__()
        self.model_type = model_type
        self.use_fourier = use_fourier and (model_type == "iam")

        # 1. Time Embedding & Encoder
        if model_type == "iam":
            self.time_embed = SinusoidalTimeEmbedding(
                dim=time_dim, scale=10.0, max_period=10.0
            )
        else:
            self.time_embed = SinusoidalTimeEmbedding(dim=time_dim)

        self.time_encoder = nn.Sequential(
            nn.Linear(time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, time_dim * 2),
            nn.SiLU(),
        )

        # 2. Spatial Encoder
        if self.use_fourier:
            fourier_dim = 32
            self.fourier = FourierFeatures(
                in_features=d,
                mapping_size=fourier_dim,
                scale=fourier_scale,
            )
            x_in_dim = d + fourier_dim * 2
        else:
            self.fourier = None
            x_in_dim = d

        self.x_encoder = nn.Sequential(
            nn.Linear(x_in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, time_dim * 2),
            nn.SiLU(),
        )

        # 3. Decoder Network
        self.net = nn.Sequential(
            nn.Linear(time_dim * 4, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1 if model_type == "iam" else d),
        )

    def forward(self, x, t):
        x_in = torch.cat([x, self.fourier(x)], dim=-1) if self.use_fourier else x
        xemb = self.x_encoder(x_in)

        t_emb = self.time_embed(t)
        temb = self.time_encoder(t_emb)

        h = torch.cat([xemb, temb], dim=-1)
        out = self.net(h)

        return out


class EnergyMLP(nn.Module):
    """Models a time-independent scalar potential: R^d -> R"""

    def __init__(self, d=2, hidden=128, time_dim=64, model_type="ebm"):
        super().__init__()
        self.model_type = model_type

        if model_type == "ebm":
            fourier_dim = 64
            # We scale the frequencies up slightly so it can catch the thin gap between spiral arms
            self.fourier = FourierFeatures(
                in_features=d,
                mapping_size=fourier_dim,
                scale=0.5,
                # scale=0.3,
            )
            _dim = fourier_dim * 2
        else:
            _dim = d

        # 1. Spatial Encoder (Identical geometry to VelocityMLP for fair comparison)
        self.x_encoder = nn.Sequential(
            nn.Linear(_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )

        # 2. Scalar Potential Decoder
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),  # Scalar output
        )

    def forward(self, x, t=None):
        # Accepts 't' purely for API compatibility with solvers, but ignores it
        if self.model_type == "ebm":
            features = self.fourier(x)
            return self.net(self.x_encoder(features))

        return self.net(self.x_encoder(x))
