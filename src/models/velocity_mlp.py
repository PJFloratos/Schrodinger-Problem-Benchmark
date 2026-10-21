from src.models.components import SinusoidalTimeEmbedding

import torch
from torch import nn

import math


class VelocityMLP(nn.Module):
    def __init__(self, d=2, hidden=128, time_dim=64, model_type="sde"):
        super().__init__()
        self.model_type = model_type

        # 1. Time Embedding & Encoder (Matches DSB pos_dim=16, t_enc_dim=32)
        self.time_embed = SinusoidalTimeEmbedding(dim=time_dim)
        self.time_encoder = nn.Sequential(
            nn.Linear(time_dim, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, time_dim * 2),
            nn.LeakyReLU(),
        )

        # 2. Spatial Encoder (Matches DSB encoder_layers=[16], t_enc_dim=32)
        self.x_encoder = nn.Sequential(
            nn.Linear(d, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, time_dim * 2),
            nn.LeakyReLU(),
        )

        # 3. Decoder Network (Matches DSB decoder_layers=[128, 128])
        self.net = nn.Sequential(
            nn.Linear(time_dim * 4, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, d),
        )

    def forward(self, x, t):
        # Encode spatial features
        xemb = self.x_encoder(x)

        # Obtain sinusoidal representation and encode it
        t_emb = self.time_embed(t)
        temb = self.time_encoder(t_emb)

        # Concatenate encoded representations and decode
        h = torch.cat([xemb, temb], dim=-1)

        return self.net(h)


class EnergyMLP(nn.Module):
    """Models a time-independent scalar potential: R^d -> R"""

    def __init__(self, d=2, hidden=128, time_dim=64, model_type="ebm"):
        super().__init__()
        self.model_type = model_type

        # 1. Spatial Encoder (Identical geometry to VelocityMLP for fair comparison)
        self.x_encoder = nn.Sequential(
            nn.Linear(d, hidden),
            nn.SiLU(),
            nn.Linear(hidden, time_dim * 2),
            nn.SiLU(),
        )

        # 2. Scalar Potential Decoder
        self.net = nn.Sequential(
            nn.Linear(time_dim * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),  # Scalar output
        )

    def forward(self, x, t=None):
        # Accepts 't' purely for API compatibility with solvers, but ignores it
        return self.net(self.x_encoder(x))
