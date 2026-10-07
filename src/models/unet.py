from src.models.components import SinusoidalTimeEmbedding
from src.models.components import ResBlock

import torch
import torch.nn as nn
import math

from tqdm import tqdm
from typing import Optional


class SimpleUNet(nn.Module):
    def __init__(self, channels=1, image_size=28, model_type="sde"):
        super().__init__()
        self.model_type = model_type
        self.channels = channels
        self.image_size = image_size
        time_dim = 64

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(dim=time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        self.conv0 = nn.Conv2d(channels, 32, 3, padding=1)

        # Down blocks
        self.down1 = ResBlock(32, 64, time_dim)
        self.down2 = ResBlock(64, 128, time_dim)

        # Up blocks
        self.up1 = ResBlock(128, 64, time_dim, up=True)
        self.up2 = ResBlock(128, 32, time_dim, up=True)

        self.out_norm = nn.GroupNorm(8, 64)
        self.out_act = nn.SiLU()
        self.out = nn.Conv2d(64, channels, 3, padding=1)

        # Initialize output layer to zero
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, t):
        t = self.time_mlp(t)

        x0 = self.conv0(x)
        x1 = self.down1(x0, t)
        x2 = self.down2(x1, t)

        x = self.up1(x2, t)
        x = self.up2(torch.cat([x, x1], dim=1), t)

        # Apply the new output projection
        x = self.out(self.out_act(self.out_norm(torch.cat([x, x0], dim=1))))

        return x
