import torch
from torch import nn

import math


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t shape: (Batch, 1)
        device = t.device
        half_dim = self.dim // 2

        # Create varying frequencies
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)

        # Apply sine and cosine
        embeddings = t * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class VelocityMLP(nn.Module):
    def __init__(self, d=2, hidden=128, time_dim=64, model_type="sde"):
        super().__init__()
        if model_type not in ["sde", "minibatch", "flow_m", "ipf"]:
            self.model_type = "sde"
        else:
            self.model_type = model_type

        # 1. Time Embedding & Encoder (Matches DSB pos_dim=16, t_enc_dim=32)
        self.time_embed = SinusoidalTimeEmbedding(dim=time_dim)
        self.time_encoder = nn.Sequential(
            nn.Linear(time_dim, 16),
            nn.LeakyReLU(),
            nn.Linear(16, time_dim * 2),
            nn.LeakyReLU(),
        )

        # 2. Spatial Encoder (Matches DSB encoder_layers=[16], t_enc_dim=32)
        self.x_encoder = nn.Sequential(
            nn.Linear(d, 16),
            nn.LeakyReLU(),
            nn.Linear(16, time_dim * 2),
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
        # return self.net(torch.cat([x, t], dim=-1))

        # Obtain sinusoidal representation and encode it
        t_emb = self.time_embed(t)
        temb = self.time_encoder(t_emb)

        # Encode spatial features
        xemb = self.x_encoder(x)

        # Concatenate encoded representations and decode
        h = torch.cat([xemb, temb], dim=-1)
        return self.net(h)

    @torch.no_grad()
    def generate(
        self,
        n_samples: int = 2000,
        steps: int = 500,
        device: torch.device = torch.device("cpu"),
        return_path: bool = False,
        t_end: float = 1.0,  # to prevent variance collapse in SDE
    ) -> torch.Tensor:
        """
        Simulates the SDE to generate samples from the standard normal prior.
        """
        h = 1.0 / steps
        x = torch.randn(n_samples, 2, device=device)

        # Calculate how many steps to actually execute
        integration_steps = int(steps * t_end)

        # Store initial positions if tracking is enabled
        if return_path:
            trajectories = [x.clone().cpu().numpy()]

        for i in range(integration_steps):
            t_val = i * h
            t_tensor = torch.full((n_samples, 1), t_val, device=device)
            drift = self.forward(x, t_tensor)

            if self.model_type in ["minibach", "flow_m"]:
                # Forward ODE: dx = u_t(x) dt (No noise injection!)
                x = x + h * drift

            elif self.model_type == "sde":
                # Omit noise strictly on the final integration step
                if i < integration_steps - 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                # Forward SDE: dx = u_t(x) dt + dW_t
                x = x + h * drift + (h**0.5) * noise

            if return_path:
                trajectories.append(x.clone().cpu().numpy())

        if return_path:
            import numpy as np

            # Stack into shape: (n_samples, steps + 1, 2)
            return np.stack(trajectories, axis=1)

        return x
