from src.training.trainer import Trainer
from src.utils.seed import SeedOffsets

import torch

import math
import contextlib
from tqdm import tqdm
from typing import Any, Tuple


class EBMTrainer(Trainer):
    """
    Neural Static Schrödinger Bridge via Energy-Based Modeling (Semi-Dual).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        opt: torch.optim.Optimizer,
        metric_logger: Any,
        seed: int,
        sigma: float = 1.0,
        ula_steps: int = 50,
        ula_step_size: float = 1e-3,
        spatial_clip: float = 10.0,
        l2_reg: float = 1e-4,
        ebm_steps: int = 1000,  # K_infer for generation
        eta_min: float = 1e-4,
        cold_start: float = 0.02,
        **kwargs,
    ) -> None:
        # Pass standard arguments up to the standard Trainer
        super().__init__(
            model=model,
            dataset=dataset,
            batch_size=batch_size,
            opt=opt,
            metric_logger=metric_logger,
            sigma=sigma,
            seed=seed,
            **kwargs,
        )
        self.eta = ula_step_size
        self.eta_min = eta_min
        self.ula_steps = ula_steps
        # Scale the clip dynamically: weaker prior (high eps) requires larger neural gradients
        self.spatial_clip = spatial_clip / (sigma**2)
        self.l2_reg = l2_reg
        self.ebm_steps = ebm_steps
        self.cold_start = cold_start
        self.buffer_size = 10000

    def _init_replay_buffer(self, sample_shape: torch.Size, dtype: torch.dtype) -> None:
        """Initializes the persistent persistent MCMC replay buffer."""
        g_init = torch.Generator(device=self.device).manual_seed(
            self.seed + SeedOffsets.EBM_BUFFER_INIT
        )
        shape = (self.buffer_size,) + sample_shape

        self.replay_buffer_x = torch.randn(
            shape, device=self.device, dtype=dtype, generator=g_init
        )
        self.replay_buffer_y = self.replay_buffer_x + self.sigma * torch.randn(
            shape, device=self.device, dtype=dtype, generator=g_init
        )

    @torch.no_grad()
    def _generate_fake_samples(
        self,
        model: torch.nn.Module,
        batch_size: int,
        sample_shape: torch.Size,
        dtype: torch.dtype,
        gen: torch.Generator,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The Negative Phase: Uses Unadjusted Langevin Algorithm (ULA) to sample from the model."""
        # 1. Fetch from Replay Buffer
        idx = torch.randperm(self.buffer_size, device=self.device, generator=gen)[
            :batch_size
        ]
        x_fake = self.replay_buffer_x[idx].clone()
        y_fake = self.replay_buffer_y[idx].clone()

        # 2. Cold Start Injection (5%)
        cold = (
            torch.rand(batch_size, device=self.device, generator=gen) < self.cold_start
        )
        n_cold = int(cold.sum())
        if n_cold > 0:
            xc = torch.randn(
                (n_cold, *sample_shape), device=self.device, dtype=dtype, generator=gen
            )
            x_fake[cold] = xc
            y_fake[cold] = xc + self.sigma * torch.randn(
                xc.shape, device=self.device, dtype=dtype, generator=gen
            )

        # Pre-allocate noise to avoid allocating new memory inside the tight ULA loop
        noise_buffer = torch.empty_like(y_fake)

        # 3. Langevin MCMC Steps
        for i in range(self.ula_steps):
            # Anneal the step size so particles freeze into the dots instead of blurring
            progress = i / max(1, self.ula_steps - 1)
            current_eta = self.eta * (self.eta_min / self.eta) ** progress

            # y_fake.requires_grad_(True)
            y_fake = y_fake.detach().requires_grad_(True)
            with torch.enable_grad():
                # psi_fake = model(y_fake)
                # Let MCMC benefit from AMP if it is enabled
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.use_amp,
                ):
                    psi_fake = model(y_fake)
                grad_psi = torch.autograd.grad(
                    outputs=psi_fake.sum(), inputs=y_fake, create_graph=False
                )[0]

            if self.spatial_clip > 0:
                grad_psi = grad_psi.clamp(-self.spatial_clip, self.spatial_clip)

            drift = grad_psi + (x_fake - y_fake) / (self.sigma**2)

            # Fill the pre-allocated buffer with standard normal noise
            noise_buffer.normal_(generator=gen)

            y_fake = (
                y_fake.detach()
                + 0.5 * current_eta * drift
                + math.sqrt(current_eta) * noise_buffer
            )

        return x_fake, y_fake, idx

    def _compute_contrastive_loss(
        self, model: torch.nn.Module, y_real: torch.Tensor, y_fake: torch.Tensor
    ) -> torch.Tensor:
        """The Positive Phase: Computes Contrastive Divergence + L2 Regularization."""
        psi_real = model(y_real)
        psi_gen = model(y_fake.detach())

        loss = psi_gen.mean() - psi_real.mean()
        loss = loss + self.l2_reg * (psi_real.pow(2).mean() + psi_gen.pow(2).mean())
        return loss

    def _process_data_loaders(
        self, dl: torch.utils.data.DataLoader, epoch: int, model: torch.nn.Module
    ) -> float:
        total_loss = torch.tensor(0.0, device=self.device)
        training = model.training
        phase = "train" if training else "valid"
        gen = self.train_gen if training else self.valid_gen

        for y_batch in tqdm(dl, ascii=True, desc=f"              {phase}"):
            y_batch = y_batch.to(
                self.device, non_blocking=True, memory_format=self.memory_format
            )
            batch_size = y_batch.shape[0]
            sample_shape = y_batch.shape[1:]

            # 1. Initialize buffer on the very first batch
            if getattr(self, "replay_buffer_x", None) is None:
                self._init_replay_buffer(sample_shape, y_batch.dtype)

            # 2. Negative Phase (Sample Generation)
            x_fake, y_fake, idx = self._generate_fake_samples(
                model=model,
                batch_size=batch_size,
                sample_shape=sample_shape,
                dtype=y_batch.dtype,
                gen=gen,
            )

            # 3. Update Persistent Chains
            if training:
                self.replay_buffer_x[idx] = x_fake.detach()
                self.replay_buffer_y[idx] = y_fake.detach()

            # 4. Positive Phase (Loss Computation)
            if training:
                self.opt.zero_grad(set_to_none=True)

            cm = contextlib.nullcontext() if training else torch.no_grad()
            with cm:
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.use_amp,
                ):
                    loss = self._compute_contrastive_loss(model, y_batch, y_fake)

            # 5. Model Optimization
            if training:
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.opt)
                else:
                    loss.backward()

                self.log_inner_step(
                    model=model,
                    loss=loss,
                    optimizer=self.opt,
                    phase=phase,
                    ipf_iter=epoch,
                )

                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.model_base.parameters(), self.grad_clip
                    )

                if self.use_amp:
                    self.scaler.step(self.opt)
                    self.scaler.update()
                else:
                    self.opt.step()

                if self.use_ema:
                    self.ema.update(self.model_base)

            total_loss += loss.detach().float()

        return (total_loss / len(dl)).item()

    def _validation_step(
        self, valid_dl: torch.utils.data.DataLoader, epoch: int, model: torch.nn.Module
    ) -> float:
        """
        Contrastive Divergence loss on a validation set is mathematically uninformative
        because the partition function is intractable. Running 50 steps of MCMC per
        validation batch wastes massive compute. We bypass it entirely.
        """
        return None
