from src.training.trainer import Trainer
from src.utils.seed import derive_seed

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
        gamma: float = 1.0,
        buffer_size: int = 10000,
        ula_steps: int = 50,
        ula_step_size: float = 1e-3,
        spatial_clip: float = 10.0,
        l2_reg: float = 1e-4,
        gp_weight: float = 0.0,
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

        # Precompute the analytical transition moments at t=1
        # self.gamma = gamma
        # self.alpha = math.exp(-self.gamma)
        # self.beta_sq = (sigma**2 / (2 * self.gamma)) * (1 - math.exp(-2 * self.gamma))
        # self.beta = math.sqrt(self.beta_sq)
        self.sigma = sigma

        self.l2_reg = l2_reg
        self.ebm_steps = ebm_steps
        self.cold_start = cold_start
        self.buffer_size = buffer_size
        self.gp_weight = gp_weight

    def _init_replay_buffer(self, sample_shape: torch.Size, dtype: torch.dtype) -> None:
        """Initializes the persistent persistent MCMC replay buffer."""
        g_init = torch.Generator(device=self.device).manual_seed(
            derive_seed(self.seed, "ebm_buffer_init")
        )
        shape = (self.buffer_size,) + sample_shape

        self.replay_buffer_x = (
            torch.randn(shape, device=self.device, dtype=dtype, generator=g_init)
            * self.sigma
        )
        self.replay_buffer_y = self.replay_buffer_x + self.sigma * torch.randn(
            shape, device=self.device, dtype=dtype, generator=g_init
        )
        # self.replay_buffer_y = (
        #     self.alpha * self.replay_buffer_x
        #     + self.beta
        #     * torch.randn(shape, device=self.device, dtype=dtype, generator=g_init)
        # )

    @torch.no_grad()
    def _generate_fake_samples(
        self,
        model: torch.nn.Module,
        batch_size: int,
        sample_shape: torch.Size,
        dtype: torch.dtype,
        gen: torch.Generator,
        y_real: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The Negative Phase: Uses Unadjusted Langevin Algorithm (ULA) to sample from the model."""
        # 1. Fetch from Replay Buffer
        idx = torch.randperm(self.buffer_size, device=self.device, generator=gen)[
            :batch_size
        ]
        x_fake = self.replay_buffer_x[idx].clone()
        y_fake = self.replay_buffer_y[idx].clone()

        # This constant wash of fresh prior particles prevents the buffer from
        # stagnating in "islands" and forces continuous exploration.
        cold_mask = (
            torch.rand(batch_size, device=self.device, generator=gen) < self.cold_start
        )
        n_cold = int(cold_mask.sum())
        if n_cold > 0:
            xc = (
                torch.randn(
                    (n_cold, *sample_shape),
                    device=self.device,
                    dtype=dtype,
                    generator=gen,
                )
                * self.sigma
            )
            x_fake[cold_mask] = xc
            # y_fake[cold_mask] = self.alpha * xc + self.beta * torch.randn_like(xc)
            y_fake[cold_mask] = xc + self.sigma * torch.randn(
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

            # drift = grad_psi + (self.alpha * x_fake - y_fake) / self.beta_sq
            drift = grad_psi + (x_fake - y_fake) / self.sigma**2

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

        # 1. Create fresh leaf tensors strictly for tracking the gradient penalty.
        # We detach y_fake first so PyTorch doesn't try to backpropagate
        # all the way through the 80 MCMC loops!
        y_real_tracked = y_real.detach().requires_grad_(True)
        y_fake_tracked = y_fake.detach().requires_grad_(True)

        # 2. Forward passes using the specifically tracked tensors
        psi_real = model(y_real_tracked)
        psi_gen = model(y_fake_tracked)

        # 3. Standard Contrastive Divergence
        loss = psi_gen.mean() - psi_real.mean()

        # 4. Soft Two-Sided Anchor (Prevents the -2400 runaway scale)
        loss = loss + self.l2_reg * (psi_real.pow(2).mean() + psi_gen.pow(2).mean())

        # 5. Full-Space Gradient Penalty (The Black Hole Preventer)
        if torch.is_grad_enabled():
            # Gradients on the data manifold
            grad_real = torch.autograd.grad(
                outputs=psi_real.sum(),
                inputs=y_real_tracked,
                create_graph=True,
                retain_graph=True,
            )[0]

            # Gradients on the MCMC particles
            grad_fake = torch.autograd.grad(
                outputs=psi_gen.sum(),
                inputs=y_fake_tracked,
                create_graph=True,
                retain_graph=True,
            )[0]

            loss = loss + self.gp_weight * (
                grad_real.pow(2).sum(dim=1).mean() + grad_fake.pow(2).sum(dim=1).mean()
            )

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
                y_real=y_batch,
            )

            # 3. Update Persistent Chains
            if training:
                self.replay_buffer_x[idx] = x_fake.detach()
                self.replay_buffer_y[idx] = y_fake.detach()

            # 4. Positive Phase (Loss Computation)
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
                self._optimize_step(
                    base_model=self.model_base,
                    opt=self.opt,
                    scaler=self.scaler,
                    loss=loss,
                    ema=self.ema,
                    log=True,
                    phase=phase,
                    epoch=epoch,
                )

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
