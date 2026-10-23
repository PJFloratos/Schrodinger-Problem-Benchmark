from src.training.base_trainer import BaseTrainer
from src.core.dynamics import ConditionalVectorField
from src.models.components import SCONEWrapper
from src.utils.log import text_logger
from src.utils.seed import derive_seed

import torch
from torch.utils.data import DataLoader, random_split

import math
import time
import contextlib
from tqdm import tqdm
from typing import Any, Tuple, Dict, Optional, Callable, Union


class SCONETrainer(BaseTrainer):
    """
    Full Dual Schrödinger Bridge via Joint Energy-Based Modeling.
    Simultaneously trains two dual potentials (phi, psi) using Joint Langevin Dynamics.
    """

    logger = text_logger(__name__)

    def __init__(
        self,
        phi_model: torch.nn.Module,
        psi_model: torch.nn.Module,
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        phi_opt: torch.optim.Optimizer,
        psi_opt: torch.optim.Optimizer,
        metric_logger: Any,
        seed: int,
        device: torch.device,
        train_prop: float = 0.8,
        sigma: float = 1.0,
        reference: str = "brownian",
        ula_steps: int = 100,
        ula_step_size: float = 5e-3,
        eta_min: float = 1e-4,
        spatial_clip: float = 10.0,
        l2_reg: float = 1e-5,
        grad_clip: float = 1.0,
        ema_mu: float = 0.999,
        use_ema: bool = True,
        use_amp: bool = False,
        use_compile: bool = True,
    ) -> None:
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            seed=seed,
            device=device,
            metric_logger=metric_logger,
            sigma=sigma,
            use_amp=use_amp,
        )
        self.phi_opt = phi_opt
        self.psi_opt = psi_opt
        self.train_prop = train_prop
        self.grad_clip = grad_clip

        self.reference = reference
        self.eta = ula_step_size
        self.eta_min = eta_min
        self.ula_steps = ula_steps
        self.spatial_clip = spatial_clip
        self.l2_reg = l2_reg
        self.buffer_size = 10000

        self.train_gen = torch.Generator(device=self.device).manual_seed(
            derive_seed(self.seed, "scone_train_noise")
        )
        self.valid_gen = torch.Generator(device=self.device).manual_seed(
            derive_seed(self.seed, "scone_valid_noise")
        )

        # Setup Dual Models
        self.phi_base, self.phi_model, self.phi_ema, _ = self._setup_model(
            model=phi_model, ema_mu=ema_mu, use_ema=use_ema, use_compile=use_compile
        )
        self.psi_base, self.psi_model, self.psi_ema, _ = self._setup_model(
            model=psi_model, ema_mu=ema_mu, use_ema=use_ema, use_compile=use_compile
        )

        self.phi_scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.psi_scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    @contextlib.contextmanager
    def _freeze_weights(self, *models: torch.nn.Module):
        """Temporarily disables weight gradients to massively speed up MCMC graph building."""
        try:
            for m in models:
                for p in m.parameters():
                    p.requires_grad_(False)
            yield
        finally:
            for m in models:
                for p in m.parameters():
                    p.requires_grad_(True)

    def _get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        split_gen = torch.Generator().manual_seed(
            derive_seed(self.seed, "scone_shuffle")
        )
        train_ds, valid_ds = random_split(
            self.dataset, [self.train_prop, 1 - self.train_prop], generator=split_gen
        )
        is_toy = len(self.data_shape) == 1
        nw = 0 if is_toy else 2

        train_dl = DataLoader(
            train_ds,
            self.batch_size,
            shuffle=True,
            num_workers=nw,
            pin_memory=not is_toy,
            drop_last=True,
            generator=torch.Generator().manual_seed(
                derive_seed(self.seed, "scone_train_loader")
            ),
        )
        valid_dl = DataLoader(
            valid_ds,
            self.batch_size,
            shuffle=False,
            num_workers=nw,
            pin_memory=not is_toy,
            drop_last=True,
            generator=torch.Generator().manual_seed(
                derive_seed(self.seed, "scone_valid_loader")
            ),
        )
        return train_dl, valid_dl

    def _init_replay_buffer(self, sample_shape: torch.Size, dtype: torch.dtype) -> None:
        g_init = torch.Generator(device=self.device).manual_seed(
            derive_seed(self.seed, "scone_buffer_init")
        )
        shape = (self.buffer_size,) + sample_shape

        # Initialize from the target Joint Reference Measure R_01
        self.replay_buffer_x = torch.randn(
            shape, device=self.device, dtype=dtype, generator=g_init
        )
        self.replay_buffer_y = self.replay_buffer_x + self.sigma * torch.randn(
            shape, device=self.device, dtype=dtype, generator=g_init
        )

    @torch.no_grad()
    def _generate_joint_samples(
        self, batch_size: int, gen: torch.Generator, y_real
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The Negative Phase: Joint Unadjusted Langevin Algorithm (ULA)"""
        idx = torch.randperm(self.buffer_size, device=self.device, generator=gen)[
            :batch_size
        ]
        x_fake = self.replay_buffer_x[idx].clone()
        y_fake = self.replay_buffer_y[idx].clone()

        # --- 5% Cold Start to stabilize fast PCD ---
        cold_start_prob = 0.05
        cold = (
            torch.rand(batch_size, device=self.device, generator=gen) < cold_start_prob
        )
        n_cold = int(cold.sum())
        if n_cold > 0:
            # Prior side: standard normal
            x_fake[cold] = torch.randn(
                (n_cold, *self.data_shape),
                device=self.device,
                dtype=x_fake.dtype,
                generator=gen,
            ).contiguous(memory_format=self.memory_format)

            # Data side: teleport directly to the real data distribution (with slight noise for exploration)
            idx_real = torch.randperm(batch_size, device=self.device, generator=gen)[
                :n_cold
            ]
            y_fake[cold] = y_real[idx_real].clone() + 0.1 * torch.randn_like(
                y_real[idx_real]
            )

        noise_x = torch.empty_like(x_fake)
        noise_y = torch.empty_like(y_fake)

        # noise_x = torch.empty_like(x_fake)
        # noise_y = torch.empty_like(y_fake)

        for i in range(self.ula_steps):
            # Simulated Annealing
            # progress = i / max(1, self.ula_steps - 1)
            # current_eta = self.eta * (self.eta_min / self.eta) ** progress

            x_fake = x_fake.detach().requires_grad_(True)
            y_fake = y_fake.detach().requires_grad_(True)

            with torch.enable_grad():
                # phi_fake = self.phi_model(x_fake)
                # psi_fake = self.psi_model(y_fake)
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.use_amp,
                ):
                    phi_fake = self.phi_model(x_fake)
                    psi_fake = self.psi_model(y_fake)

                grad_phi = torch.autograd.grad(
                    phi_fake.sum(), x_fake, create_graph=False
                )[0]
                grad_psi = torch.autograd.grad(
                    psi_fake.sum(), y_fake, create_graph=False
                )[0]

            if self.spatial_clip > 0:
                grad_phi = grad_phi.clamp(-self.spatial_clip, self.spatial_clip)
                grad_psi = grad_psi.clamp(-self.spatial_clip, self.spatial_clip)

            # Route reference score logic
            if self.reference == "brownian":
                ref_x, ref_y = ConditionalVectorField.get_reference_scores(
                    x_fake, y_fake, self.sigma, self.reference
                )
            else:
                # TODO: Implement Algorithm 1: Denoising Score Matching network evaluation here
                raise NotImplementedError(
                    f"Pre-trained score network for '{self.reference}' not yet integrated."
                )

            drift_x = grad_phi + ref_x
            drift_y = grad_psi + ref_y

            noise_x.normal_(generator=gen)
            noise_y.normal_(generator=gen)

            x_fake = x_fake + 0.5 * self.eta * drift_x + math.sqrt(self.eta) * noise_x
            y_fake = y_fake + 0.5 * self.eta * drift_y + math.sqrt(self.eta) * noise_y

        return x_fake, y_fake, idx

    def _compute_losses(
        self,
        x_real: torch.Tensor,
        y_real: torch.Tensor,
        x_fake: torch.Tensor,
        y_fake: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The Positive Phase: Dual Contrastive Divergence"""
        phi_real = self.phi_model(x_real)
        phi_fake = self.phi_model(x_fake.detach())
        loss_phi = (
            phi_fake.mean()
            - phi_real.mean()
            + self.l2_reg * (phi_real.pow(2).mean() + phi_fake.pow(2).mean())
        )

        psi_real = self.psi_model(y_real)
        psi_fake = self.psi_model(y_fake.detach())
        loss_psi = (
            psi_fake.mean()
            - psi_real.mean()
            + self.l2_reg * (psi_real.pow(2).mean() + psi_fake.pow(2).mean())
        )

        return loss_phi, loss_psi

    def _process_data_loaders(
        self, dl: DataLoader, epoch: int, is_train: bool
    ) -> Tuple[float, float]:
        total_loss_phi = torch.tensor(0.0, device=self.device)
        total_loss_psi = torch.tensor(0.0, device=self.device)

        phase = "train" if is_train else "valid"
        gen = self.train_gen if is_train else self.valid_gen

        for y_real in tqdm(dl, ascii=True, desc=f"              {phase}"):
            y_real = y_real.to(
                self.device, non_blocking=True, memory_format=self.memory_format
            )
            batch_size = y_real.shape[0]

            x_real = torch.randn(
                y_real.shape, device=self.device, dtype=y_real.dtype, generator=gen
            ).contiguous(memory_format=self.memory_format)

            if getattr(self, "replay_buffer_x", None) is None:
                self._init_replay_buffer(y_real.shape[1:], y_real.dtype)

            # Freeze the network parameters so PyTorch doesn't cache intermediate activations
            with self._freeze_weights(self.phi_model, self.psi_model):
                x_fake, y_fake, idx = self._generate_joint_samples(
                    batch_size, gen, y_real
                )

            if is_train:
                self.replay_buffer_x[idx] = x_fake.detach()
                self.replay_buffer_y[idx] = y_fake.detach()

                self.phi_opt.zero_grad(set_to_none=True)
                self.psi_opt.zero_grad(set_to_none=True)

            cm = contextlib.nullcontext() if is_train else torch.no_grad()
            with cm:
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.use_amp,
                ):
                    loss_phi, loss_psi = self._compute_losses(
                        x_real, y_real, x_fake, y_fake
                    )

            if is_train:
                if self.use_amp:
                    self.phi_scaler.scale(loss_phi).backward()
                    self.phi_scaler.unscale_(self.phi_opt)
                else:
                    loss_phi.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.phi_base.parameters(), self.grad_clip
                    )
                if self.use_amp:
                    self.phi_scaler.step(self.phi_opt)
                    self.phi_scaler.update()
                else:
                    self.phi_opt.step()

                if self.use_amp:
                    self.psi_scaler.scale(loss_psi).backward()
                    self.psi_scaler.unscale_(self.psi_opt)
                else:
                    loss_psi.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.psi_base.parameters(), self.grad_clip
                    )
                if self.use_amp:
                    self.psi_scaler.step(self.psi_opt)
                    self.psi_scaler.update()
                else:
                    self.psi_opt.step()

                if self.phi_ema:
                    self.phi_ema.update(self.phi_base)
                    self.psi_ema.update(self.psi_base)

                self.total_gradient_steps += 1

            total_loss_phi += loss_phi.detach().float()
            total_loss_psi += loss_psi.detach().float()

        num_batches = len(dl)
        return (total_loss_phi / num_batches).item(), (
            total_loss_psi / num_batches
        ).item()

    def fit(
        self,
        epochs: int,
        save_per: Union[int, None] = None,
        save_path: Union[str, None] = None,
        eval_per: Optional[int] = None,
        eval_callback: Optional[Callable] = None,
    ) -> Dict:
        train_dl, valid_dl = self._get_loaders()

        for epoch in range(1, epochs + 1):
            self.logger.debug(f"-> Epoch: {epoch}/{epochs}")
            phase_start = time.time()

            self.phi_model.train()
            self.psi_model.train()
            train_phi, train_psi = self._process_data_loaders(
                train_dl, epoch, is_train=True
            )

            self.phi_model.eval()
            self.psi_model.eval()
            valid_phi, valid_psi = 0.0, 0.0  # Skip intractable CD validation

            # Determine which models to evaluate/save (EMA or base model)
            phi_eval = self.phi_ema.model if self.phi_ema else self.phi_base
            psi_eval = self.psi_ema.model if self.psi_ema else self.psi_base

            metrics = {
                "train_loss_phi": train_phi,
                "train_loss_psi": train_psi,
                "phase_time_sec": time.time() - phase_start,
            }

            _run_eval = (
                eval_callback is not None
                and eval_per is not None
                and (epoch % eval_per == 0)
            )
            if _run_eval:
                # To generate samples, the Evaluator requires both dual potentials. We pack them together.
                eval_wrapper = SCONEWrapper(
                    phi_eval,
                    psi_eval,
                    self.sigma,
                    self.reference,
                )
                with self._isolated_global_rng(derive_seed(self.seed, "scone_eval")):
                    metrics.update(eval_callback(eval_wrapper, direction="b"))

                _log = self.logger.info
            else:
                _log = self.logger.debug

            self.log_phase_end("epoch", epoch, metrics)
            qual_str = self._format_eval_str(metrics, _run_eval)

            _log(
                f"     Epoch {epoch} | Phi Loss: {train_phi:.4f} | Psi Loss: {train_psi:.4f} | {qual_str}"
            )

            # Saving the model
            _save_model = (
                save_per is not None
                and save_path is not None
                and (epoch % save_per == 0)
            )
            if _save_model:
                save_model(
                    phi_eval,
                    f"{save_path}/{self.phi_base.__class__.__name__}_phi_checkpoint_{epoch}.pth",
                )
                save_model(
                    psi_eval,
                    f"{save_path}/{self.psi_base.__class__.__name__}_psi_checkpoint_{epoch}.pth",
                )

            self.logger.debug(("-" * 100))

        # Log final hardware footprint and parameters
        self.log_compute_cost([self.phi_base, self.psi_base])

        self.logger.debug("Training Process Completed Successfully.")

        phi_eval = self.phi_ema.model if self.phi_ema else self.phi_base
        psi_eval = self.psi_ema.model if self.psi_ema else self.psi_base

        eval_wrapper = SCONEWrapper(
            phi_eval,
            psi_eval,
            self.sigma,
            self.reference,
        )

        # Save models after training
        if save_path:
            save_model(
                phi_eval,
                f"{save_path}/{self.phi_base.__class__.__name__}_phi_checkpoint_final.pth",
            )
            save_model(
                psi_eval,
                f"{save_path}/{self.psi_base.__class__.__name__}_psi_checkpoint_final.pth",
            )

        return eval_wrapper, metrics
