from src.training.base_trainer import BaseTrainer
from src.core.solver import EulerSampler
from src.utils.seed import derive_seed
from src.utils.log import text_logger
from src.utils import save_model

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import math
import time
from tqdm import tqdm
from typing import Any, Tuple, Dict, Optional, Callable, Union


class ENOTTrainer(BaseTrainer):
    """
    Entropic Neural Optimal Transport (ENOT) Trainer.
    Solves the dynamic Schrödinger Problem by recasting it as a minimax game:
      - Discriminator (Dual Potential): Maximizes the dual objective.
      - Generator (Drift Network): Minimizes the finite-energy condition (SDE Rollout).
    """

    logger = text_logger(__name__)

    def __init__(
        self,
        drift_model: nn.Module,  # f_theta (Generator)
        potential_model: nn.Module,  # beta_phi (Discriminator)
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        drift_opt: torch.optim.Optimizer,
        potential_opt: torch.optim.Optimizer,
        metric_logger: Any,
        seed: int,
        device: torch.device,
        sigma: float = 1.0,  # Base noise level
        inner_iters: int = 5,  # K_f: Generator steps per Discriminator step
        integration_steps: int = 50,  # N: Euler-Maruyama discretization steps
        energy_weight: float = 0.5,  # norm_square_normalization_constant (1/eps)
        potential_reg: float = 1e-3,
        gp_weight: float = 0.0,
        grad_clip: float = 1.0,
        ema_mu: float = 0.999,
        use_ema: bool = True,
        use_amp: bool = True,
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

        # Optimization & Architecture
        self.drift_opt = drift_opt
        self.potential_opt = potential_opt
        self.grad_clip = grad_clip
        self.potential_reg = potential_reg
        self.gp_weight = gp_weight

        # Static ENOT Hyperparameters
        self.target_epsilon = sigma**2
        self.inner_iters = inner_iters
        self.energy_weight = energy_weight

        # Sampler for SDE Rollouts
        self.sampler = EulerSampler(
            model_type="enot", steps=integration_steps, use_amp=self.use_amp
        )

        # Dedicated generators for reproducible noise streaming
        self.train_gen_D = torch.Generator(device=self.device).manual_seed(
            derive_seed(self.seed, "enot_train_noise_D")
        )
        self.train_gen_G = torch.Generator(device=self.device).manual_seed(
            derive_seed(self.seed, "enot_train_noise_G")
        )

        # Setup Dual Models (Base, Compiled, EMA)
        self.drift_base, self.drift_model, self.drift_ema, _ = self._setup_model(
            model=drift_model, ema_mu=ema_mu, use_ema=use_ema, use_compile=use_compile
        )
        self.potential_base, self.potential_model, self.potential_ema, _ = (
            self._setup_model(
                model=potential_model,
                ema_mu=ema_mu,
                use_ema=use_ema,
                use_compile=use_compile,
            )
        )

        # Ensure the model identifies itself for inference compatibility
        self.drift_base.model_type = "enot"
        self.drift_base.sigma = self.sigma

        # AMP Scalers for both models
        self.drift_scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.potential_scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def _training_step(self, dl: DataLoader, epoch: int) -> Tuple[float, float]:
        total_loss_D = torch.tensor(0.0, device=self.device)
        total_loss_G = torch.tensor(0.0, device=self.device)

        for y_real in tqdm(dl, ascii=True, desc="              train"):
            y_real = y_real.to(
                self.device, non_blocking=True, memory_format=self.memory_format
            )

            # JOINT PHASE: Shared Rollout Trick
            for step_k in range(self.inner_iters):
                # 1. Shared SDE Rollout (Keep gradients attached for the Generator)
                x0 = self._randn_like(y_real, self.train_gen_G)
                xN, norm_square = self.sampler.rollout_enot_training(
                    model=self.drift_model,
                    x0=x0,
                    epsilon=self.target_epsilon,
                    gen=self.train_gen_G,
                )

                # 2. DISCRIMINATOR PHASE (Only runs on the first step of the batch)
                if step_k == 0:
                    # DETACH xN so gradients don't flow back into the Generator
                    xN_detached = xN.detach()

                    # Track inputs for the Full-Space Gradient Penalty
                    inputs_tracked = (
                        torch.cat([xN_detached, y_real]).detach().requires_grad_(True)
                    )

                    with torch.autocast(
                        self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
                    ):
                        # d = self.potential_model(torch.cat([xN_detached, y_real]))
                        d = self.potential_model(inputs_tracked).squeeze(-1)
                        d_fake, d_real = d.float().chunk(2)

                        # Minimax Objective + Exact L2 Magnitude Anchor from EBM
                        loss_D = (d_fake - d_real).mean() + self.potential_reg * (
                            d_fake.pow(2).mean() + d_real.pow(2).mean()
                        )

                    # Full-Space Gradient Penalty (The BPTT Stabilizer)
                    if torch.is_grad_enabled():
                        grad_d = torch.autograd.grad(
                            outputs=d.sum(),
                            inputs=inputs_tracked,
                            create_graph=True,
                            retain_graph=True,
                        )[0]
                        loss_D = (
                            loss_D + self.gp_weight * grad_d.pow(2).sum(dim=1).mean()
                        )

                    self._optimize_step(
                        base_model=self.potential_base,
                        opt=self.potential_opt,
                        scaler=self.potential_scaler,
                        loss=loss_D,
                        ema=self.potential_ema,
                        log=True,
                        epoch=epoch,
                        phase="train_D",
                    )
                    total_loss_D += loss_D.detach().float()

                # 3. GENERATOR PHASE (Runs every step, reusing xN on step_k == 0)
                # Freeze Discriminator to save massive VRAM and backward compute time
                self.potential_base.requires_grad_(False)

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.use_amp,
                ):
                    loss_G = (
                        self.energy_weight * norm_square
                        - self.potential_model(xN).squeeze(-1)
                    ).mean()

                self._optimize_step(
                    base_model=self.drift_base,
                    opt=self.drift_opt,
                    scaler=self.drift_scaler,
                    loss=loss_G,
                    ema=self.drift_ema,
                    log=True,
                    epoch=epoch,
                    phase="train_G",
                )

                # Unfreeze Discriminator for the next batch
                self.potential_base.requires_grad_(True)

                total_loss_G += loss_G.detach().float()

        num_batches = len(dl)
        return (total_loss_D / num_batches).item(), (
            total_loss_G / (num_batches * self.inner_iters)
        ).item()

    def fit(
        self,
        epochs: int,
        save_per: Union[int, None] = None,
        save_path: Union[str, None] = None,
        eval_per: Optional[int] = None,
        eval_callback: Optional[Callable] = None,
    ) -> Dict:
        train_dl = self._get_loader(shuffle_context="enot_train_loader")

        for epoch in range(1, epochs + 1):
            self.logger.debug(f"-> Epoch: {epoch}/{epochs}")
            phase_start = time.time()

            self.drift_model.train()
            self.potential_model.train()
            train_loss_D, train_loss_G = self._training_step(train_dl, epoch)

            self.drift_model.eval()
            self.potential_model.eval()

            drift_eval = self.drift_ema.model if self.drift_ema else self.drift_base
            potential_eval = (
                self.potential_ema.model if self.potential_ema else self.potential_base
            )

            metrics = {
                "train_loss_D": train_loss_D,
                "train_loss_G": train_loss_G,
                "phase_time_sec": time.time() - phase_start,
            }

            _run_eval = (
                eval_callback is not None
                and eval_per is not None
                and (epoch % eval_per == 0)
            )
            if _run_eval:
                with self._isolated_global_rng(derive_seed(self.seed, "enot_eval")):
                    metrics.update(
                        eval_callback(drift_eval, direction="b", epoch=epoch)
                    )
                _log = self.logger.info
            else:
                _log = self.logger.debug

            self.log_phase_end("epoch", epoch, metrics)
            self.track_parameter_drift(self.drift_base, ipf_iter=epoch)

            qual_str = self._format_eval_str(metrics, _run_eval)
            _log(
                f"     Epoch {epoch} | D Loss: {train_loss_D:.4f} | G Loss: {train_loss_G:.4f} | {qual_str}"
            )

            if save_per and save_path and (epoch % save_per == 0):
                save_model(drift_eval, f"{save_path}/ENOT_G_ep{epoch}.pth")
                save_model(potential_eval, f"{save_path}/ENOT_D_ep{epoch}.pth")

        self.log_compute_cost([self.drift_base, self.potential_base])
        self.logger.debug("Training Process Completed.")

        drift_eval = self.drift_ema.model if self.drift_ema else self.drift_base
        if save_path:
            save_model(drift_eval, f"{save_path}/ENOT_G_final.pth")
            save_model(potential_eval, f"{save_path}/ENOT_D_final.pth")

        return drift_eval, metrics
