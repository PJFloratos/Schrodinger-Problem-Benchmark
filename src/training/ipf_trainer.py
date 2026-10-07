from src.training.base_trainer import BaseTrainer
from src.core.dynamics import ConditionalVectorField
from src.core.solver import EulerSampler
from src.models.components import EMAHelper
from src.utils.seed import SeedOffsets
from src.utils import save_model, text_logger

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import copy
import math
import time
from contextlib import contextmanager
from tqdm import tqdm
from typing import Union, Any, Optional, Callable


class IPFTrainer(BaseTrainer):
    """
    Iterative Proportional Fitting (Diffusion Schrodinger Bridge).

    Conventions (kept identical for both networks):
      * Both networks take *forward* time t in [0, 1] as input.
      * Both networks output a *drift* (per unit time).
      * Forward chain : x_{k+1} = x_k + h * f(x_k, t_k) + sqrt(h) * z
      * Backward chain: y_{i+1} = y_i + h * b(y_i, 1 - t_i) + sqrt(h) * z
    """

    logger = text_logger(__name__)

    def __init__(
        self,
        forward_model: nn.Module,
        backward_model: nn.Module,
        dataset: torch.utils.data.Dataset,
        forward_opt: torch.optim.Optimizer,
        backward_opt: torch.optim.Optimizer,
        device: torch.device,
        metric_logger: Any,
        seed: int,
        batch_size: int = 256,
        sde_steps: int = 20,
        num_cache_batches: int = 10,  # Number of dataset batches to cache per iteration
        refresh_every: int = 500,  # regenerate the cache every N gradient steps
        sigma: float = 1.0,
        grad_clip: float = 2.0,
        ema_mu: float = 0.999,
        lr_decay: bool = True,  # cosine decay inside each training phase
        lr_final_ratio: float = 0.05,  # final lr = ratio * base lr
        use_amp: bool = True,
        use_ema: bool = True,
    ):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            seed=seed,
            device=device,
            metric_logger=metric_logger,
            use_amp=use_amp,
        )

        self.f_opt = forward_opt
        self.b_opt = backward_opt
        self.sde_steps = sde_steps
        self.h = 1.0 / sde_steps
        self.num_cache_batches = num_cache_batches
        self.refresh_every = refresh_every
        self.sigma = sigma
        self.grad_clip = grad_clip
        self.lr_decay = lr_decay
        self.lr_final_ratio = lr_final_ratio
        self.criterion = nn.MSELoss()
        self.use_ema = use_ema

        # DataLoader shuffling (must be a CPU generator).
        self.shuffle_gen = torch.Generator()
        self.shuffle_gen.manual_seed(self.seed + SeedOffsets.IPF_SHUFFLE)

        # Cache-simulation noise: one continuous stream, touched by nothing else.
        self.noise_gen = torch.Generator(device=self.device)
        self.noise_gen.manual_seed(self.seed + SeedOffsets.IPF_CACHE_NOISE)

        # Minibatch order inside each cache.
        self.perm_gen = torch.Generator(device=self.device)
        self.perm_gen.manual_seed(self.seed + SeedOffsets.IPF_CACHE_PERM)

        self._init_cache_dataloader(SeedOffsets.IPF_SHUFFLE)

        # Cache-staleness probe: re-seeded at the start of every probe, so the probe before
        # and after a refresh sees the exact same (index, t, z) draws.
        self.probe_gen = torch.Generator(device=self.device)

        # Amp scalers for the two models
        self.f_scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.b_scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Model setup
        self.f_model_base, self.f_model, self.f_ema, self.f_sampler = self._setup_model(
            model=forward_model,
            ema_mu=ema_mu,
            use_ema=use_ema,
            sigma=self.sigma,
            create_sampler=True,
        )
        self.b_model_base, self.b_model, self.b_ema, self.b_sampler = self._setup_model(
            model=backward_model,
            ema_mu=ema_mu,
            use_ema=use_ema,
            sigma=self.sigma,
            create_sampler=True,
        )

    # ------------------------------------------------------------------
    # Reference Process
    # ------------------------------------------------------------------
    @staticmethod
    def reference_drift(x: torch.Tensor) -> torch.Tensor:
        """Standard Brownian Motion reference process (zero drift)."""
        return torch.zeros_like(x)

    # ------------------------------------------------------------------
    # CACHE GENERATRION
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _simulate_and_cache(
        self,
        source_model: nn.Module,
        src_ema: EMAHelper,
        sampler: nn.Module,
        direction: str,
        ipf_iteration: int,
    ):
        """
        Simulate the chain of `source_model` in `direction` and record regression
        data (x_{k+1}, t_{k+1}, target) for the network of the OPPOSITE direction.

        The chain is simulated with the EMA weights if EMA is enabled, otherwise with
        the live weights.

        Noise is injected at EVERY step (as in the official repo with sample=False).
        Dropping the last noise is only correct when producing final samples.
        """
        use_reference = ipf_iteration == 1 and direction == "f"

        if not use_reference:
            # Simulate with the EMA weights if enabled, else the live weights
            if src_ema is not None:
                src_ema.copy_to(sampler)
            else:
                getattr(sampler, "_orig_mod", sampler).load_state_dict(
                    source_model.state_dict()
                )
            sampler.eval()

        def _get_drift(x, t):
            if use_reference:
                return IPFTrainer.reference_drift(x)

            if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()

            with torch.autocast(
                device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
            ):
                return sampler(x, t)

        # Pre-allocate contiguous memory blocks
        total_samples = self.num_cache_batches * self.batch_size * self.sde_steps
        X_cache = torch.empty(
            (total_samples, *self.data_shape),
            device=self.device,
            memory_format=self.memory_format,
        )
        T_cache = torch.empty((total_samples, 1), device=self.device)
        U_cache = torch.empty(
            (total_samples, *self.data_shape),
            device=self.device,
            memory_format=self.memory_format,
        )

        idx = 0

        for _ in tqdm(
            range(self.num_cache_batches),
            ascii=True,
            leave=False,
            desc=f"Simulating {direction.upper()} Cache",
        ):
            if direction == "f":
                x = self._next_data()
            else:
                x = torch.randn(
                    self.batch_size,
                    *self.data_shape,
                    device=self.device,
                    generator=self.noise_gen,
                ).contiguous(memory_format=self.memory_format)

            b_size = x.shape[0]

            for i in range(self.sde_steps):
                # sampler runs its own chain: time i*h
                t_now = torch.full((b_size, 1), i * self.h, device=self.device)
                # trained net (opposite chain) sees this state at its own time 1-(i+1)*h
                t_next = torch.full(
                    (b_size, 1), 1.0 - (i + 1) * self.h, device=self.device
                )

                drift = _get_drift(x, t_now)
                z = self._randn_like(x, self.noise_gen)

                x_next = EulerSampler.euler_maruyama_step(
                    x, drift, self.h, z, sigma=self.sigma
                )

                drift_next = _get_drift(x_next, t_now)

                target = ConditionalVectorField.get_dsb_target(
                    drift_next, z, self.h, sigma=self.sigma
                )

                # The trained network is queried at x_{k+1}, at the forward time of x_{k+1}.
                X_cache[idx : idx + b_size] = x_next
                T_cache[idx : idx + b_size] = t_next
                U_cache[idx : idx + b_size] = target
                x = x_next
                idx += b_size

        # Save to the BaseTrainer state for the probe!
        self._probe_tensors = (X_cache[:idx], T_cache[:idx], U_cache[:idx])

        return self._cache_iterator(X_cache, T_cache, U_cache)

    # ------------------------------------------------------------------
    # TRAINING LOOP
    # ------------------------------------------------------------------
    def _train_cache(
        self,
        target_model: nn.Module,
        opt: torch.optim.Optimizer,
        ema_helper: EMAHelper,
        make_cache: Callable,
        direction: str,
        num_iter: int,
        ipf_iter: int,
    ) -> float:
        target_model.train()
        scaler = self.b_scaler if direction == "b" else self.f_scaler
        base_lrs = [g["lr"] for g in opt.param_groups]

        cache_iter = make_cache()
        if direction == "f":
            phase = "forward"
            desc = "Training Forward Model"
        else:
            phase = "backward"
            desc = "Training Backward Model"

        total_loss = torch.tensor(0.0, device=self.device)
        prev_loss_val = None

        for it in tqdm(range(num_iter), ascii=True, desc=desc):
            if self.lr_decay:
                r = self.lr_final_ratio
                scale = r + (1 - r) * 0.5 * (1 + math.cos(math.pi * it / num_iter))
                for g, lr0 in zip(opt.param_groups, base_lrs):
                    g["lr"] = lr0 * scale

            if it > 0 and self.refresh_every and it % self.refresh_every == 0:
                pre_loss = self._probe_loss(target_model, "ipf", phase)
                cache_iter = make_cache()
                post_loss = self._probe_loss(target_model, "ipf", phase)
                self.track_cache_staleness(
                    pre_loss, post_loss, self.total_gradient_steps
                )

            x_batch, t_batch, u_batch = next(cache_iter)
            opt.zero_grad(set_to_none=True)

            if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()

            with torch.autocast(
                device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
            ):
                pred_u = target_model(x_batch, t_batch)
                loss = ConditionalVectorField.compute_loss(
                    model_type="ipf",
                    pred_u=pred_u,
                    target_u=u_batch,
                    t_net=t_batch,
                    h=self.h,
                    sigma=self.sigma,
                )

            if self.use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
            else:
                loss.backward()

            # --- BaseTrainer Metric Hook ---
            self.log_inner_step(target_model, loss, opt, phase=phase, ipf_iter=ipf_iter)

            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    target_model.parameters(), self.grad_clip
                )

            if self.use_amp:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()

            if self.use_ema:
                # EMA tracks the live weights after every optimizer step
                ema_helper.update(target_model)

            total_loss += loss.detach().float()

            if prev_loss_val is None or (
                self.refresh_every and it % self.refresh_every == 0
            ):
                prev_loss_val = loss.item()

        for g, lr0 in zip(opt.param_groups, base_lrs):  # restore for the next phase
            g["lr"] = lr0

        return (total_loss / num_iter).item()

    def fit(
        self,
        ipf_iterations: int,
        inner_iterations: int = 5000,
        save_per: Union[int, None] = None,
        save_path: Union[str, None] = None,
        eval_per: Optional[int] = None,
        eval_callback: Optional[callable] = None,
    ):
        # Models to evaluate/save: the EMA copies if enabled, else the live base models
        f_eval = self.f_ema.model if self.use_ema else self.f_model_base
        b_eval = self.b_ema.model if self.use_ema else self.b_model_base

        for n in range(1, ipf_iterations + 1):
            IPFTrainer.logger.debug(f"\n--- IPF Iteration {n}/{ipf_iterations} ---")

            # ==========================================
            # Phase 1: Train Backward Model
            # ==========================================
            phase_start = time.time()
            b_loss = self._train_cache(
                self.b_model,
                self.b_opt,
                self.b_ema,
                lambda: self._simulate_and_cache(
                    self.f_model_base, self.f_ema, self.f_sampler, "f", n
                ),
                "b",
                inner_iterations,
                n,
            )
            b_time = time.time() - phase_start

            # The cache builds num_cache_batches, each running sde_steps
            train_nfes = self.num_cache_batches * self.batch_size * self.sde_steps
            self.total_nfes += train_nfes

            # Aggregate Phase 1 Metrics
            b_metrics = {
                "loss": b_loss,
                "phase_time_sec": b_time,
                "train_nfes": train_nfes,
            }

            # Run evaluation condition
            _run_eval = (
                eval_callback is not None
                and eval_per is not None
                and (n % eval_per == 0)
            )
            if _run_eval:
                with self._isolated_global_rng(self.seed + SeedOffsets.IPF_EVAL):
                    # Trigger the evaluator purely as a callback
                    b_metrics.update(eval_callback(b_eval, direction="b"))

            self.log_phase_end("backward", n, b_metrics)

            qual_str = self._format_eval_str(b_metrics, _run_eval, prefix="B-")
            self.logger.info(
                f"Iteration {n} (Backward) | B-Loss: {b_loss:.4f} | {qual_str}"
            )

            # ==========================================
            # Phase 2: Train Forward Model
            # ==========================================
            phase_start = time.time()
            f_loss = self._train_cache(
                self.f_model,
                self.f_opt,
                self.f_ema,
                lambda: self._simulate_and_cache(
                    self.b_model_base, self.b_ema, self.b_sampler, "b", n
                ),
                "f",
                inner_iterations,
                n,
            )
            f_time = time.time() - phase_start
            self.total_nfes += train_nfes

            # Aggregate Phase 2 Metrics
            f_metrics = {
                "loss": f_loss,
                "phase_time_sec": f_time,
                "train_nfes": train_nfes,
            }
            if _run_eval:
                with self._isolated_global_rng(self.seed + SeedOffsets.IPF_EVAL):
                    f_metrics.update(eval_callback(f_eval, direction="f"))

            self.log_phase_end("forward", n, f_metrics)

            qual_str = self._format_eval_str(f_metrics, _run_eval, prefix="F-")
            self.logger.info(
                f"Iteration {n} (Forward) | F-Loss: {f_loss:.4f} | {qual_str}"
            )

            # --- Outer-Loop Diagnostics ---
            self.track_parameter_drift(self.b_model, ipf_iter=n)
            self.track_path_consistency(self.f_model, self.b_model, ipf_iter=n)

            # --- Intermediate Checkpoint Saving ---
            _save_model = (
                save_per is not None and save_path is not None and (n % save_per == 0)
            )
            if _save_model:
                save_model(
                    b_eval,
                    f"{save_path}/{self.b_model_base.__class__.__name__}_backward_checkpoint_{n}.pth",
                )

        self.logger.debug(("-" * 100))

        # Hand back the smoothed weights in the base models
        if self.use_ema:
            self.f_ema.copy_to(self.f_model_base)
            self.b_ema.copy_to(self.b_model_base)

        # Log final hardware and time footprint
        self.log_compute_cost([self.f_model_base, self.b_model_base])

        # --- Final Model Saving ---
        if save_path:
            save_model(
                self.b_model_base,
                f"{save_path}/{self.b_model_base.__class__.__name__}_backward_final.pth",
            )
            save_model(
                self.f_model_base,
                f"{save_path}/{self.f_model_base.__class__.__name__}_forward_final.pth",
            )

        return self.b_model_base, {"f_metrics": f_metrics, "b_metrics": b_metrics}
