from src.training.base_trainer import BaseTrainer
from src.core.dynamics import ConditionalVectorField
from src.core.solver import EulerSampler
from src.models.components import EMAHelper
from src.utils.seed import SeedOffsets
from src.utils import save_model, text_logger

import torch
from torch import nn
from torch.utils.data import DataLoader

import copy
import math
import time
from contextlib import contextmanager

from tqdm import tqdm
from typing import Any, Callable, Optional, Tuple, Union


class IMFTrainer(BaseTrainer):
    """
    What differs from IPF: the cache stores ENDPOINT PAIRS (x0, x1) only, not whole trajectories.
    Every gradient step draws a fresh t ~ U[eps, 1-eps] and fresh noise, and regresses onto
    the Brownian-bridge targets
        x_t   = (1 - t) x0 + t x1 + sigma sqrt(t (1 - t)) z
        f-tgt = (x1 - x_t) / (1 - t)          (= x1 - x0 - sigma sqrt(t / (1 - t)) z)
        b-tgt = (x0 - x_t) / t                (= -(x1 - x0) - sigma sqrt((1 - t) / t) z)

    One IMF iteration n:
        phase B: train b on couplings   (x0 = data, x1 = forward-chain sample)  [n = 0: first coupling]
        phase F: train f on couplings   (x1 = prior, x0 = backward-chain sample)
    first_coupling (used at n = 0 only):
        "ind": (data, N(0, I)) independent. Used for BOTH b and f at n = 0.
        "ref": (data, data + sigma * eps), the Brownian reference. Used for b only (like IPF's
               first step); f is then trained on b's simulation.
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
        num_cache_batches: int = 10,  # cache holds num_cache_batches * batch_size PAIRS
        refresh_every: int = 500,  # regenerate the pairs every N gradient steps
        sigma: float = 1.0,  # Brownian reference volatility (IPF is hard-wired to 1)
        eps: float = 1e-4,  # t ~ U[eps, 1 - eps]; targets blow up like 1/t at the ends
        first_coupling: str = "ref",  # in ["ind", "ref"]
        grad_clip: float = 2.0,
        ema_mu: float = 0.999,
        lr_decay: bool = True,  # cosine decay inside each training phase
        lr_final_ratio: float = 0.05,  # final lr = ratio * base lr
        use_amp: bool = True,
        use_ema: bool = True,
    ):
        if first_coupling not in ("ind", "ref"):
            raise ValueError(
                f"first_coupling must be 'ind' or 'ref', got {first_coupling!r}"
            )

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
        self.eps = eps
        self.first_coupling = first_coupling
        self.grad_clip = grad_clip
        self.lr_decay = lr_decay
        self.lr_final_ratio = lr_final_ratio
        self.use_ema = use_ema

        # DataLoader shuffling (must be a CPU generator).
        self.shuffle_gen = torch.Generator()
        self.shuffle_gen.manual_seed(self.seed + SeedOffsets.IMF_SHUFFLE)

        # Coupling-construction noise: one continuous stream, touched by nothing else.
        self.noise_gen = torch.Generator(device=self.device)
        self.noise_gen.manual_seed(self.seed + SeedOffsets.IMF_PAIR_NOISE)

        # Minibatch indices into the pair cache.
        self.perm_gen = torch.Generator(device=self.device)
        self.perm_gen.manual_seed(self.seed + SeedOffsets.IMF_PAIR_PERM)

        # Fresh (t, z) at every gradient step.
        self.bridge_gen = torch.Generator(device=self.device)
        self.bridge_gen.manual_seed(self.seed + SeedOffsets.IMF_BRIDGE_NOISE)

        self._init_cache_dataloader(SeedOffsets.IMF_SHUFFLE)

        # The AMP scalers for the two models
        self.f_scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.b_scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Model setup
        self.f_model_base, self.f_model, self.f_ema, self.f_sampler = self._setup_model(
            model=forward_model,
            ema_mu=ema_mu,
            use_ema=use_ema,
            sigma=sigma,
            create_sampler=True,
        )
        self.b_model_base, self.b_model, self.b_ema, self.b_sampler = self._setup_model(
            model=backward_model,
            ema_mu=ema_mu,
            use_ema=use_ema,
            sigma=sigma,
            create_sampler=True,
        )

    # ------------------------------------------------------------------
    # COUPLING GENERATION
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _make_pairs(
        self, phase: str, imf_iter: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Endpoint couplings (X0, X1) used to train the network of `phase` ("b" or "f").
          phase "b": x0 = data,  x1 = forward chain (EMA of f) simulated from x0
          phase "f": x1 = prior, x0 = backward chain (EMA of b) simulated from x1
        """
        use_first = imf_iter == 1 and (phase == "b" or self.first_coupling == "ind")

        if not use_first:
            # Grab the uncompiled base, the EMA, and our persistent compiled sampler
            src_base, src_ema, sampler = (
                (self.f_model_base, self.f_ema, self.f_sampler)
                if phase == "b"
                else (self.b_model_base, self.b_ema, self.b_sampler)
            )

            # Simulate with the EMA weights if enabled, else the live weights
            if src_ema is not None:
                src_ema.copy_to(sampler)
            else:
                getattr(sampler, "_orig_mod", sampler).load_state_dict(
                    src_base.state_dict()
                )
            sampler.eval()

        # Initialize the generalized sampler engine for cache building
        sampler_engine = EulerSampler(
            model_type="imf", steps=self.sde_steps, use_amp=self.use_amp
        )

        # Pre-allocate contiguous memory blocks on the GPU
        total_samples = self.num_cache_batches * self.batch_size
        X0_cache = torch.empty(
            (total_samples, *self.data_shape),
            device=self.device,
            memory_format=self.memory_format,
        )
        X1_cache = torch.empty(
            (total_samples, *self.data_shape),
            device=self.device,
            memory_format=self.memory_format,
        )

        idx = 0

        for _ in tqdm(
            range(self.num_cache_batches),
            ascii=True,
            leave=False,
            desc=f"Building {phase.upper()} couplings",
        ):
            if use_first:
                x0 = self._next_data()
                if self.first_coupling == "ref":
                    x1 = x0 + self.sigma * torch.randn_like(x0)
                else:
                    x1 = self._randn_like(x0, self.noise_gen)
            elif phase == "b":
                x0 = self._next_data()
                x1 = sampler_engine.generate(
                    model=sampler,
                    x_init=x0,
                    batch_size=self.batch_size,
                    step_seed=self.noise_gen,
                    drop_last_noise=False,
                    verbose=False,
                )
            else:
                x1 = torch.randn(
                    self.batch_size,
                    *self.data_shape,
                    device=self.device,
                    generator=self.noise_gen,
                ).contiguous(memory_format=self.memory_format)
                # x0 = self._simulate(sampler, x1)
                x0 = sampler_engine.generate(
                    model=sampler,
                    x_init=x1,
                    batch_size=self.batch_size,
                    step_seed=self.noise_gen,
                    drop_last_noise=False,
                    verbose=False,
                )

            # Write directly into the pre-allocated tensors
            b_size = x0.shape[0]
            X0_cache[idx : idx + b_size] = x0
            X1_cache[idx : idx + b_size] = x1
            idx += b_size

        # Save to the BaseTrainer state for the probe
        self._probe_tensors = (X0_cache[:idx], X1_cache[:idx])

        return self._cache_iterator(X0_cache, X1_cache)

    # ------------------------------------------------------------------
    # TRAINING LOOP
    # ------------------------------------------------------------------
    def _train_phase(
        self,
        model: nn.Module,
        opt: torch.optim.Optimizer,
        ema_helper: EMAHelper,
        make_cache: Callable,
        phase: str,
        num_iter: int,
        imf_iter: int,
    ) -> float:
        model.train()
        scaler = self.f_scaler if phase == "f" else self.b_scaler
        base_lrs = [g["lr"] for g in opt.param_groups]

        cache_iter = make_cache()

        if phase == "f":
            log_phase, desc = "forward", "Training Forward Model"
        else:
            log_phase, desc = "backward", "Training Backward Model"

        total_loss = torch.tensor(0.0, device=self.device)
        for it in tqdm(range(num_iter), ascii=True, desc=desc):
            if self.lr_decay:
                r = self.lr_final_ratio
                scale = r + (1 - r) * 0.5 * (1 + math.cos(math.pi * it / num_iter))
                for g, lr0 in zip(opt.param_groups, base_lrs):
                    g["lr"] = lr0 * scale

            if it > 0 and self.refresh_every and it % self.refresh_every == 0:
                pre_loss = self._probe_loss(model, "imf", phase)
                cache_iter = make_cache()
                post_loss = self._probe_loss(model, "imf", phase)
                self.track_cache_staleness(
                    pre_loss, post_loss, self.total_gradient_steps
                )

            x0, x1 = next(cache_iter)
            t = (
                torch.rand(
                    self.batch_size, 1, device=self.device, generator=self.bridge_gen
                )
                * (1.0 - 2.0 * self.eps)
                + self.eps
            )

            x_t, target, t_net = ConditionalVectorField.get_interpolant_and_target(
                model_type="imf",
                z_batch=x1,
                t=t,
                gen=self.bridge_gen,
                x0_batch=x0,
                sigma=self.sigma,
                direction=phase,
                memory_format=self.memory_format,
            )

            opt.zero_grad(set_to_none=True)

            if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()

            with torch.autocast(
                device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
            ):
                # pred = model(x_t, t_net)
                pred = model(x_t, t)
                loss = ConditionalVectorField.compute_loss(
                    "imf", pred, target, t_net, sigma=self.sigma
                )

            if self.use_amp:
                # Scale the loss and backward pass
                scaler.scale(loss).backward()

                # UNSCALE BEFORE CLIPPING
                scaler.unscale_(opt)
            else:
                loss.backward()

            # --- BaseTrainer Metric Hook (pre-clip grad norm) ---
            self.log_inner_step(model, loss, opt, phase=log_phase, ipf_iter=imf_iter)

            # Clip the unscaled gradients
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)

            if self.use_amp:
                # Step optimizer through the scaler (skips step if inf/nan gradients are found)
                scaler.step(opt)

                # Update the scaler for the next iteration
                scaler.update()
            else:
                opt.step()

            if self.use_ema:
                # EMA tracks the live weights after every optimizer step
                ema_helper.update(model)

            # We cast to .float() in case AMP is using float16, to prevent overflow
            # when accumulating thousands of steps.
            total_loss += loss.detach().float()

        for g, lr0 in zip(opt.param_groups, base_lrs):  # restore for the next phase
            g["lr"] = lr0

        return (total_loss / num_iter).item()

    def fit(
        self,
        imf_iterations: int,
        inner_iterations: int = 5000,
        save_per: Union[int, None] = None,
        save_path: Union[str, None] = None,
        eval_per: Optional[int] = None,
        eval_callback: Optional[Callable] = None,
    ):
        # Models to evaluate/save: the EMA copies if enabled, else the live base models
        f_eval = self.f_ema.model if self.use_ema else self.f_model_base
        b_eval = self.b_ema.model if self.use_ema else self.b_model_base

        for n in range(1, imf_iterations + 1):
            IMFTrainer.logger.debug(f"\n--- IMF Iteration {n}/{imf_iterations} ---")

            # ==========================================
            # Phase 1: Train Backward Model (prior -> data)
            # ==========================================
            nfes_before, phase_start = self.total_nfes, time.time()
            b_loss = self._train_phase(
                self.b_model,
                self.b_opt,
                self.b_ema,
                lambda: self._make_pairs("b", n),
                "b",
                inner_iterations,
                n,
            )

            b_metrics = {
                "loss": b_loss,
                "phase_time_sec": time.time() - phase_start,
                # every pair-cache build (incl. refreshes) is counted in _simulate
                "train_nfes": self.total_nfes - nfes_before,
            }
            # Run evaluation condition
            _run_eval = (
                eval_callback is not None
                and eval_per is not None
                and (n % eval_per == 0)
            )
            if _run_eval:
                with self._isolated_global_rng(self.seed + SeedOffsets.IMF_EVAL):
                    b_metrics.update(eval_callback(b_eval, direction="b"))

            self.log_phase_end("backward", n, b_metrics)

            # Dynamically format the log string based on whether evaluation ran
            qual_str = self._format_eval_str(b_metrics, _run_eval, prefix="B-")
            self.logger.info(
                f"Iteration {n} (Backward) | B-Loss: {b_loss:.4f} | {qual_str}"
            )

            # ==========================================
            # Phase 2: Train Forward Model (data -> prior)
            # ==========================================
            nfes_before, phase_start = self.total_nfes, time.time()
            f_loss = self._train_phase(
                self.f_model,
                self.f_opt,
                self.f_ema,
                lambda: self._make_pairs("f", n),
                "f",
                inner_iterations,
                n,
            )
            f_metrics = {
                "loss": f_loss,
                "phase_time_sec": time.time() - phase_start,
                "train_nfes": self.total_nfes - nfes_before,
            }
            if _run_eval:
                with self._isolated_global_rng(self.seed + SeedOffsets.IMF_EVAL):
                    f_metrics.update(eval_callback(f_eval, direction="f"))

            self.log_phase_end("forward", n, f_metrics)

            # Dynamically format the log string based on whether evaluation ran
            qual_str = self._format_eval_str(f_metrics, _run_eval, prefix="F-")
            self.logger.info(
                f"Iteration {n} (Forward) | F-Loss: {f_loss:.4f} | {qual_str}"
            )

            # --- Outer-Loop Diagnostics (same hooks as IPF) ---
            self.track_parameter_drift(self.b_model, ipf_iter=n)
            self.track_path_consistency(self.f_model, self.b_model, ipf_iter=n)

            # --- Intermediate Checkpoint Saving (EMA weights) ---
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

        # Log hardware footprint of the base models
        self.log_compute_cost([self.f_model_base, self.b_model_base])

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
