from src.training.base_trainer import BaseTrainer
from src.models.ema import EMAHelper
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
        super().__init__(device=device, metric_logger=metric_logger)

        if first_coupling not in ("ind", "ref"):
            raise ValueError(
                f"first_coupling must be 'ind' or 'ref', got {first_coupling!r}"
            )

        self.dataset = dataset
        self.f_opt = forward_opt
        self.b_opt = backward_opt
        self.seed = seed
        self.device = device
        self.batch_size = batch_size
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
        self.data_shape = tuple(self.dataset[0].shape)
        self.use_ema = use_ema

        # -------------------------------------------------------------
        # 0. Private RNG streams
        # -------------------------------------------------------------
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

        # Cache-staleness probe: re-seeded at the start of every probe, so the probe before
        # and after a refresh sees the exact same (index, t, z) draws.
        self.probe_gen = torch.Generator(device=self.device)

        # -------------------------------------------------------------
        # 1. Global channels_last & Hardware Settings
        # -------------------------------------------------------------
        is_image = len(self.data_shape) == 3  # (C, H, W)
        self.memory_format = (
            torch.channels_last
            if (self.device.type == "cuda" and is_image)
            else torch.contiguous_format
        )

        # if self.device.type == "cuda" and is_image:
        #     torch.backends.cudnn.benchmark = True

        # -------------------------------------------------------------
        # 2. Precision & Compilation Settings
        # ------------------------------------------------------------
        self.use_amp = use_amp and (self.device.type == "cuda")
        # One scaler per network: the two nets have different gradient statistics,
        # so they should not share a loss-scale that adapts to the other's overflows.
        self.f_scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.b_scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.amp_dtype = torch.float16 if self.device.type == "cuda" else torch.bfloat16

        self.use_compile = self.device.type == "cuda"
        if self.use_compile:
            self._fused_bridge_calc = torch.compile(self._bridge_kernel, mode="default")
        else:
            self._fused_bridge_calc = self._bridge_kernel

        # -------------------------------------------------------------
        # 3. Data Pipeline & Probe Batch
        # -------------------------------------------------------------
        self.dl = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            generator=self.shuffle_gen,
        )
        self._data_iter = self._repeater(self.dl)
        self.fixed_probe_batch = self._make_probe_batch()

        # -------------------------------------------------------------
        # 4. Models, EMA, and Compilation
        # -------------------------------------------------------------
        # Keep base models UNCOMPILED for EMA tracking and checkpoint saving
        self.f_model_base = forward_model.to(device, memory_format=self.memory_format)
        self.b_model_base = backward_model.to(device, memory_format=self.memory_format)

        # generate() reads this, and we set it BEFORE copying so samplers get it too
        self.f_model_base.sigma = sigma
        self.b_model_base.sigma = sigma

        if self.use_ema:
            # Register EMA on the base models
            self.ema_f = EMAHelper(mu=ema_mu, device=self.device)
            self.ema_f.register(self.f_model_base)
            self.ema_b = EMAHelper(mu=ema_mu, device=self.device)
            self.ema_b.register(self.b_model_base)
        else:
            self.ema_f, self.ema_b = None, None

        # Create persistent samplers for cache generation
        self.f_sampler = copy.deepcopy(self.f_model_base)
        self.b_sampler = copy.deepcopy(self.b_model_base)

        if self.use_compile:
            # "reduce-overhead" uses CUDA graphs, which gives a massive speedup
            # for the SDE simulation loop.
            self.f_model = torch.compile(self.f_model_base, mode="reduce-overhead")
            self.b_model = torch.compile(self.b_model_base, mode="reduce-overhead")
            self.f_sampler = torch.compile(self.f_sampler, mode="reduce-overhead")
            self.b_sampler = torch.compile(self.b_sampler, mode="reduce-overhead")
        else:
            self.f_model = self.f_model_base
            self.b_model = self.b_model_base

    ############ RNG helpers

    @contextmanager
    def _isolated_global_rng(self, seed: int):
        """
        Seed the global CPU/CUDA RNGs for the duration of the block and restore the
        previous state afterwards. Used around eval_callback, whose Evaluator and
        model.generate() draw from the global generators: it now sees the same noise
        every time it runs, and can never shift anybody else's random stream.
        """
        devices = []
        if self.device.type == "cuda":
            idx = (
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device()
            )
            devices = [idx]

        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            yield

    def _randn_like(self, ref: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        # torch.randn_like has no `generator` argument, so draw explicitly and put the
        # result in the memory format randn_like would have preserved.
        return torch.randn(
            ref.shape, device=ref.device, dtype=ref.dtype, generator=gen
        ).contiguous(memory_format=self.memory_format)

    ############ UTILS

    @staticmethod
    def _repeater(dataloader):
        """Infinite generator to continuously yield batches."""
        while True:
            for batch in dataloader:
                yield batch

    def _next_data(self) -> torch.Tensor:
        batch = next(self._data_iter)
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        return batch.to(
            self.device, memory_format=self.memory_format, non_blocking=True
        )

    def _make_probe_batch(self) -> torch.Tensor:
        """
        Fixed batch for parameter-drift tracking. Taken straight from the dataset
        (first `batch_size` items) instead of iterating a DataLoader: creating a
        DataLoader iterator draws a base seed from the loader's generator (or from
        the global one), which would shift the shuffle stream.
        """
        n = min(self.batch_size, len(self.dataset))
        items = [self.dataset[i] for i in range(n)]
        items = [it[0] if isinstance(it, (list, tuple)) else it for it in items]
        return torch.stack(items).to(self.device, memory_format=self.memory_format)

    ############ COUPLING GENERATION

    @torch.no_grad()
    def _simulate(self, sampler: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """
        Euler-Maruyama simulation of `sampler`'s own chain (own time t_k = k h), starting at x.
        Noise is injected at EVERY step (as in the official DSBM code when producing couplings);
        dropping the last noise is only correct when producing final samples (see generate()).
        """
        for i in range(self.sde_steps):
            t = torch.full((x.shape[0], 1), i * self.h, device=self.device)

            if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()

            with torch.autocast(
                device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
            ):
                drift = sampler(x, t)

            x = (
                x
                + self.h * drift
                + self.sigma * math.sqrt(self.h) * torch.randn_like(x)
            )
        self.total_nfes += x.shape[0] * self.sde_steps
        return x

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
                (self.f_model_base, self.ema_f, self.f_sampler)
                if phase == "b"
                else (self.b_model_base, self.ema_b, self.b_sampler)
            )

            # Simulate with the EMA weights if enabled, else the live weights
            if src_ema is not None:
                src_ema.copy_to(sampler)
            else:
                getattr(sampler, "_orig_mod", sampler).load_state_dict(
                    src_base.state_dict()
                )
            sampler.eval()

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
                x1 = self._simulate(sampler, x0)
            else:
                x1 = torch.randn(
                    self.batch_size,
                    *self.data_shape,
                    device=self.device,
                    generator=self.noise_gen,
                ).contiguous(memory_format=self.memory_format)
                x0 = self._simulate(sampler, x1)

            # Write directly into the pre-allocated tensors
            b_size = x0.shape[0]
            X0_cache[idx : idx + b_size] = x0
            X1_cache[idx : idx + b_size] = x1
            idx += b_size

        # Return exact slice in case the final batch was smaller (if drop_last=False was ever used)
        return X0_cache[:idx], X1_cache[:idx]

    ############ REGRESSION TARGETS
    @staticmethod
    def _bridge_kernel(
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        z: torch.Tensor,
        sigma: float,
        is_forward: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fused kernel computing Brownian-bridge sample and regression targets.
        Inductor will fuse all arithmetic into a single Triton pass over GPU VRAM.
        """
        te = t.view(x0.shape[0], *([1] * (x0.ndim - 1)))
        x_t = (1.0 - te) * x0 + te * x1 + sigma * torch.sqrt(te * (1.0 - te)) * z

        if is_forward:
            target = (x1 - x0) - sigma * torch.sqrt(te / (1.0 - te)) * z
            t_in = t
            weight = 1.0 / (1.0 + (sigma**2 * te) / (1.0 - te))
        else:
            target = -(x1 - x0) - sigma * torch.sqrt((1.0 - te) / te) * z
            t_in = 1.0 - t
            weight = 1.0 / (1.0 + (sigma**2 * (1.0 - te)) / te)

        return x_t, t_in, target, weight

    def _bridge_batch(
        self, x0: torch.Tensor, x1: torch.Tensor, phase: str, gen: torch.Generator
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Draws uniform time and noise, evaluating targets via the fused kernel."""
        B = x0.shape[0]
        t = (
            torch.rand(B, 1, device=self.device, generator=gen) * (1.0 - 2.0 * self.eps)
            + self.eps
        )
        z = self._randn_like(x0, gen)

        return self._fused_bridge_calc(x0, x1, t, z, self.sigma, phase == "f")

    @torch.no_grad()
    def _probe_loss(
        self, model: nn.Module, X0: torch.Tensor, X1: torch.Tensor, phase: str
    ) -> float:
        """
        Loss of `model` on a fixed set of (index, t, noise) draws (used for the
        cache-staleness diagnostic). The generator is re-seeded on every call, so probing
        the old and the new pairs uses identical draws: the difference between the two
        numbers comes from the couplings alone.
        """
        was_training = model.training
        model.eval()

        gen = self.probe_gen
        gen.manual_seed(self.seed + SeedOffsets.IMF_PROBE_NOISE)

        idx = torch.randint(
            0, X0.shape[0], (self.batch_size,), device=self.device, generator=gen
        )

        x_t, t_in, target, weight = self._bridge_batch(X0[idx], X1[idx], phase, gen)

        if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()

        with torch.autocast(
            device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
        ):
            pred = model(x_t, t_in)
            raw_loss = torch.nn.functional.mse_loss(pred, target, reduction="none")
            loss = (raw_loss * weight).mean()

        model.train(was_training)

        return loss.item()

    ############ TRAINING LOOP

    def _train_phase(
        self,
        model: nn.Module,
        opt: torch.optim.Optimizer,
        ema_helper: EMAHelper,
        phase: str,
        num_iter: int,
        imf_iter: int,
    ) -> float:
        model.train()
        scaler = self.f_scaler if phase == "f" else self.b_scaler
        base_lrs = [g["lr"] for g in opt.param_groups]
        X0, X1 = self._make_pairs(phase, imf_iter)

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
                # Same model, old vs. new couplings: how much did the data distribution move?
                pre_loss = self._probe_loss(model, X0, X1, phase)
                X0, X1 = self._make_pairs(phase, imf_iter)
                post_loss = self._probe_loss(model, X0, X1, phase)
                self.track_cache_staleness(
                    pre_loss, post_loss, self.total_gradient_steps
                )

            idx = torch.randint(
                0,
                X0.shape[0],
                (self.batch_size,),
                device=self.device,
                generator=self.perm_gen,
            )

            x_t, t_in, target, weight = self._bridge_batch(
                X0[idx], X1[idx], phase, self.bridge_gen
            )

            opt.zero_grad(set_to_none=True)

            if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()

            with torch.autocast(
                device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
            ):
                pred = model(x_t, t_in)
                raw_loss = torch.nn.functional.mse_loss(pred, target, reduction="none")
                loss = (raw_loss * weight).mean()

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
        f_eval = self.ema_f.model if self.use_ema else self.f_model_base
        b_eval = self.ema_b.model if self.use_ema else self.b_model_base

        for n in range(1, imf_iterations + 1):
            IMFTrainer.logger.debug(f"\n--- IMF Iteration {n}/{imf_iterations} ---")

            # ==========================================
            # Phase 1: Train Backward Model (prior -> data)
            # ==========================================
            nfes_before, phase_start = self.total_nfes, time.time()
            b_loss = self._train_phase(
                self.b_model, self.b_opt, self.ema_b, "b", inner_iterations, n
            )
            b_metrics = {
                "loss": b_loss,
                "phase_time_sec": time.time() - phase_start,
                # every pair-cache build (incl. refreshes) is counted in _simulate
                "train_nfes": self.total_nfes - nfes_before,
            }
            # Run evaluation condition
            run_eval = False
            if eval_callback is not None:
                if eval_per is not None and (n % eval_per == 0):
                    run_eval = True

            if run_eval:
                with self._isolated_global_rng(self.seed + SeedOffsets.IMF_EVAL):
                    b_metrics.update(eval_callback(b_eval, direction="b"))

            self.log_phase_end("backward", n, b_metrics)

            # Dynamically format the log string based on whether evaluation ran
            if not run_eval:
                qual_str = ""
            elif "FID" in b_metrics:
                qual_str = (
                    f"B-FID: {b_metrics['FID']:.3f} | "
                    f"B-Prec: {b_metrics['Precision']:.3f} | "
                    f"B-Rec: {b_metrics['Recall']:.3f}"
                )
            else:
                qual_str = f"B-MMD: {b_metrics.get('eval_MMD', float('nan')):.4f}"

            self.logger.info(
                f"Iteration {n} (Backward) | B-Loss: {b_loss:.4f} | {qual_str}"
            )

            # ==========================================
            # Phase 2: Train Forward Model (data -> prior)
            # ==========================================
            nfes_before, phase_start = self.total_nfes, time.time()
            f_loss = self._train_phase(
                self.f_model, self.f_opt, self.ema_f, "f", inner_iterations, n
            )
            f_metrics = {
                "loss": f_loss,
                "phase_time_sec": time.time() - phase_start,
                "train_nfes": self.total_nfes - nfes_before,
            }
            if run_eval:
                with self._isolated_global_rng(self.seed + SeedOffsets.IMF_EVAL):
                    f_metrics.update(eval_callback(f_eval, direction="f"))

            self.log_phase_end("forward", n, f_metrics)

            # Dynamically format the log string based on whether evaluation ran
            if not run_eval:
                qual_str = ""
            else:
                qual_str = f"F-MMD: {f_metrics.get('eval_MMD', float('nan')):.4f}"

            self.logger.info(
                f"Iteration {n} (Forward) | F-Loss: {f_loss:.4f} | {qual_str}"
            )

            # --- Outer-Loop Diagnostics (same hooks as IPF) ---
            self.track_parameter_drift(self.b_model, ipf_iter=n)
            self.track_path_consistency(self.f_model, self.b_model, ipf_iter=n)

            # --- Intermediate Checkpoint Saving (EMA weights) ---
            if save_per and save_path and ((n + 1) % save_per == 0):
                save_model(
                    b_eval,
                    f"{save_path}/{self.b_model_base.__class__.__name__}_backward_checkpoint_{n+1}.pth",
                )

        self.logger.debug(("-" * 100))

        # Hand back the smoothed weights in the base models
        if self.use_ema:
            self.ema_f.copy_to(self.f_model_base)
            self.ema_b.copy_to(self.b_model_base)

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
