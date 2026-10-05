from src.training.base_trainer import BaseTrainer
from src.models.ema import EMAHelper
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
from typing import Union, Any, Optional


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
        grad_clip: float = 2.0,
        ema_mu: float = 0.999,
        lr_decay: bool = True,  # cosine decay inside each training phase
        lr_final_ratio: float = 0.05,  # final lr = ratio * base lr
        use_amp: bool = True,
        use_ema: bool = True,
    ):
        super().__init__(device=device, metric_logger=metric_logger)

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
        self.grad_clip = grad_clip
        self.lr_decay = lr_decay
        self.lr_final_ratio = lr_final_ratio
        self.criterion = nn.MSELoss()
        self.data_shape = tuple(self.dataset[0].shape)
        self.use_ema = use_ema

        # -------------------------------------------------------------
        # 0. Private RNG streams
        # -------------------------------------------------------------
        # DataLoader shuffling (must be a CPU generator).
        self.shuffle_gen = torch.Generator()
        self.shuffle_gen.manual_seed(self.seed + SeedOffsets.IPF_SHUFFLE)

        # Cache-simulation noise: one continuous stream, touched by nothing else.
        self.noise_gen = torch.Generator(device=self.device)
        self.noise_gen.manual_seed(self.seed + SeedOffsets.IPF_CACHE_NOISE)

        # Minibatch order inside each cache.
        self.perm_gen = torch.Generator(device=self.device)
        self.perm_gen.manual_seed(self.seed + SeedOffsets.IPF_CACHE_PERM)

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
        # -------------------------------------------------------------
        self.use_amp = use_amp and (self.device.type == "cuda")
        # One scaler per network: the two nets have different gradient statistics,
        # so they should not share a loss-scale that adapts to the other's overflows.
        self.f_scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.b_scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.amp_dtype = torch.float16 if self.device.type == "cuda" else torch.bfloat16

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
        # Keep base models in self.memory_format
        self.f_model_base = forward_model.to(device, memory_format=self.memory_format)
        self.b_model_base = backward_model.to(device, memory_format=self.memory_format)

        if self.use_ema:
            # Initialize and register EMA trackers for both networks
            self.ema_f = EMAHelper(mu=ema_mu, device=self.device)
            self.ema_f.register(self.f_model_base)
            self.ema_b = EMAHelper(mu=ema_mu, device=self.device)
            self.ema_b.register(self.b_model_base)
        else:
            self.ema_f, self.ema_b = None, None

        # Persistent samplers for cache generation
        self.f_sampler = copy.deepcopy(self.f_model_base)
        self.b_sampler = copy.deepcopy(self.b_model_base)

        self.use_compile = self.device.type == "cuda"
        if self.use_compile:
            self.f_model = torch.compile(self.f_model_base, mode="reduce-overhead")
            self.b_model = torch.compile(self.b_model_base, mode="reduce-overhead")
            self.f_sampler = torch.compile(self.f_sampler, mode="reduce-overhead")
            self.b_sampler = torch.compile(self.b_sampler, mode="reduce-overhead")
        else:
            self.f_model = self.f_model_base
            self.b_model = self.b_model_base

    ############ Reference Process

    @staticmethod
    def reference_drift(x: torch.Tensor) -> torch.Tensor:
        """Standard Brownian Motion reference process (zero drift)."""
        return torch.zeros_like(x)

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
        """Infinite generator to continuously yield batches (mimics DSB repeater)."""
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

    def _cache_iterator(self, X, T, U):
        """Fast infinite minibatch iterator over GPU tensors (no per-item DataLoader overhead)."""
        n = X.shape[0]
        while True:
            perm = torch.randperm(n, device=X.device, generator=self.perm_gen)
            for j in range(0, n - self.batch_size + 1, self.batch_size):
                idx = perm[j : j + self.batch_size]
                yield X[idx], T[idx], U[idx]

    ############ CACHE GENERATRION

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

        def get_drift(x, t):
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

                drift = get_drift(x, t_now)
                z = self._randn_like(x, self.noise_gen)
                x_next = x + self.h * drift + math.sqrt(self.h) * z

                drift_next = get_drift(x_next, t_now)
                target = -drift_next - z / math.sqrt(self.h)

                # The trained network is queried at x_{k+1}, at the forward time of x_{k+1}.
                X_cache[idx : idx + b_size] = x_next
                T_cache[idx : idx + b_size] = t_next
                U_cache[idx : idx + b_size] = target
                x = x_next
                idx += b_size

        return self._cache_iterator(X_cache, T_cache, U_cache)

    ############ TRAINING

    def _train_cache(
        self,
        target_model: nn.Module,
        opt: torch.optim.Optimizer,
        ema_helper: EMAHelper,
        make_cache,
        num_iter: int,
        direction: str,
        ipf_iter: int,
    ) -> float:
        target_model.train()
        scaler = self.b_scaler if direction == "b" else self.f_scaler

        cache_iter = make_cache()
        total_loss = torch.tensor(0.0, device=self.device)
        base_lrs = [g["lr"] for g in opt.param_groups]
        prev_loss_val = None

        if direction == "f":
            phase = "forward"
            desc = "Training Forward Model"
        else:
            phase = "backward"
            desc = "Training Backward Model"

        for it in tqdm(range(num_iter), ascii=True, desc=desc):
            if self.lr_decay:
                r = self.lr_final_ratio
                scale = r + (1 - r) * 0.5 * (1 + math.cos(math.pi * it / num_iter))
                for g, lr0 in zip(opt.param_groups, base_lrs):
                    g["lr"] = lr0 * scale

            refreshing = (
                bool(self.refresh_every) and it > 0 and it % self.refresh_every == 0
            )

            if refreshing:
                cache_iter = make_cache()  # fresh trajectories, old cache is freed

            x_batch, t_batch, u_batch = next(cache_iter)
            opt.zero_grad(set_to_none=True)

            if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()

            with torch.autocast(
                device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
            ):
                pred_u = target_model(x_batch, t_batch)
                mse = self.criterion(pred_u, u_batch)

            # On a refresh step this batch comes from the brand-new cache.
            if refreshing and prev_loss_val is not None:
                self.track_cache_staleness(
                    prev_loss_val, mse.item(), self.total_gradient_steps
                )

            loss = self.h * mse

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
        f_eval = self.ema_f.model if self.use_ema else self.f_model_base
        b_eval = self.ema_b.model if self.use_ema else self.b_model_base

        for n in range(1, ipf_iterations + 1):
            IPFTrainer.logger.debug(f"\n--- IPF Iteration {n}/{ipf_iterations} ---")

            # ==========================================
            # Phase 1: Train Backward Model
            # ==========================================
            phase_start = time.time()
            b_loss = self._train_cache(
                self.b_model,
                self.b_opt,
                self.ema_b,
                lambda: self._simulate_and_cache(
                    self.f_model_base, self.ema_f, self.f_sampler, "f", n
                ),
                inner_iterations,
                direction="b",
                ipf_iter=n,
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
            run_eval = False
            if eval_callback is not None:
                if eval_per is not None and (n % eval_per == 0):
                    run_eval = True

            if run_eval:
                with self._isolated_global_rng(self.seed + SeedOffsets.IPF_EVAL):
                    # Trigger the evaluator purely as a callback
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
            # Phase 2: Train Forward Model
            # ==========================================
            phase_start = time.time()
            f_loss = self._train_cache(
                self.f_model,
                self.f_opt,
                self.ema_f,
                lambda: self._simulate_and_cache(
                    self.b_model_base, self.ema_b, self.b_sampler, "b", n
                ),
                inner_iterations,
                direction="f",
                ipf_iter=n,
            )
            f_time = time.time() - phase_start
            self.total_nfes += train_nfes

            # Aggregate Phase 2 Metrics
            f_metrics = {
                "loss": f_loss,
                "phase_time_sec": f_time,
                "train_nfes": train_nfes,
            }
            if run_eval:
                with self._isolated_global_rng(self.seed + SeedOffsets.IPF_EVAL):
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

            # --- Outer-Loop Diagnostics ---
            self.track_parameter_drift(self.b_model, ipf_iter=n)
            self.track_path_consistency(self.f_model, self.b_model, ipf_iter=n)

            # --- Intermediate Checkpoint Saving ---
            if save_per and save_path and (n % save_per == 0):
                save_model(
                    b_eval,
                    f"{save_path}/{self.b_model_base.__class__.__name__}_backward_checkpoint_{n}.pth",
                )

        self.logger.debug(("-" * 100))

        # Hand back the smoothed weights in the base models
        if self.use_ema:
            self.ema_f.copy_to(self.f_model_base)
            self.ema_b.copy_to(self.b_model_base)

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
