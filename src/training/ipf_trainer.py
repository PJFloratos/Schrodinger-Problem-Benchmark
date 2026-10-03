from src.training.base_trainer import BaseTrainer
from src.models.ema import EMAHelper
from src.utils import save_model, text_logger

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import math
import time
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
        batch_size: int = 256,
        sde_steps: int = 20,
        num_cache_batches: int = 10,  # Number of dataset batches to cache per iteration
        refresh_every: int = 500,  # regenerate the cache every N gradient steps
        grad_clip: float = 2.0,
        ema_mu: float = 0.999,
        lr_decay: bool = True,  # cosine decay inside each training phase
        lr_final_ratio: float = 0.05,  # final lr = ratio * base lr
        use_amp: bool = True,
    ):
        super().__init__(device=device, metric_logger=metric_logger)

        self.dataset = dataset
        self.f_opt = forward_opt
        self.b_opt = backward_opt
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

        # -------------------------------------------------------------
        # 1. Global channels_last & Hardware Settings
        # -------------------------------------------------------------
        is_image = len(self.data_shape) == 3  # (C, H, W)
        self.memory_format = (
            torch.channels_last
            if (self.device.type == "cuda" and is_image)
            else torch.contiguous_format
        )

        if self.device.type == "cuda" and is_image:
            torch.backends.cudnn.benchmark = True

        # -------------------------------------------------------------
        # 2. Precision & Compilation Settings
        # -------------------------------------------------------------
        self.use_amp = use_amp and (self.device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.amp_dtype = torch.float16 if self.device.type == "cuda" else torch.bfloat16

        # -------------------------------------------------------------
        # 3. Data Pipeline & Probe Batch
        # -------------------------------------------------------------
        self.dl = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )
        self._data_iter = self._repeater(self.dl)

        probe_batch = next(iter(self.dl))
        if isinstance(probe_batch, (list, tuple)):
            probe_batch = probe_batch[0]
        self.fixed_probe_batch = probe_batch.to(
            self.device, memory_format=self.memory_format
        )

        # -------------------------------------------------------------
        # 4. Models, EMA, and Compilation
        # -------------------------------------------------------------
        # Keep base models in self.memory_format
        self.f_model_base = forward_model.to(device, memory_format=self.memory_format)
        self.b_model_base = backward_model.to(device, memory_format=self.memory_format)

        # Initialize and register EMA trackers for both networks
        self.ema_f = EMAHelper(mu=ema_mu, device=self.device)
        self.ema_f.register(self.f_model_base)
        self.ema_b = EMAHelper(mu=ema_mu, device=self.device)
        self.ema_b.register(self.b_model_base)

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

    def _cache_iterator(self, X, T, U):
        """Fast infinite minibatch iterator over GPU tensors (no per-item DataLoader overhead)."""
        n = X.shape[0]
        while True:
            perm = torch.randperm(n, device=X.device)
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

        Noise is injected at EVERY step (as in the official repo with sample=False).
        Dropping the last noise is only correct when producing final samples.
        """
        use_reference = ipf_iteration == 0 and direction == "f"

        if not use_reference:
            temp_model = src_ema.ema_copy(source_model)
            if hasattr(sampler, "_orig_mod"):
                sampler._orig_mod.load_state_dict(temp_model.state_dict())
            else:
                sampler.load_state_dict(temp_model.state_dict())
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

        # Pre-allocate contiguous memory blocks (massive speedup)
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
                    self.batch_size, *self.data_shape, device=self.device
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
                z = torch.randn_like(x)
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

            if it > 0 and self.refresh_every and it % self.refresh_every == 0:
                cache_iter = make_cache()  # fresh trajectories, old cache is freed

                # Check for cache staleness by pulling immediate next batch
                x_batch, t_batch, u_batch = next(cache_iter)
                opt.zero_grad(set_to_none=True)

                if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                    torch.compiler.cudagraph_mark_step_begin()

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.use_amp,
                ):
                    pred_u = target_model(x_batch, t_batch)
                    fresh_loss = self.criterion(pred_u, u_batch)

                if prev_loss_val is not None:
                    self.track_cache_staleness(
                        prev_loss_val, fresh_loss.item(), self.total_gradient_steps
                    )
                loss = self.h * fresh_loss
            else:
                x_batch, t_batch, u_batch = next(cache_iter)
                opt.zero_grad(set_to_none=True)

                if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                    torch.compiler.cudagraph_mark_step_begin()

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.use_amp,
                ):
                    pred_u = target_model(x_batch, t_batch)
                    loss = self.h * self.criterion(pred_u, u_batch)

            # Backpropagation under AMP
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(opt)

            # --- BaseTrainer Metric Hook ---
            self.log_inner_step(target_model, loss, opt, phase=phase, ipf_iter=ipf_iter)

            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    target_model.parameters(), self.grad_clip
                )

            self.scaler.step(opt)
            self.scaler.update()

            # Update EMA shadow weights immediately after every gradient step
            actual_model = getattr(target_model, "_orig_mod", target_model)
            ema_helper.update(actual_model)

            total_loss += loss.detach().float()

            if it % self.refresh_every == 0 or prev_loss_val is None:
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
        eval_callback: Optional[callable] = None,
    ):
        for n in range(ipf_iterations):
            IPFTrainer.logger.debug(f"\n--- IPF Iteration {n+1}/{ipf_iterations} ---")

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
            if eval_callback:
                # Trigger the evaluator purely as a callback
                b_metrics.update(eval_callback(self.b_model, direction="b"))

            self.log_phase_end("backward", n, b_metrics)

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
            if eval_callback:
                f_metrics.update(eval_callback(self.f_model, direction="f"))

            self.log_phase_end("forward", n, f_metrics)

            self.logger.info(
                f"Iteration {n+1} | "
                f"F-Loss: {f_loss:.4f} ({f_metrics.get('eval_MMD', 0):.6f} MMD) | "
                f"B-Loss: {b_loss:.4f} ({b_metrics.get('FID', b_metrics.get('eval_MMD', 0)):.6f} Dist)"
            )

            # --- Outer-Loop Diagnostics ---
            self.track_parameter_drift(self.b_model, ipf_iter=n)
            self.track_path_consistency(self.f_model, self.b_model, ipf_iter=n)

            # --- Intermediate Checkpoint Saving ---
            if save_per and save_path and ((n + 1) % save_per == 0):
                # Generate a temporary copy with EMA weights applied for an accurate checkpoint
                temp_b_model = self.ema_b.ema_copy(self.b_model)
                save_model(
                    temp_b_model,
                    f"{save_path}/{self.b_model.__class__.__name__}_backward_checkpoint_{n+1}.pth",
                )

        # Load smoothed weights into the models used for evaluation
        self.ema_f.ema(self.f_model_base)
        self.ema_b.ema(self.b_model_base)

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


"""
Opt:
B - 1:23
F - 1:22

Unopt:
B - 4:14
F - 4:14
"""
