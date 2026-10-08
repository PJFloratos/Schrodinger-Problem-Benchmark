from src.core.dynamics import ConditionalVectorField
from src.models.components import EMAHelper
from src.utils.seed import SeedOffsets

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import time
import copy
from contextlib import contextmanager
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple


class BaseTrainer(ABC):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        seed: int,
        device: torch.device,
        metric_logger: Any,  # e.g., WandB or TensorBoard writer
        use_amp: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.device = device
        self.metric_logger = metric_logger

        # 1. Global channels_last & Hardware Settings
        # Sniff dataset shape safely (handles tuples/lists returned by Dataset)
        sample = self.dataset[0]
        if isinstance(sample, (list, tuple)):
            sample = sample[0]

        self.data_shape = tuple(sample.shape)
        is_image = len(self.data_shape) == 3
        self.memory_format = (
            torch.channels_last
            if (self.device.type == "cuda" and is_image)
            else torch.contiguous_format
        )

        if self.device.type == "cuda" and is_image:
            torch.backends.cudnn.benchmark = True

        # 2. Precision Settings
        self.use_amp = use_amp and (self.device.type == "cuda")
        self.amp_dtype = torch.float16 if self.device.type == "cuda" else torch.bfloat16

        # Compute Cost: Hardware Footprint & Timing
        self.start_time = time.time()
        self.total_gradient_steps = 0
        self.total_nfes = 0

        # State tracking for Outer-Loop Convergence
        self.previous_model_outputs: Optional[torch.Tensor] = None
        self.fixed_probe_batch = self._make_probe_batch()

        # 3. Cache staleness helpers
        self.probe_gen = torch.Generator(device=self.device)
        self._probe_tensors: Optional[Tuple[torch.Tensor, ...]] = None

    # ------------------------------------------------------------------
    # Shared Model Factory & Cache Dataloader
    # ------------------------------------------------------------------
    def _setup_model(
        self,
        model: nn.Module,
        ema_mu: float,
        use_ema: bool,
        sigma: Optional[float] = None,
        create_sampler: bool = False,
    ) -> Tuple[nn.Module, nn.Module, Optional[EMAHelper], Optional[nn.Module]]:
        """
        Standardizes model preparation: device placement, sigma injection,
        EMA registration, sampler cloning, and torch.compile wrapping.
        """
        # 1. Place Base Model
        model_base = model.to(self.device, memory_format=self.memory_format)

        if sigma is not None:
            model_base.sigma = sigma

        # 2. Setup EMA
        if use_ema:
            ema_helper = EMAHelper(mu=ema_mu, device=self.device)
            ema_helper.register(model_base)
        else:
            ema_helper = None

        # 3. Clone persistent Sampler (for cache generation in dual-model setups)
        sampler_base = copy.deepcopy(model_base) if create_sampler else None

        # 4. Compilation
        if self.device.type == "cuda":
            model_compiled = torch.compile(model_base, mode="reduce-overhead")
            sampler_compiled = (
                torch.compile(sampler_base, mode="reduce-overhead")
                if create_sampler
                else None
            )
        else:
            model_compiled = model_base
            sampler_compiled = sampler_base

        return model_base, model_compiled, ema_helper, sampler_compiled

    def _init_cache_dataloader(self, shuffle_offset: int) -> None:
        """Initializes the infinite data stream used by cache-based trainers (IPF, IMF, SF2M)."""
        shuffle_gen = torch.Generator().manual_seed(self.seed + shuffle_offset)
        self.dl = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            generator=shuffle_gen,
        )
        self._data_iter = self._repeater(self.dl)

    # ------------------------------------------------------------------
    # Shared Utils
    # ------------------------------------------------------------------
    def _format_eval_str(
        self, metrics: Dict[str, Any], run_eval: bool, prefix: str = ""
    ) -> str:
        """
        Formats evaluation metrics into a standardized string for logging.
        Returns an empty string if no evaluation was run, preventing trailing separators.
        """
        if not run_eval:
            return ""

        if "FID" in metrics:
            return (
                f" {prefix}FID: {metrics['FID']:.3f}"
                f" | {prefix}Prec: {metrics['Precision']:.3f}"
                f" | {prefix}Rec: {metrics['Recall']:.3f}"
            )
        elif "eval_MMD" in metrics:
            return f" {prefix}MMD: {metrics['eval_MMD']:.6f}"

        return ""

    @contextmanager
    def _isolated_global_rng(self, seed: int):
        """
        Seed the global CPU/CUDA RNGs for the duration of the block and restore the
        previous state afterwards. Used around eval_callback, whose Evaluator and
        model.generate() draw from the global generators.
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
        """Draws explicitly and puts result in the memory format randn_like would preserve."""
        return torch.randn(
            ref.shape, device=ref.device, dtype=ref.dtype, generator=gen
        ).contiguous(memory_format=self.memory_format)

    def _make_probe_batch(self) -> torch.Tensor:
        """Fixed batch for parameter-drift tracking, sliced securely from the dataset."""
        n = min(self.batch_size, len(self.dataset))
        items = [self.dataset[i] for i in range(n)]
        items = [it[0] if isinstance(it, (list, tuple)) else it for it in items]
        return torch.stack(items).to(self.device, memory_format=self.memory_format)

    @torch.no_grad()
    def _probe_loss(
        self, model: nn.Module, model_type: str, phase: str, *probe_tensors
    ) -> float:
        """
        Loss of `model` on a fixed set of cached draws (used for the cache-staleness diagnostic).
        The generato_probe_lossr is re-seeded on every call, ensuring apples-to-apples drift comparisons.
        """
        if self._probe_tensors is None:
            return 0.0

        was_training = model.training
        model.eval()

        gen = self.probe_gen
        # Use a unified offset
        gen.manual_seed(self.seed + SeedOffsets.IMF_PROBE_NOISE)

        # Draw identical batch indices before and after refresh
        idx = torch.randint(
            0,
            self._probe_tensors[0].shape[0],
            (self.batch_size,),
            device=self.device,
            generator=gen,
        )

        eps = getattr(self, "eps", 1e-4)
        t = (
            torch.rand(self.batch_size, 1, device=self.device, generator=gen)
            * (1.0 - 2.0 * eps)
            + eps
        )

        # Data Unpacking (Differs based on cache structure)
        if model_type == "imf":
            # IMF unpacks Endpoints (X0, X1) and dynamically builds the bridge
            x0, x1 = self._probe_tensors[0][idx], self._probe_tensors[1][idx]
            x_t, target, t_net = ConditionalVectorField.get_interpolant_and_target(
                model_type="imf",
                z_batch=x1,
                t=t,
                gen=gen,
                x0_batch=x0,
                sigma=getattr(self, "sigma", 1.0),
                direction=phase,
                memory_format=self.memory_format,
            )
        elif model_type == "ipf":
            # IPF unpacks the pre-computed Trajectory (X, T, U) directly
            x_t = self._probe_tensors[0][idx]
            t_net = self._probe_tensors[1][idx]
            target = self._probe_tensors[2][idx]
        else:
            raise ValueError(f"Probe loss not implemented for model type: {model_type}")

        # Loss Computation (Shared)
        if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()

        with torch.autocast(
            device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
        ):
            # pred = model(x_t, t_net)
            pred = model(x_t, t)
            loss = ConditionalVectorField.compute_loss(
                model_type,
                pred,
                target,
                t_net,
                sigma=getattr(self, "sigma", 1.0),
                h=getattr(self, "h", None),
            )

        model.train(was_training)
        return loss.item()

    @staticmethod
    def _repeater(dataloader):
        """Infinite generator to continuously yield batches."""
        while True:
            for batch in dataloader:
                yield batch

    def _next_data(self) -> torch.Tensor:
        """Grabs next batch from an instantiated self._data_iter safely."""
        batch = next(self._data_iter)
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        return batch.to(
            self.device, memory_format=self.memory_format, non_blocking=True
        )

    def _cache_iterator(self, *tensors: torch.Tensor):
        """Generic fast infinite minibatch iterator over multiple parallel GPU tensors."""
        n = tensors[0].shape[0]
        while True:
            perm = torch.randperm(n, device=self.device, generator=self.perm_gen)
            for j in range(0, n - self.batch_size + 1, self.batch_size):
                idx = perm[j : j + self.batch_size]
                yield tuple(t[idx] for t in tensors)

    # ------------------------------------------------------------------
    # Logging & Diagnostics
    # ------------------------------------------------------------------

    def log_inner_step(
        self,
        model: nn.Module,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        phase: str,
        ipf_iter: int,
        log_freq: int = 50,
    ) -> None:
        """Inner-Phase Diagnostics."""
        # 1. Skip expensive syncs and logging for most steps
        if self.total_gradient_steps % log_freq != 0:
            self.total_gradient_steps += 1
            return

        # 2. Compute grad norm entirely on the GPU
        grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]
        if grads:
            # Stack all norms on GPU, compute total norm, then do ONE .item() sync
            grad_norm = (
                torch.stack([torch.norm(g, p=2) for g in grads]).norm(p=2).item()
            )
        else:
            grad_norm = 0.0

        current_lr = optimizer.param_groups[0].get("lr", 0.0)

        if hasattr(self.metric_logger, "log"):
            self.metric_logger.log(
                {
                    f"{phase}/inner_loss": loss.item(),  # ONE sync every 50 steps.
                    f"{phase}/grad_norm_pre_clip": grad_norm,
                    f"{phase}/effective_lr": current_lr,
                    "global_step": self.total_gradient_steps,
                    "ipf_iteration": ipf_iter,
                }
            )
        self.total_gradient_steps += 1

    def track_cache_staleness(
        self, pre_refresh_loss: float, post_refresh_loss: float, step_idx: int
    ):
        """Detects model drift at cache boundaries."""
        if hasattr(self.metric_logger, "log"):
            self.metric_logger.log(
                {
                    "diagnostics/cache_staleness_jump": post_refresh_loss
                    - pre_refresh_loss,
                    "global_step": step_idx,
                }
            )

    def log_phase_end(self, phase: str, ipf_iter: int, metrics: Dict[str, float]):
        """Logs aggregated phase metrics (Loss, MMD, Time, NFEs) received from fit()."""
        logged_metrics = {}
        for k, v in metrics.items():
            # Safely cast any rogue PyTorch tensors to standard Python numbers
            if isinstance(v, torch.Tensor):
                logged_metrics[f"{phase}/{k}"] = v.item()
            else:
                logged_metrics[f"{phase}/{k}"] = v

        logged_metrics["ipf_iteration"] = ipf_iter

        if hasattr(self.metric_logger, "log"):
            self.metric_logger.log(logged_metrics)

    @torch.no_grad()
    def track_parameter_drift(self, model: nn.Module, ipf_iter: int):
        """Section 2: Evaluates ||f_model_n - f_model_{n-1}||."""
        if self.fixed_probe_batch is None:
            return

        t_probe = torch.full(
            (self.fixed_probe_batch.size(0), 1), 0.5, device=self.device
        )
        current_outputs = model(self.fixed_probe_batch, t_probe)

        if self.previous_model_outputs is not None:
            drift = torch.norm(
                current_outputs - self.previous_model_outputs, p=2
            ).item()
            if hasattr(self.metric_logger, "log"):
                self.metric_logger.log(
                    {"outer_loop/parameter_drift": drift, "ipf_iteration": ipf_iter}
                )

        self.previous_model_outputs = current_outputs.clone()

    @torch.no_grad()
    def track_path_consistency(
        self,
        f_model: nn.Module,
        b_model: nn.Module,
        ipf_iter: int,
        t_points: tuple = (0.25, 0.5, 0.75),
    ):
        """Evaluates forward/backward vector field alignment across t in [0, 1]."""
        if self.fixed_probe_batch is None:
            return

        f_model.eval()
        b_model.eval()
        device = self.device

        total_mse = 0.0
        for t_val in t_points:
            t_tensor = torch.full(
                (self.fixed_probe_batch.size(0), 1), t_val, device=device
            )
            f_vec = f_model(self.fixed_probe_batch, t_tensor)
            b_vec = b_model(self.fixed_probe_batch, t_tensor)

            # Measure field alignment consistency
            total_mse += torch.mean((f_vec + b_vec) ** 2).item()

        consistency_mse = total_mse / len(t_points)

        if hasattr(self.metric_logger, "log"):
            self.metric_logger.log(
                {
                    "outer_loop/path_consistency_mse": consistency_mse,
                    "ipf_iteration": ipf_iter,
                }
            )

    def log_compute_cost(self, models: List[nn.Module]):
        """Section 5: Hardware Footprint & Timing."""
        hw_footprint = sum(
            p.numel() for m in models for p in m.parameters() if p.requires_grad
        )
        metrics = {
            "compute/wall_clock_time_seconds": time.time() - self.start_time,
            "compute/total_gradient_steps": self.total_gradient_steps,
            "compute/total_nfes": self.total_nfes,
            "compute/hardware_footprint_params": hw_footprint,
        }
        if hasattr(self.metric_logger, "log"):
            self.metric_logger.log(metrics)

    @abstractmethod
    def fit(self, *args, **kwargs):
        pass
