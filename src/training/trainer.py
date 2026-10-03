from src.training.base_trainer import BaseTrainer
from src.core.dynamics import ConditionalVectorField
from src.models.ema import EMAHelper
from src.utils.seed import SeedOffsets
from src.utils import text_logger, save_model

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

import time
from tqdm import tqdm
from contextlib import contextmanager
from typing import Callable, Tuple, Dict, Union, Any, Optional


class Trainer(BaseTrainer):
    logger = text_logger(__name__)

    # ------------------------------------------------------------------
    # Seeding design
    #
    # Weight init is already handled by TrainingOrchestrator._build_model
    # (fork_rng + seed + {0, 1}). Everything the trainer randomises gets its OWN
    # generator, seeded from (seed + offset), so no stream depends on how many
    # numbers any other consumer (eval, other loaders, other solvers) has drawn.
    # Offsets start at 1000 to stay clear of the init offsets (0, 1).
    # ------------------------------------------------------------------

    def __init__(
        self,
        model: nn.Module,
        dataset: Dataset,
        batch_size: int,
        opt: torch.optim.Optimizer,
        metric_logger: Any,
        seed: int,
        train_prop: float = 0.8,
        grad_clip: Optional[float] = 1.0,
        device: torch.device = torch.device("cpu"),
        ema_mu: float = 0.999,
        use_ema: bool = True,
        use_amp: bool = True,
    ) -> None:
        super().__init__(device=device, metric_logger=metric_logger)

        self.dataset = dataset
        self.batch_size = batch_size
        self.opt = opt
        self.seed = seed
        self.train_prop = train_prop
        self.grad_clip = grad_clip
        self.device = device
        self.use_ema = use_ema

        # -------------------------------------------------------------
        # 0. Private noise generators (live on the compute device)
        # -------------------------------------------------------------
        # Training noise: one continuous stream across epochs, touched by nothing else.
        self.train_gen = torch.Generator(device=self.device)
        self.train_gen.manual_seed(self.seed + SeedOffsets.TRAIN_NOISE)

        # Validation noise: re-seeded at the start of every validation pass, so every
        # epoch (and every run) validates on the exact same (t, noise) draws and
        # valid_loss only changes when the weights change.
        self.valid_gen = torch.Generator(device=self.device)

        # -------------------------------------------------------------
        # 1. Global channels_last & Hardware Settings
        # -------------------------------------------------------------
        self.data_shape = tuple(self.dataset[0].shape)
        is_image = len(self.data_shape) == 3  # (C, H, W)

        self.memory_format = (
            torch.channels_last
            if (self.device.type == "cuda" and is_image)
            else torch.contiguous_format
        )

        # if self.device.type == "cuda" and is_image:
        #     torch.backends.cudnn.benchmark = True

        # -------------------------------------------------------------
        # 2. Precision Settings (AMP)
        # -------------------------------------------------------------
        self.use_amp = use_amp and (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.amp_dtype = torch.float16 if self.device.type == "cuda" else torch.bfloat16

        # -------------------------------------------------------------
        # 3. Model, EMA & Compilation
        # -------------------------------------------------------------
        self.model_base = model.to(device, memory_format=self.memory_format)

        if self.use_ema:
            # The helper owns a persistent EMA copy of the model (`self.ema.model`)
            # and keeps it up to date after every optimizer step.
            self.ema = EMAHelper(mu=ema_mu, device=self.device)
            self.ema.register(self.model_base)
        else:
            self.ema = None

        self.use_compile = self.device.type == "cuda"
        if self.use_compile:
            self.model = torch.compile(self.model_base, mode="reduce-overhead")
        else:
            self.model = self.model_base

    # ------------------------------------------------------------------
    # RNG helpers
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        split_gen = torch.Generator().manual_seed(self.seed + SeedOffsets.TRAIN_SPLIT)

        train_ds, valid_ds = random_split(
            self.dataset,
            [self.train_prop, 1 - self.train_prop],
            generator=split_gen,
        )

        train_dl = DataLoader(
            train_ds,
            self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
            generator=torch.Generator().manual_seed(
                self.seed + SeedOffsets.TRAIN_SHUFFLE
            ),
        )
        valid_dl = DataLoader(
            valid_ds,
            self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
            generator=torch.Generator().manual_seed(
                self.seed + SeedOffsets.TRAIN_VALID_LOADER
            ),
        )

        return train_dl, valid_dl

    @staticmethod
    def _greedy_assignment_gpu(
        cost_matrix: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pure-GPU greedy approximation of Minibatch Optimal Transport.
        Eliminates the massive CPU synchronization bottleneck caused by SciPy.
        """
        B = cost_matrix.shape[0]
        row_ind = torch.arange(B, device=cost_matrix.device)
        col_ind = torch.zeros(B, dtype=torch.long, device=cost_matrix.device)

        flat_cost = cost_matrix.clone().flatten()
        for _ in range(B):
            min_idx = torch.argmin(flat_cost)
            r, c = min_idx // B, min_idx % B
            col_ind[r] = c

            # Mask out the assigned row and column
            flat_cost[r * B : (r + 1) * B] = float("inf")
            flat_cost[c::B] = float("inf")

        return row_ind, col_ind

    # ------------------------------------------------------------------
    # One pass over a loader (train if model.training, else validation)
    # ------------------------------------------------------------------

    def _process_data_loaders(
        self, dl: DataLoader, epoch: int, model: nn.Module
    ) -> Tuple[float, float]:
        total_loss = torch.tensor(0.0, device=self.device)
        training = model.training
        phase = "train" if training else "valid"
        model_type = self.model_base.model_type

        # All randomness in this pass comes from ONE private generator.
        gen = self.train_gen if training else self.valid_gen

        for z_batch in tqdm(dl, ascii=True, desc=f"             {phase}"):
            z_batch = z_batch.to(
                self.device, non_blocking=True, memory_format=self.memory_format
            )
            B = z_batch.shape[0]

            # Sample time uniformly t ~ U[0, 1]. Cap at 0.999 to avoid div by zero.
            t = torch.rand(B, 1, device=self.device, generator=gen) * 0.999

            # Target Construction via Dynamics
            x_t, target_u, t_expand = ConditionalVectorField.get_interpolant_and_target(
                model_type=model_type,
                z_batch=z_batch,
                t=t,
                gen=gen,
                memory_format=self.memory_format,
            )

            if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()

            with torch.autocast(
                device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
            ):
                pred_u = model(x_t, t)
                loss = ConditionalVectorField.compute_loss(
                    model_type=model_type,
                    pred_u=pred_u,
                    target_u=target_u,
                    t_expand=t_expand,
                )

            if training:
                self.opt.zero_grad(set_to_none=True)

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.opt)
                else:
                    loss.backward()

                # --- BaseTrainer Inner Metric Hook ---
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
                    # EMA tracks the live weights after every optimizer step
                    self.ema.update(self.model_base)

            total_loss += loss.detach().float()

        return (total_loss / len(dl)).item()

    def _training_step(self, train_dl: DataLoader, epoch: int) -> Tuple[float, float]:
        """
        Performs a single training step over the training DataLoader.
        """
        self.model.train()
        train_loss = self._process_data_loaders(train_dl, epoch, self.model)
        self.model.eval()

        return train_loss

    def _validation_step(
        self, valid_dl: DataLoader, epoch: int, model: nn.Module
    ) -> Tuple[float, float]:
        """
        Performs a single validation step over the validation DataLoader.
        """
        model.eval()

        # Same (t, noise) draws every epoch => valid_loss is comparable across epochs
        # and across runs; it only moves when the weights move.
        self.valid_gen.manual_seed(self.seed + SeedOffsets.TRAIN_VALID_NOISE)

        with torch.inference_mode():
            valid_loss = self._process_data_loaders(valid_dl, epoch, model)

        return valid_loss

    def fit(
        self,
        epochs: int,
        save_per: Union[int, None] = None,
        save_path: Union[str, None] = None,
        eval_per: Optional[int] = None,
        eval_callback: Optional[Callable] = None,
    ) -> Dict:
        Trainer.logger.debug("Start Training Process...")

        train_dl, valid_dl = self._get_loaders()

        # Set the probe batch for parameter drift tracking safely
        probe_batch = next(iter(valid_dl))
        if isinstance(probe_batch, (list, tuple)):
            probe_batch = probe_batch[0]
        self.fixed_probe_batch = probe_batch.to(self.device)

        for epoch in range(1, epochs + 1):
            Trainer.logger.debug(f"-> Epoch: {epoch}/{epochs}")

            phase_start = time.time()

            # Train the online model, then load its EMA weights ONCE for this epoch.
            train_loss = self._training_step(train_dl, epoch)

            # Determine which model to evaluate/save (EMA or base model)
            eval_model = self.ema.model if self.use_ema else self.model_base

            # Validate with EMA weights (NB: not directly comparable to train_loss,
            # which comes from the online weights).
            valid_loss = self._validation_step(valid_dl, epoch, eval_model)

            phase_time = time.time() - phase_start

            # Aggregate Outer-Loop Epoch Metrics
            metrics = {
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "phase_time_sec": phase_time,
            }

            # Run evaluation condition
            run_eval = False
            if eval_callback is not None:
                if eval_per is not None and (epoch % eval_per == 0):
                    run_eval = True

            if run_eval:
                # 'b' direction used traditionally for generative path evaluation
                with self._isolated_global_rng(self.seed + SeedOffsets.TRAIN_EVAL):
                    metrics.update(eval_callback(eval_model, direction="b"))

            # --- BaseTrainer Outer Metric Hooks ---
            self.log_phase_end("epoch", epoch, metrics)

            self.track_parameter_drift(self.model_base, ipf_iter=epoch)

            # Dynamically format the log string based on whether evaluation ran
            qual_str = ""
            if not run_eval:
                qual_str = ""
            elif "FID" in metrics:
                qual_str = (
                    f"FID: {metrics['FID']:.3f} | "
                    f"Prec: {metrics['Precision']:.3f} | "
                    f"Rec: {metrics['Recall']:.3f}"
                )
            else:
                qual_str = f"MMD: {metrics.get('eval_MMD', float('nan')):.6f}"

            Trainer.logger.info(
                f"     Epoch {epoch} | Train Loss: {train_loss:.6f} | Valid Loss: {valid_loss:.6f} | {qual_str}"
            )

            # Saving the model
            if save_per and save_path and (epoch % save_per == 0):
                save_model(
                    eval_model,
                    f"{save_path}/{self.model_base.__class__.__name__}_checkpoint_{epoch}.pth",
                )

            Trainer.logger.debug(("-" * 100))

        # Log final hardware footprint and parameters
        self.log_compute_cost([self.model_base])

        Trainer.logger.debug("Training Process Completed Successfully.")

        # Save model after training
        if save_path:
            save_model(
                eval_model,
                f"{save_path}/{self.model_base.__class__.__name__}_checkpoint_{epoch}.pth",
            )

        return eval_model, metrics


"""
Opt: 0:10
Unopt: 0:17
"""
