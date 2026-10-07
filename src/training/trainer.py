from src.training.base_trainer import BaseTrainer
from src.core.dynamics import ConditionalVectorField
from src.models.components import EMAHelper
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
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            seed=seed,
            device=device,
            metric_logger=metric_logger,
            use_amp=use_amp,
        )

        self.opt = opt
        self.train_prop = train_prop
        self.grad_clip = grad_clip
        self.use_ema = use_ema

        # Training noise: one continuous stream across epochs, touched by nothing else.
        self.train_gen = torch.Generator(device=self.device)
        self.train_gen.manual_seed(self.seed + SeedOffsets.TRAIN_NOISE)

        # Validation noise: re-seeded at the start of every validation pass, so every
        # epoch (and every run) validates on the exact same (t, noise) draws and
        # valid_loss only changes when the weights change.
        self.valid_gen = torch.Generator(device=self.device)

        # AMP scaler
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Model setup
        self.model_base, self.model, self.ema, _ = self._setup_model(
            model=model, ema_mu=ema_mu, use_ema=use_ema, create_sampler=False
        )

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

    # ------------------------------------------------------------------
    # Training & Validation
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
            x_t, target_u, t_net = ConditionalVectorField.get_interpolant_and_target(
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
                    t_net=t_net,
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
            _run_eval = (
                eval_callback is not None
                and eval_per is not None
                and (epoch % eval_per == 0)
            )
            if _run_eval:
                # 'b' direction used traditionally for generative path evaluation
                with self._isolated_global_rng(self.seed + SeedOffsets.TRAIN_EVAL):
                    metrics.update(eval_callback(eval_model, direction="b"))

            # --- BaseTrainer Outer Metric Hooks ---
            self.log_phase_end("epoch", epoch, metrics)
            self.track_parameter_drift(self.model_base, ipf_iter=epoch)

            # Logging
            qual_str = self._format_eval_str(metrics, _run_eval)
            self.logger.info(
                f"     Epoch {epoch} | Train Loss: {train_loss:.6f} | Valid Loss: {valid_loss:.6f} | {qual_str}"
            )

            # Saving the model
            _save_model = (
                save_per is not None
                and save_path is not None
                and (epoch % save_per == 0)
            )
            if _save_model:
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
                f"{save_path}/{self.model_base.__class__.__name__}_checkpoint_final.pth",
            )

        return eval_model, metrics
