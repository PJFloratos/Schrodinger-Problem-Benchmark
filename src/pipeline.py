from src.models import VelocityMLP, EnergyMLP, SimpleUNet, EnergyUNet
from src.models.components import LightSBM
from src.training import (
    Trainer,
    IPFTrainer,
    IMFTrainer,
    SF2MTrainer,
    EBMTrainer,
    SCONETrainer,
    ENOTTrainer,
)
from src.metrics import Evaluator
from src.metrics.plot import (
    generate_and_plot,
    visualize_energy_landscape,
    visualize_enot_landscape,
)
from src.utils import text_logger
from src.utils.seed import derive_seed

import torch
from torch import optim

import os
from omegaconf import DictConfig
from typing import Callable, Any, Dict, Tuple, Optional


class Pipeline:
    """Orchestrates the instantiation and training of generative methodologies."""

    logger = text_logger(__name__)

    def __init__(
        self,
        cfg: DictConfig,
        train_dataset: torch.utils.data.Dataset,
        test_dataset: torch.utils.data.Dataset,
        metric_logger: Any,
    ):
        self.cfg = cfg
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.metric_logger = metric_logger

        # Convert string device to torch.device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"
        )

        # Safely extract variables that might be overridden by the dataset config
        self.batch_size = self.cfg.dataset.get("batch_size", self.cfg.batch_size)
        self.learning_rate = self.cfg.dataset.get(
            "learning_rate", self.cfg.learning_rate
        )
        self.use_compile = self.cfg.model.get("use_compile", self.cfg.use_compile)
        self.use_amp = self.cfg.model.get("use_amp", self.cfg.use_amp)
        self.use_ema = self.cfg.model.get("use_ema", self.cfg.use_ema)
        self.weight_decay = self.cfg.model.get("weight_decay", self.cfg.weight_decay)

        # Safely fetch the dataset scale (fallback to 1.0 for image datasets)
        self.dataset_scale = getattr(self.train_dataset, "scale", 1.0)

        # Mapped Sigma: \tilde{\sigma} = \sigma_{raw} / s
        self.normalized_sigma = self.cfg.sigma / self.dataset_scale

        self.gamma = self.cfg.gamma

        self.logger.info(
            f"Physics Scaled: sigma_raw={self.cfg.sigma} -> "
            f"sigma_norm={self.normalized_sigma:.5f} (scale={self.dataset_scale:.3f}) "
            f"gamma={self.gamma:.3f} (Invariant)"
        )

        # Extract solver-specific kwargs directly from the config
        solver_kwargs = {}
        if self.cfg.model.type == "ebm":
            solver_kwargs = {
                "sigma": self.normalized_sigma,
                "gamma": self.gamma,
                "eta_max_infer": self.cfg.model.get("ula_step_size", 1e-3),
                "eta_min_infer": self.cfg.model.get("eta_min_infer", 1e-4),
                "spatial_clip": self.cfg.model.get("spatial_clip", 10.0),
                "ebm_steps": self.cfg.model.get("ebm_steps", 1000),
            }

        # Initialize the evaluator internally
        self.evaluator = Evaluator(
            test_ds=self.test_dataset,
            device=self.device,
            sde_steps=self.cfg.eval_sim_steps,
            seed=self.cfg.eval_seed,
            solver_kwargs=solver_kwargs,
        )

    def execute(
        self, visualize: bool = True
    ) -> Tuple[torch.nn.Module, Dict[str, float]]:
        """Runs the full training and evaluation lifecycle."""
        # 1. Build and Train
        generative_model, train_metrics = self._build_and_train()

        # 2. Log final selection criteria
        model_type_str = "EMA Weights" if self.cfg.use_ema else "Base Weights (No EMA)"
        amp_str = "Enabled" if self.cfg.use_amp else "Disabled"
        self.logger.info(
            f"Model selected for generation: {model_type_str} | AMP: {amp_str}"
        )

        # 3. Final Evaluation
        eval_res = self.evaluator.evaluate(
            generative_model,
            log=True,
            use_amp=self.cfg.use_amp,
            num_samples=self.cfg.eval_gen_samples,
        )

        # 4. Visualize final generation samples
        if visualize:
            generate_and_plot(generative_model, eval_res, self.cfg)

        return generative_model, eval_res

    def _build_and_train(self) -> torch.nn.Module:
        """Routes to the correct training methodology and returns the generative model."""
        self.logger.info(
            f"Initializing training pipeline for model type: {self.cfg.model.type.upper()}"
        )

        if self.cfg.model.type == "ipf":
            return self._run_ipf()
        elif self.cfg.model.type == "imf":
            return self._run_imf()
        elif self.cfg.model.type == "sf2m":
            return self._run_sf2m()
        elif self.cfg.model.type == "ebm":
            return self._run_ebm()
        elif self.cfg.model.type == "scone":
            return self._run_scone()
        elif self.cfg.model.type == "enot":
            return self._run_enot()
        else:
            return self._run_standard()

    def _eval_callback(
        self, model: torch.nn.Module, direction: str, epoch: Optional[int] = None
    ) -> Dict[str, float]:
        """Internal callback passed to the trainers for mid-training evaluation."""
        eval_res = self.evaluator.evaluate(
            model,
            num_samples=self.cfg.track_gen_samples,
            use_amp=self.cfg.use_amp,
            direction=direction,
            epoch=epoch,
            cfg=self.cfg,
            trainer=self.trainer,
        )

        return eval_res

    def _get_model_class_and_kwargs(self, model_type: str):
        """Resolves the base architecture based on the dataset config."""
        # Safely resolve the flattened dimension for models that require a 1D state space
        if "input_dim" in self.cfg.dataset:
            flat_dim = self.cfg.dataset.input_dim
        else:
            flat_dim = self.cfg.dataset.input_channels * (
                self.cfg.dataset.image_size**2
            )

        if model_type == "lightsbm":
            return LightSBM, {
                "dim": flat_dim,
                "n_potentials": self.cfg.model.get("n_potentials", 1000),
                "epsilon": self.normalized_sigma**2,
            }

        if "input_dim" in self.cfg.dataset:
            # SCONE, EBM, and ENOT's Dual Potential require a scalar Energy/Potential architecture
            if model_type in ["ebm", "scone", "enot_potential"]:
                model_cls = EnergyMLP
            else:
                model_cls = VelocityMLP

            return model_cls, {
                "d": flat_dim,
                "hidden": self.cfg.hidden_dim,
                "time_dim": self.cfg.time_dim,
            }

        if model_type in ["ebm", "scone", "enot_potential"]:
            model_cls = EnergyUNet
        else:
            model_cls = SimpleUNet

        return model_cls, {
            "channels": self.cfg.dataset.input_channels,
            "image_size": self.cfg.dataset.image_size,
        }

    def _build_model(self, model_type: str, context: str, name: str) -> torch.nn.Module:
        """
        Builds a network whose initial weights depend ONLY on (cfg.seed + offset)
        and the architecture, not on how many random numbers were drawn before.

        fork_rng snapshots the global CPU/CUDA RNG state and restores it on exit,
        so the seeding done here does not disturb the rest of the run.
        """
        model_class, kwargs = self._get_model_class_and_kwargs(model_type)

        with torch.random.fork_rng():
            torch.manual_seed(derive_seed(self.cfg.seed, context))
            model = model_class(**kwargs, model_type=model_type)  # built on CPU

        # Same value across solvers => same init.
        checksum = sum(p.detach().double().sum().item() for p in model.parameters())
        self.logger.info(f"[{name}] init checksum: {checksum:.10f}")

        return model.to(self.device)

    def _run_standard(self) -> torch.nn.Module:
        model = self._build_model(
            self.cfg.model.type,
            "standard_model_init",
            name=f"{self.cfg.model.type}-model",
        )
        self.logger.info(f"Standard Model deployed on: {self.device}")

        # Data-driven initialization
        if hasattr(model, "init_from_dataset"):
            self.logger.debug(
                f"Performing data-driven initialization for {self.cfg.model.type}."
            )
            model.init_from_dataset(self.train_dataset)
            self.logger.debug("Successfully initialized data-driven parameters.")

        opt = optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.cfg.weight_decay,
            fused=True if self.device.type == "cuda" else False,
        )

        self.trainer = Trainer(
            model=model,
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            opt=opt,
            metric_logger=self.metric_logger,
            seed=self.cfg.seed,
            device=self.device,
            sigma=self.normalized_sigma,
            grad_clip=self.cfg.grad_clip,
            use_amp=self.cfg.use_amp,
            use_ema=self.cfg.use_ema,
            use_compile=self.use_compile,
        )

        gen_model, metrics = self.trainer.fit(
            epochs=self.cfg.epochs,
            save_per=self.cfg.model.get("save_interval", 500),
            save_path=self.cfg.models_path,
            eval_callback=self._eval_callback,
            eval_per=self.cfg.eval_per,
            plot_epoch=self.cfg.plot_epoch,
        )

        return gen_model, metrics

    def _run_ipf(self) -> torch.nn.Module:
        f_model = self._build_model("sde", "ipf_f_model_init", name="f_model")
        b_model = self._build_model("sde", "ipf_b_model_init", name="b_model")
        self.logger.info(f"IPF Models deployed on: {self.device}")

        f_opt = optim.AdamW(
            f_model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        b_opt = optim.AdamW(
            b_model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )

        self.trainer = IPFTrainer(
            forward_model=f_model,
            backward_model=b_model,
            dataset=self.train_dataset,
            forward_opt=f_opt,
            backward_opt=b_opt,
            device=self.device,
            metric_logger=self.metric_logger,
            seed=self.cfg.seed,
            sigma=self.normalized_sigma,
            batch_size=self.batch_size,
            sde_steps=self.cfg.model.get("sim_steps", 200),
            num_cache_batches=self.cfg.model.get("num_cache_batches", 10),
            grad_clip=self.cfg.grad_clip,
            refresh_every=self.cfg.model.get("refresh_every", 500),
            use_amp=self.cfg.use_amp,
            use_ema=self.cfg.use_ema,
        )

        gen_model, metrics = self.trainer.fit(
            ipf_iterations=self.cfg.epochs,
            inner_iterations=self.cfg.model.get("num_iter", 5000),
            save_per=self.cfg.model.get("save_interval", 500),
            save_path=self.cfg.models_path,
            eval_callback=self._eval_callback,
            eval_per=self.cfg.eval_per,
        )

        return gen_model, metrics

    def _run_imf(self) -> torch.nn.Module:
        f_model = self._build_model("sde", "imf_f_model_init", name="f_model")
        b_model = self._build_model("sde", "imf_b_model_init", name="b_model")
        self.logger.info(f"IMF Models deployed on: {self.device}")

        f_opt = optim.AdamW(
            f_model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        b_opt = optim.AdamW(
            b_model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )

        self.trainer = IMFTrainer(
            forward_model=f_model,
            backward_model=b_model,
            dataset=self.train_dataset,
            forward_opt=f_opt,
            backward_opt=b_opt,
            device=self.device,
            metric_logger=self.metric_logger,
            seed=self.cfg.seed,
            sigma=self.normalized_sigma,
            batch_size=self.batch_size,
            sde_steps=self.cfg.model.get("sim_steps", 200),
            num_cache_batches=self.cfg.model.get("num_cache_batches", 10),
            grad_clip=self.cfg.grad_clip,
            refresh_every=self.cfg.model.get("refresh_every", 500),
            use_amp=self.cfg.use_amp,
            use_ema=self.cfg.use_ema,
            use_compile=self.use_compile,
        )

        gen_model, metrics = self.trainer.fit(
            imf_iterations=self.cfg.epochs,
            inner_iterations=self.cfg.model.get("num_iter", 5000),
            save_per=self.cfg.model.get("save_interval", 500),
            save_path=self.cfg.models_path,
            eval_callback=self._eval_callback,
            eval_per=self.cfg.eval_per,
        )

        return gen_model, metrics

    def _run_sf2m(self) -> torch.nn.Module:
        u_model = self._build_model("sde", "sf2m_u_model_init", name="u_model")
        s_model = self._build_model("sde", "sf2m_s_model_init", name="s_model")
        self.logger.info(f"SF2M Models deployed on: {self.device}")

        u_opt = optim.AdamW(
            u_model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        s_opt = optim.AdamW(
            s_model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )

        self.trainer = SF2MTrainer(
            u_model=u_model,
            s_model=s_model,
            dataset=self.train_dataset,
            u_opt=u_opt,
            s_opt=s_opt,
            device=self.device,
            metric_logger=self.metric_logger,
            seed=self.cfg.seed,
            sigma=self.normalized_sigma,
            batch_size=self.batch_size,
            sde_steps=self.cfg.model.get("sim_steps", 200),
            num_cache_batches=self.cfg.model.get("num_cache_batches", 10),
            ot_method=self.cfg.model.get("ot_method", "minibatch"),
            grad_clip=self.cfg.grad_clip,
            use_amp=self.cfg.use_amp,
            use_ema=self.cfg.use_ema,
        )

        gen_model, metrics = self.trainer.fit(
            outer_iterations=self.cfg.epochs,
            inner_iterations=self.cfg.model.get("num_iter", 5000),
            save_per=self.cfg.model.get("save_interval", 500),
            save_path=self.cfg.models_path,
            eval_callback=self._eval_callback,
            eval_per=self.cfg.eval_per,
        )

        return gen_model, metrics

    def _run_ebm(self) -> torch.nn.Module:
        model = self._build_model("ebm", "ebm_model_init", name="ebm_model")
        self.logger.info(f"EBM Model deployed on: {self.device}")

        opt = optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )

        self.trainer = EBMTrainer(
            model=model,
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            opt=opt,
            metric_logger=self.metric_logger,
            seed=self.cfg.seed,
            device=self.device,
            sigma=self.normalized_sigma,
            gamma=self.gamma,
            buffer_size=self.cfg.model.get("buffer_size", 0),
            gp_weight=self.cfg.model.get("gp_weight", 0),
            ula_steps=self.cfg.model.get("ula_steps", 30),
            ula_step_size=self.cfg.model.get("ula_step_size", 5e-3),
            spatial_clip=self.cfg.model.get("spatial_clip", 50.0),
            l2_reg=self.cfg.model.get("l2_reg", 1e-4),
            cold_start=self.cfg.model.get("cold_start", 0.02),
            grad_clip=self.cfg.grad_clip,
            use_amp=self.cfg.use_amp,
            use_ema=self.cfg.use_ema,
            use_compile=False,  # torch.compile frequently breaks on double autograd
        )

        # Standard epoch orchestration inherited from Trainer
        gen_model, metrics = self.trainer.fit(
            epochs=self.cfg.epochs,
            save_per=self.cfg.model.get("save_interval", 500),
            save_path=self.cfg.models_path,
            eval_callback=self._eval_callback,
            eval_per=self.cfg.eval_per,
            plot_epoch=self.cfg.plot_epoch,
        )

        return gen_model, metrics

    def _run_scone(self) -> torch.nn.Module:
        phi_model = self._build_model("ebm", "scone_phi_model_init", name="phi_model")
        psi_model = self._build_model("ebm", "scone_psi_model_init", name="psi_model")
        self.logger.info(f"SCONE Dual Models deployed on: {self.device}")

        phi_opt = optim.AdamW(
            phi_model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        psi_opt = optim.AdamW(
            psi_model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )

        self.trainer = SCONETrainer(
            phi_model=phi_model,
            psi_model=psi_model,
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            phi_opt=phi_opt,
            psi_opt=psi_opt,
            metric_logger=self.metric_logger,
            seed=self.cfg.seed,
            device=self.device,
            sigma=self.normalized_sigma,
            reference=self.cfg.reference,
            spatial_clip=self.cfg.model.get("spatial_clip", 0.0),
            cold_start=self.cfg.model.get("cold_start", 0.0),
            buffer_size=self.cfg.model.get("buffer_size", 0),
            ula_steps=self.cfg.model.get("ula_steps", 30),
            ula_step_size=self.cfg.model.get("ula_step_size", 5e-3),
            eta_min=self.cfg.model.get("eta_min_infer", 1e-4),
            l2_reg=self.cfg.model.get("l2_reg", 1e-4),
            use_amp=self.cfg.use_amp,
            use_ema=self.cfg.use_ema,
            use_compile=False,  # Disable compile to protect double-autograd
        )

        gen_model, metrics = self.trainer.fit(
            epochs=self.cfg.epochs,
            save_per=self.cfg.model.get("save_interval", 500),
            eval_callback=self._eval_callback,
            eval_per=self.cfg.eval_per,
        )

        return gen_model, metrics

    def _run_enot(self) -> torch.nn.Module:
        drift_model = self._build_model(
            "enot_drift", "enot_drift_init", name="drift_model"
        )
        potential_model = self._build_model(
            "ebm", "enot_potential_init", name="potential_model"
        )
        self.logger.info(f"ENOT Dual Models deployed on: {self.device}")

        # Extract specific learning rates for TTUR (Two Time-Scale Update Rule)
        g_lr = self.cfg.model.get("drift_lr", self.learning_rate)
        d_lr = self.cfg.model.get("potential_lr", self.learning_rate / 10.0)

        drift_opt = optim.AdamW(
            drift_model.parameters(),
            lr=g_lr,
            weight_decay=self.weight_decay,
        )
        potential_opt = optim.AdamW(
            potential_model.parameters(),
            lr=d_lr,
            weight_decay=self.weight_decay,
        )

        weight_cfg = self.cfg.model.get("energy_weight", "auto")
        if weight_cfg == "auto":
            # Exact mathematical Schrödinger Bridge formulation: 1 / (2 * epsilon)
            energy_weight = 1.0 / (2.0 * (self.normalized_sigma**2))
        else:
            # If you hardcoded a float in the yaml like `energy_weight: 0.5`
            energy_weight = float(weight_cfg)

        self.trainer = ENOTTrainer(
            drift_model=drift_model,
            potential_model=potential_model,
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            drift_opt=drift_opt,
            potential_opt=potential_opt,
            metric_logger=self.metric_logger,
            seed=self.cfg.seed,
            device=self.device,
            sigma=self.normalized_sigma,
            inner_iters=self.cfg.model.get("inner_iters", 5),
            integration_steps=self.cfg.model.get("integration_steps", 50),
            energy_weight=energy_weight,
            potential_reg=self.cfg.model.get("potential_reg", 1e-3),
            gp_weight=self.cfg.model.get("gp_weight", 0.0),
            grad_clip=self.cfg.grad_clip,
            use_amp=self.use_amp,
            use_ema=self.use_ema,
            use_compile=self.use_compile,  # ENOT uses standard backprop through time, so compilation is safe
        )

        gen_model, metrics = self.trainer.fit(
            epochs=self.cfg.epochs,
            save_per=self.cfg.model.get("save_interval", 500),
            save_path=self.cfg.models_path,
            eval_callback=self._eval_callback,
            eval_per=self.cfg.eval_per,
        )

        return gen_model, metrics
