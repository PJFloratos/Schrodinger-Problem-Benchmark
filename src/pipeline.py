from src.models import VelocityMLP, SimpleUNet
from src.training import Trainer, IPFTrainer, IMFTrainer, SF2MTrainer
from src.metrics import Evaluator
from src.metrics.plot import generate_and_plot
from src.utils import text_logger
from src.utils.seed import SeedOffsets
from src.configs import BaseConfig, Toy2dConfig, MNISTConfig

import torch
from torch import optim

from typing import Callable, Any, Dict, Tuple


class Pipeline:
    """Orchestrates the instantiation and training of generative methodologies."""

    logger = text_logger(__name__)

    def __init__(
        self,
        cfg: BaseConfig,
        train_dataset: torch.utils.data.Dataset,
        test_dataset: torch.utils.data.Dataset,
        metric_logger: Any,
    ):
        self.cfg = cfg
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.metric_logger = metric_logger

        # Initialize the evaluator internally
        self.evaluator = Evaluator(
            test_ds=self.test_dataset,
            device=self.cfg.device,
            sde_steps=self.cfg.eval_sim_steps,
            seed=self.cfg.eval_seed,
        )

    def execute(
        self, visualize: bool = True
    ) -> Tuple[torch.nn.Module, Dict[str, float]]:
        """Runs the full training and evaluation lifecycle."""
        # 1. Build and Train
        generative_model, train_metrics = self._build_and_train()

        # 2. Log final selection criteria
        model_type_str = (
            "EMA Weights"
            if getattr(self.cfg, "use_ema", True)
            else "Base Weights (No EMA)"
        )
        amp_str = "Enabled" if getattr(self.cfg, "use_amp", True) else "Disabled"
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

        # 4. Visualize
        if visualize:
            self.logger.info("Generating plots and visualizing results...")
            generate_and_plot(generative_model, eval_res, self.cfg)

        return generative_model, eval_res

    def _build_and_train(self) -> torch.nn.Module:
        """Routes to the correct training methodology and returns the generative model."""
        self.logger.info(
            f"Initializing training pipeline for model type: {self.cfg.model_type.upper()}"
        )

        if self.cfg.model_type == "ipf":
            return self._run_ipf()
        elif self.cfg.model_type == "imf":
            return self._run_imf()
        elif self.cfg.model_type == "sf2m":
            return self._run_sf2m()
        else:
            return self._run_standard()

    def _eval_callback(
        self, model: torch.nn.Module, direction: str
    ) -> Dict[str, float]:
        """Internal callback passed to the trainers for mid-training evaluation."""
        return self.evaluator.evaluate(
            model,
            num_samples=self.cfg.track_gen_samples,
            use_amp=self.cfg.use_amp,
            direction=direction,  # Let the evaluator handle the routing
        )

    def _get_model_class_and_kwargs(self):
        """Resolves the base architecture based on the dataset config."""
        if isinstance(self.cfg, Toy2dConfig):
            return VelocityMLP, {"d": self.cfg.input_dim, "hidden": self.cfg.hidden_dim}
        return SimpleUNet, {"channels": getattr(self.cfg, "input_channels", 1)}

    def _build_model(
        self, model_type: str, seed_offset: int, name: str
    ) -> torch.nn.Module:
        """
        Builds a network whose initial weights depend ONLY on (cfg.seed + offset)
        and the architecture, not on how many random numbers were drawn before.

        fork_rng snapshots the global CPU/CUDA RNG state and restores it on exit,
        so the seeding done here does not disturb the rest of the run.
        """
        model_class, kwargs = self._get_model_class_and_kwargs()

        with torch.random.fork_rng():
            torch.manual_seed(self.cfg.seed + seed_offset)
            model = model_class(**kwargs, model_type=model_type)  # built on CPU

        # Same value across solvers => same init.
        checksum = sum(p.detach().double().sum().item() for p in model.parameters())
        self.logger.info(f"[{name}] init checksum: {checksum:.10f}")

        return model.to(self.cfg.device)

    def _run_standard(self) -> torch.nn.Module:
        model = self._build_model(
            self.cfg.model_type,
            SeedOffsets.GEN_INIT,
            name=f"{self.cfg.model_type}-model",
        )
        self.logger.info(f"Standard Model deployed on: {self.cfg.device}")

        opt = optim.AdamW(
            model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
            fused=True,
        )

        trainer = Trainer(
            model=model,
            dataset=self.train_dataset,
            batch_size=self.cfg.batch_size,
            opt=opt,
            metric_logger=self.metric_logger,
            seed=self.cfg.seed,
            device=self.cfg.device,
            grad_clip=self.cfg.grad_clip,
            use_amp=self.cfg.use_amp,
            use_ema=self.cfg.use_ema,
        )

        gen_model, metrics = trainer.fit(
            epochs=self.cfg.epochs,
            save_per=self.cfg.save_interval,
            save_path=self.cfg.models_path,
            eval_callback=self._eval_callback,
            eval_per=self.cfg.eval_per,
        )

        return gen_model, metrics

    def _run_ipf(self) -> torch.nn.Module:
        f_model = self._build_model("sde", SeedOffsets.AUX_INIT, name="f_model")
        b_model = self._build_model("sde", SeedOffsets.GEN_INIT, name="b_model")
        self.logger.info(f"IPF Models deployed on: {self.cfg.device}")

        f_opt = optim.AdamW(
            f_model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        b_opt = optim.AdamW(
            b_model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )

        trainer = IPFTrainer(
            forward_model=f_model,
            backward_model=b_model,
            dataset=self.train_dataset,
            forward_opt=f_opt,
            backward_opt=b_opt,
            device=self.cfg.device,
            metric_logger=self.metric_logger,
            seed=self.cfg.seed,
            batch_size=self.cfg.batch_size,
            sde_steps=self.cfg.sim_steps,
            num_cache_batches=self.cfg.num_cache_batches,
            grad_clip=self.cfg.grad_clip,
            refresh_every=self.cfg.refresh_every,
            use_amp=self.cfg.use_amp,
            use_ema=self.cfg.use_ema,
        )

        gen_model, metrics = trainer.fit(
            ipf_iterations=self.cfg.epochs,
            inner_iterations=self.cfg.num_iter,
            save_per=self.cfg.save_interval,
            save_path=self.cfg.models_path,
            eval_callback=self._eval_callback,
            eval_per=self.cfg.eval_per,
        )

        return gen_model, metrics

    def _run_imf(self) -> torch.nn.Module:
        f_model = self._build_model("sde", SeedOffsets.AUX_INIT, name="f_model")
        b_model = self._build_model("sde", SeedOffsets.GEN_INIT, name="b_model")
        self.logger.info(f"IMF Models deployed on: {self.cfg.device}")

        f_opt = optim.AdamW(
            f_model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        b_opt = optim.AdamW(
            b_model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )

        trainer = IMFTrainer(
            forward_model=f_model,
            backward_model=b_model,
            dataset=self.train_dataset,
            forward_opt=f_opt,
            backward_opt=b_opt,
            device=self.cfg.device,
            metric_logger=self.metric_logger,
            seed=self.cfg.seed,
            batch_size=self.cfg.batch_size,
            sde_steps=self.cfg.sim_steps,
            num_cache_batches=self.cfg.num_cache_batches,
            grad_clip=self.cfg.grad_clip,
            refresh_every=self.cfg.refresh_every,
            use_amp=self.cfg.use_amp,
            use_ema=self.cfg.use_ema,
        )

        gen_model, metrics = trainer.fit(
            imf_iterations=self.cfg.epochs,
            inner_iterations=self.cfg.num_iter,
            save_per=self.cfg.save_interval,
            save_path=self.cfg.models_path,
            eval_callback=self._eval_callback,
            eval_per=self.cfg.eval_per,
        )

        return gen_model, metrics

    def _run_sf2m(self) -> torch.nn.Module:
        u_model = self._build_model("sde", SeedOffsets.GEN_INIT, name="u_model")
        s_model = self._build_model("sde", SeedOffsets.AUX_INIT, name="s_model")
        self.logger.info(f"SF2M Models deployed on: {self.cfg.device}")

        u_opt = optim.AdamW(
            u_model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        s_opt = optim.AdamW(
            s_model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )

        trainer = SF2MTrainer(
            u_model=u_model,
            s_model=s_model,
            dataset=self.train_dataset,
            u_opt=u_opt,
            s_opt=s_opt,
            device=self.cfg.device,
            metric_logger=self.metric_logger,
            seed=self.cfg.seed,
            batch_size=self.cfg.batch_size,
            sde_steps=self.cfg.sim_steps,
            num_cache_batches=self.cfg.num_cache_batches,
            ot_method=getattr(self.cfg, "ot_method", "minibatch"),
            grad_clip=self.cfg.grad_clip,
            use_amp=self.cfg.use_amp,
            use_ema=self.cfg.use_ema,
        )

        gen_model, metrics = trainer.fit(
            outer_iterations=self.cfg.epochs,
            inner_iterations=self.cfg.num_iter,
            save_per=self.cfg.save_interval,
            save_path=self.cfg.models_path,
            eval_callback=self._eval_callback,
            eval_per=self.cfg.eval_per,
        )

        # gen_model here is the SF2MInferenceWrapper, which combines u_model and s_model
        # to generate the correct drift for inference via Anderson's formula.
        return gen_model, metrics
