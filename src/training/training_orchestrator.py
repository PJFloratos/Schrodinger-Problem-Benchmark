from src.models import VelocityMLP, SimpleUNet
from src.training import Trainer, IPFTrainer
from src.utils import configure_logger
from src.configs import BaseConfig, Toy2dConfig, MNISTConfig

import torch
from torch import optim


class TrainingOrchestrator:
    """Orchestrates the instantiation and training of generative methodologies."""

    logger = configure_logger(__name__)

    def __init__(self, cfg: BaseConfig, dataset: torch.utils.data.Dataset):
        self.cfg = cfg
        self.dataset = dataset

    def build_and_train(self) -> torch.nn.Module:
        """Routes to the correct training methodology and returns the generative model."""
        self.logger.info(
            f"Initializing training pipeline for model type: {self.cfg.model_type.upper()}"
        )

        if self.cfg.model_type == "ipf":
            return self._run_ipf()
        elif self.cfg.model_type == "imf":
            return self._run_imf()
        else:
            return self._run_standard()

    def _get_model_class_and_kwargs(self):
        """Resolves the base architecture based on the dataset config."""
        if isinstance(self.cfg, Toy2dConfig):
            return VelocityMLP, {"d": self.cfg.input_dim, "hidden": self.cfg.hidden_dim}
        return SimpleUNet, {"channels": getattr(self.cfg, "input_channels", 1)}

    def _run_ipf(self) -> torch.nn.Module:
        model_class, kwargs = self._get_model_class_and_kwargs()

        f_model = model_class(**kwargs, model_type="sde").to(self.cfg.device)
        b_model = model_class(**kwargs, model_type="sde").to(self.cfg.device)
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
            dataset=self.dataset,
            forward_opt=f_opt,
            backward_opt=b_opt,
            device=self.cfg.device,
            batch_size=self.cfg.batch_size,
            sde_steps=self.cfg.sim_steps,
            num_cache_batches=self.cfg.num_cache_batches,
        )

        trainer.fit(ipf_iterations=self.cfg.epochs, inner_iterations=self.cfg.num_iter)

        return b_model

    def _run_standard(self) -> torch.nn.Module:
        model_class, kwargs = self._get_model_class_and_kwargs()

        model = model_class(**kwargs, model_type=self.cfg.model_type).to(
            self.cfg.device
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
            dataset=self.dataset,
            batch_size=self.cfg.batch_size,
            opt=opt,
            device=self.cfg.device,
        )

        trainer.fit(
            epochs=self.cfg.epochs,
            save_per=self.cfg.save_interval,
            save_path=self.cfg.models_path,
        )

        return model

    def _run_imf(self) -> torch.nn.Module:
        # Placeholder for your future IMF implementation
        raise NotImplementedError("IMF pipeline is not yet implemented.")
