from src.configs.base_config import BaseConfig

from dataclasses import dataclass


@dataclass
class CIFAR10Config(BaseConfig):
    """Parameters specific to the CIFAR-10 dataset."""

    dataset_name: str = "CIFAR10"

    # Architecture (Images)
    input_channels: int = 3
    image_size: int = 32

    # Overriding base params for CIFAR-10 if needed
    batch_size: int = 256  # Standard batch size for CIFAR-10
    learning_rate: float = 2e-4  # Standard LR for image generative models
