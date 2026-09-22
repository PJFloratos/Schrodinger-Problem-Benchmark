from src.configs.base_config import BaseConfig

from dataclasses import dataclass


@dataclass
class MNISTConfig(BaseConfig):
    """Parameters specific to the MNIST dataset."""

    dataset_name: str = "MNIST"

    # Architecture (Images)
    input_channels: int = 1
    image_size: int = 28

    # Overriding base params for MNIST if needed
    batch_size: int = 256  # Example: MNIST might need a smaller batch
    learning_rate: float = 2e-4  # Example: Different LR for images
