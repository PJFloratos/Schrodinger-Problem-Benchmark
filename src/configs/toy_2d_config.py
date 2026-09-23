from src.configs.base_config import BaseConfig

from dataclasses import dataclass


@dataclass
class Toy2dConfig(BaseConfig):
    """Parameters specific to the Toy2D dataset."""

    dataset_name: str = "SwissRoll"  # Triggers the path update in __post_init__
    dataset_type: str = "swiss_roll"  # in [swiss_roll, moons, checkerboard, gaussian]

    # Data sizes
    train_samples: int = 10000
    test_samples: int = 2000

    # Architecture
    input_dim: int = 2
    hidden_dim: int = 128
