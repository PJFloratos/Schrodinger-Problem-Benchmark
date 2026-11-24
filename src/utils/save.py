from src.utils.log import text_logger

import torch
from torch import nn

import os
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import json
from typing import Dict, Union


# Get the logger for this module
logger = text_logger(__name__)


def save_model(model: torch.nn.Module, path: str, stops=False) -> None:
    """
    Save a PyTorch model to a specified path quietly.
    """
    p = Path(path)
    target_path = p.parent
    model_name = p.name

    if not (model_name.endswith(".pth") or model_name.endswith(".pt")):
        logger.error(f"Wrong extension for `{model_name}`: Expecting `.pt` or `.pth`.")
        return

    # Creating the directory that the model is going to be saved if not exists
    target_path.mkdir(parents=True, exist_ok=True)

    # Handle existing files quietly
    if p.is_file():
        if stops:
            logger.debug(f"Model `{model_name}` already exists; skipping save.")
            return
        os.remove(path)

    # Save the Model
    torch.save(obj=model.state_dict(), f=path)
    logger.debug(f"Model successfully saved to `{path}`.")


def load_model(
    model_class: nn.Module,
    model_path: str,
    device: torch.device = torch.device("cpu"),
    **kargs,
) -> nn.Module:
    """
    Loads a PyTorch model from a specified file.
    """
    # Initialize the model
    model = model_class(**kargs)

    # Load the state dict (parameters)
    state_dict = torch.load(model_path, map_location=torch.device(device))

    # Load the parameters into the model
    model.load_state_dict(state_dict)

    # Set the model to evaluation mode
    model.eval()

    logger.debug(f"Model successfully loaded from `{model_path}`.")

    return model


def save_config_snapshot(
    cfg: DictConfig, path: Union[str, Path], fmt: str = "log"
) -> None:
    """Writes a read-only resolved config snapshot to disk for reproducibility & sanity checks."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt in ("log", "yaml"):
        content = OmegaConf.to_yaml(cfg, resolve=True)
        path.write_text(content, encoding="utf-8")
    elif fmt == "json":
        resolved = OmegaConf.to_container(cfg, resolve=True)
        path.write_text(json.dumps(resolved, indent=2), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported config snapshot format: {fmt}")
