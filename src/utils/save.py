from src.utils.log import text_logger

import torch
from torch import nn

from os import remove
from pathlib import Path


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
        remove(path)

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
