from src.utils.log import text_logger

import torch
from torch import nn

from os import remove
from pathlib import Path


# Get the logger for this module
logger = text_logger(__name__)


def save_model(model: torch.nn.Module, path: str, stops=False) -> None:
    """
    Save a PyTorch model to a specified path.
    """
    target_path = Path("/".join(path.split("/")[:-1]))
    model_name = path.split("/")[-1]

    if not (model_name.endswith(".pth") or model_name.endswith(".pt")):
        logger.error("Wrong extension: Expecting `.pt` or `.pth`.")
        return

    # Creating the directory that the model is going to be saved if not exists
    if not target_path.exists():
        target_path.mkdir(parents=True, exist_ok=True)

    # If path already exists
    if Path(path).is_file():
        logger.info(f"Model `{model_name}` already exists on `{target_path}`.")
        if stops:
            return
        logger.warning(f"Deleting `{path}`.")
        remove(path)

    # Saving the Model to the given path
    logger.info(f"Saving Model `{model_name}` to `{target_path}`.")
    torch.save(obj=model.state_dict(), f=path)

    logger.debug(f"Model Successfully Saved to `{path}`.")


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

    logger.debug("Model succesfully loaded.")

    return model
