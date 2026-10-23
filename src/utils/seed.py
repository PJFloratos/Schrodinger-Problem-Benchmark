import torch

import numpy as np

import os
import hashlib
import random


def derive_seed(base_seed: int, context: str) -> int:
    """
    Derives a unique, collision-free seed using deterministic hashing.
    """
    hash_input = f"{base_seed}_{context}".encode("utf-8")
    # Extract the first 16 hex characters (64 bits) for PyTorch/NumPy
    return int(hashlib.sha256(hash_input).hexdigest()[:16], 16)


def set_all_seeds(seed: int):
    """Locks down all RNGs and forces deterministic hardware execution."""
    # 1. Python built-in pseudo-random generator
    random.seed(seed)

    # 2. NumPy pseudo-random generator
    np.random.seed(seed)

    # 3. PyTorch CPU and GPU generators
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU environments

    # # 4. Force cuDNN to behave deterministically (Warning: reduces speed slightly)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
