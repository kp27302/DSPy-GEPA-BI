"""Deterministic seed management for reproducibility."""

import random
import numpy as np
import os


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across Python, NumPy, and environment.
    
    Args:
        seed: Integer seed value for random number generators
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_deterministic_config() -> dict:
    """
    Return configuration dict for deterministic behavior.
    
    Returns:
        Dictionary of settings for reproducible execution
    """
    return {
        "seed": 42,
        "deterministic": True,
        "benchmark": False,
        "workers": 1,
    }

