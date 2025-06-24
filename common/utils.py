# utils.py

import os
import random
import csv
from typing import Dict, Tuple, List
import numpy as np
import torch
import pandas as pd
from torch import nn, Tensor


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    """
    Count the total number of trainable parameters in a model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def uniform_sampler(bounds: Dict[str, Tuple[float, float]], batch_size: int) -> Tensor:
    """
    Sample collocation points uniformly from provided domain bounds.

    Args:
        bounds: Ordered dict mapping variable names to (low, high) tuples.
                Insertion order defines input dimension order.
        batch_size: Number of samples.

    Returns:
        Tensor of shape (batch_size, len(bounds)).
    """
    lows = []
    highs = []
    for low, high in bounds.values():
        lows.append(low)
        highs.append(high)
    lows = torch.tensor(lows, dtype=torch.float32)
    highs = torch.tensor(highs, dtype=torch.float32)
    # Sample uniform [0,1)
    u = torch.rand(batch_size, len(bounds))
    # Scale to [low, high]
    return u * (highs - lows) + lows


def make_meshgrid(bounds: Dict[str, Tuple[float, float]], resolution: int) -> Tensor:
    """
    Create a 2D meshgrid for plotting over two variables.

    Args:
        bounds: Dict with exactly two keys for the two axes.
        resolution: Number of points per axis (e.g., 100 => 100x100 = 10,000 total).

    Returns:
        Tensor of shape (resolution^2, 2) with stacked grid points.
    """
    if len(bounds) != 2:
        raise ValueError("make_meshgrid requires exactly two bounds for a 2D grid.")

    if resolution > 1000:
        raise ValueError(f"Resolution too high ({resolution}x{resolution} = {resolution**2} points). "
                         f"Reduce to avoid excessive memory usage.")

    keys = list(bounds.keys())
    (low1, high1), (low2, high2) = bounds[keys[0]], bounds[keys[1]]

    # Debug check for invalid bounds
    if low1 >= high1 or low2 >= high2:
        raise ValueError(f"Invalid bounds: {keys[0]} ∈ [{low1}, {high1}], {keys[1]} ∈ [{low2}, {high2}]")

    # Generate uniform grids
    arr1 = np.linspace(low1, high1, resolution)
    arr2 = np.linspace(low2, high2, resolution)
    grid_x, grid_y = np.meshgrid(arr1, arr2, indexing="ij")

    # Stack and return
    points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    return torch.tensor(points, dtype=torch.float32)


def safe_div(numerator: Tensor, denominator: Tensor, eps: float = 1e-6) -> Tensor:
    """
    Safely divide with epsilon to avoid division by zero.
    """
    return numerator / (denominator + eps)


def log_losses(logs: Dict[str, List[float]], epoch: int) -> None:
    """
    Print the latest loss values in a formatted manner.

    Args:
        logs: Dict containing lists of 'total_loss', 'loss_pde', 'loss_data'.
        epoch: Current epoch number.
    """
    total = logs["total_loss"][-1]
    pde = logs["loss_pde"][-1]
    data = logs["loss_data"][-1]
    print(f"[Epoch {epoch}] - Total Loss: {total:.4e}, PDE Loss: {pde:.4e}, Data Loss: {data:.4e}")


def export_logs_to_csv(logs: Dict[str, List[float]], output_path: str) -> None:
    """
    Export training logs to a CSV file.

    Args:
        logs: Dict containing lists of 'total_loss', 'loss_pde', 'loss_data'.
        output_path: Path to save the CSV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(logs)
    df.to_csv(output_path, index_label="epoch")


def save_checkpoint(model: nn.Module, path: str, epoch: int) -> None:
    """
    Save model state dict to a checkpoint file.

    Args:
        model: The PyTorch model.
        path: Full file path including filename.
        epoch: Current epoch number (optional metadata).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, path)


def load_checkpoint(model: nn.Module, path: str, device: str = "cpu") -> None:
    """
    Load model state dict from a checkpoint file.

    Args:
        model: The PyTorch model to load into.
        path: Path to the checkpoint file.
        device: Device to map the loaded tensors.
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

