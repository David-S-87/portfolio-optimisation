# trainer.py

import os
import torch
from torch import nn
from tqdm import trange

def train_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    sampler,
    config: dict,
    data_dict: dict = None,
):
    """
    Train a PINN model using a model-specific loss function.

    Args:
        model: neural network implementing forward(x) -> V predictions.
        optimizer: PyTorch optimizer.
        loss_fn: Callable(model, x_pde, data_dict) -> (total_loss, loss_pde, loss_data).
        sampler: Callable(batch_size) -> x_pde Tensor of collocation points.
        config: dict containing training settings:
            - epochs (int)
            - batch_size (int)
            - device (str)
            - log_every (int)
            - save_every (int)
            - checkpoint_path (str, with `{epoch}` placeholder)
            - grad_clip (float, optional)
        data_dict: dict of supervised data (keys 'x_terminal', 'v_terminal', ...).
    Returns:
        logs: dict with lists for 'total_loss', 'loss_pde', 'loss_data'.
    """
    # Device setup
    device = config.get("device", "cpu")
    model.to(device)
    if data_dict is None:
        data_dict = {}

    # Logging containers
    logs = {"total_loss": [], "loss_pde": [], "loss_data": []}

    # Report model size
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Training {model.__class__.__name__} on {device}, parameters: {total_params:,}")

    epochs = config["epochs"]
    batch_size = config["batch_size"]
    log_every = config.get("log_every", 100)
    save_every = config.get("save_every", 500)
    ckpt_path = config.get("checkpoint_path", "")

    # Training loop
    for epoch in trange(1, epochs + 1, desc="Training", unit="epoch"):
        try:
            # Sample collocation points for PDE residual
            x_pde = sampler(batch_size).to(device)

            # Compute losses
            total_loss, loss_pde, loss_data = loss_fn(model, x_pde, data_dict)

            # Check for NaNs
            if torch.isnan(total_loss):
                print(f"[Epoch {epoch}] WARNING: NaN loss, skipping backward step.")
                continue

            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping
            grad_clip = config.get("grad_clip", None)
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            # Log losses
            logs["total_loss"].append(total_loss.item())
            logs["loss_pde"].append(loss_pde.item())
            logs["loss_data"].append(loss_data.item())

            # Periodic logging
            if epoch % log_every == 0 or epoch == 1:
                print(f"[Epoch {epoch}/{epochs}] "
                      f"Total: {total_loss.item():.4e}, "
                      f"PDE: {loss_pde.item():.4e}, "
                      f"Data: {loss_data.item():.4e}")

            # Checkpointing
            if ckpt_path and (epoch % save_every == 0 or epoch == epochs):
                path = ckpt_path.format(epoch=epoch)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                torch.save(model.state_dict(), path)

        except Exception as e:
            print(f"[Epoch {epoch}] ERROR during training: {e}")
            break

    return logs
