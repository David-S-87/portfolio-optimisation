# main.py

import os
import time
import torch
import sys

# Ensure project root is on PYTHONPATH for common imports
dir_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if dir_root not in sys.path:
    sys.path.append(dir_root)

from common import PINN, save_checkpoint, export_logs_to_csv, train_model
from config import get_config
from loss import compute_loss


def main():
    # Load configuration
    config = get_config()

    # Create necessary directories for checkpoints and plots
    os.makedirs(os.path.dirname(config["checkpoint_path"]), exist_ok=True)
    plots_dir = os.path.join("single_asset", "heston", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    device = config["device"]

    # Initialize PINN model for (t, W, v) -> logV
    model = PINN(
        input_dim=3,           # t, W, and v
        hidden_dim=64,
        output_dim=1,          # log V
        n_hidden_layers=4,
        activation="tanh"
    ).to(device)

    # Choose optimizer based on configuration
    if config["optimizer_name"].lower() == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"]
        )
    elif config["optimizer_name"].lower() == "lbfgs":
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=config["lr"],
            max_iter=20
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config['optimizer_name']}")

    # Prepare supervised terminal condition at t = T
    # Create a meshgrid over W and v
    Ws = torch.linspace(
        config["bounds"]["W"][0], config["bounds"]["W"][1],
        config["eval_resolution"], device=device
    )
    vs = torch.linspace(
        config["bounds"]["v"][0], config["bounds"]["v"][1],
        config["eval_resolution"], device=device
    )
    grid_W, grid_v = torch.meshgrid(Ws, vs, indexing="ij")
    grid_W = grid_W.reshape(-1, 1)
    grid_v = grid_v.reshape(-1, 1)
    t_T = torch.ones_like(grid_W) * config["T"]

    # Stack to shape (N, 3)
    x_terminal = torch.cat([t_T, grid_W, grid_v], dim=1)

    # Compute log of terminal utility: log(W^(1-gamma) + eps)
    eps = 1e-8
    v_terminal = torch.log(grid_W.pow(1.0 - config["gamma"]) + eps)

    data_dict = {
        "x_terminal": x_terminal,
        "v_terminal": v_terminal
    }

    # Build trainer config
    trainer_config = {
        "epochs": config["epochs"],
        "batch_size": config["batch_size"],
        "device": device,
        "log_every": config["log_every"],
        "save_every": config["save_every"],
        "checkpoint_path": config["checkpoint_path"],
        "grad_clip": config.get("grad_clip", None)
    }

    # Train the model
    start_time = time.time()
    logs = train_model(
        model=model,
        optimizer=optimizer,
        loss_fn=compute_loss,
        sampler=config["sampler"],
        config=trainer_config,
        data_dict=data_dict
    )
    end_time = time.time()

    print(f"Training completed in {(end_time - start_time)/60:.2f} minutes.")

    # Save final model checkpoint
    final_ckpt = config["checkpoint_path"].format(epoch="final")
    save_checkpoint(model, final_ckpt, epoch="final")

    # Export training logs to CSV
    logs_csv = os.path.join(plots_dir, "heston_logs.csv")
    export_logs_to_csv(logs, logs_csv)
    print(f"Logs saved to {logs_csv}")


if __name__ == "__main__":
    main()
