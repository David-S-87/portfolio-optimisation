# single_asset/Bates/main.py

import os
import time
import sys
import torch

# Ensure project root for imports
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if root not in sys.path:
    sys.path.append(root)

from common.nets import PINNWithControls
from common import train_model, save_checkpoint, export_logs_to_csv
from config import get_config
from loss import compute_loss


def main():
    # Load configuration
    config = get_config()
    device = config["device"]

    # Create directories for checkpoints and plots
    os.makedirs(os.path.dirname(config["checkpoint_path"]), exist_ok=True)
    plots_dir = os.path.join("single_asset", "Bates", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Initialize PINN model for (t, W, v) -> (logV, pi, c)
    model = PINNWithControls(
        input_dim=3,
        hidden_dim=64,
        n_hidden_layers=4,
        activation="tanh"
    ).to(device)

    # Choose optimizer
    opt_name = config["optimizer_name"].lower()
    if opt_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"]
        )
    elif opt_name == "lbfgs":
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=config["lr"],
            max_iter=20
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config['optimizer_name']}")

    # Prepare supervised terminal condition at t = T
    # Build grid over W and v
    res = config["eval_resolution"]
    W_min, W_max = config["bounds"]["W"]
    v_min, v_max = config["bounds"]["v"]
    Ws = torch.linspace(W_min, W_max, res, device=device)
    vs = torch.linspace(v_min, v_max, res, device=device)
    grid_W, grid_v = torch.meshgrid(Ws, vs, indexing="ij")
    grid_W = grid_W.reshape(-1, 1)
    grid_v = grid_v.reshape(-1, 1)
    t_T = torch.ones_like(grid_W) * config["T"]

    x_terminal = torch.cat([t_T, grid_W, grid_v], dim=1)
    # Terminal true log-value: log(W^(1-gamma)/(1-gamma))
    eps = 1e-8
    V_true = (grid_W.pow(1.0 - config["gamma"]) / (1.0 - config["gamma"]))
    v_terminal = torch.log(V_true + eps)

    data_dict = {
        "x_terminal": x_terminal,
        "v_terminal": v_terminal
    }

    # Build trainer configuration
    trainer_config = {
        "epochs": config["epochs"],
        "batch_size": config["batch_size"],
        "device": device,
        "log_every": config["log_every"],
        "save_every": config["save_every"],
        "checkpoint_path": config["checkpoint_path"],
        "grad_clip": config.get("grad_clip", None)
    }

    # Train model
    start_time = time.time()
    logs = train_model(
        model=model,
        optimizer=optimizer,
        loss_fn=compute_loss,
        sampler=config["sampler"],
        config=trainer_config,
        data_dict=data_dict
    )
    elapsed = (time.time() - start_time) / 60.0
    print(f"Training completed in {elapsed:.2f} minutes.")

    # Save final model checkpoint
    final_ckpt = config["checkpoint_path"].format(epoch="final")
    save_checkpoint(model, final_ckpt, epoch="final")

    # Export logs to CSV
    logs_csv = os.path.join(plots_dir, "bates_logs.csv")
    export_logs_to_csv(logs, logs_csv)
    print(f"Logs saved to {logs_csv}")


if __name__ == "__main__":
    main()
