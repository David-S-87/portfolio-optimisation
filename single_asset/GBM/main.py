# main.py

import os
import time
import torch

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from common import (
    PINN,
    save_checkpoint,
    export_logs_to_csv,
    train_model
)
from config import get_config
from loss import compute_loss


def main():
    # Load config as a dictionary
    config = get_config()

    # Create necessary directories
    os.makedirs(os.path.dirname(config["checkpoint_path"]), exist_ok=True)
    plots_dir = os.path.join("single_asset", "gbm", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    device = config["device"]

    # Initialize PINN model
    model = PINN(
        input_dim=2,
        hidden_dim=64,
        output_dim=1,
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

    # Supervised terminal condition V(T, W) = U(W) = W^{1-γ}/(1−γ)
    Ws = torch.linspace(config["bounds"]["W"][0], config["bounds"]["W"][1],
                        config["eval_resolution"], device=device).unsqueeze(1)
    t_T = torch.ones_like(Ws) * config["T"]
    x_terminal = torch.cat([t_T, Ws], dim=1)
    v_terminal = torch.log(Ws.pow(1.0 - config["gamma"]) + 1e-8)

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
        "grad_clip": config["grad_clip"]
    }

    # Train
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

    # Save final model
    final_ckpt = config["checkpoint_path"].format(epoch="final")
    save_checkpoint(model, final_ckpt, epoch="final")

    # Export logs
    logs_csv = os.path.join(plots_dir, "gbm_logs.csv")
    export_logs_to_csv(logs, logs_csv)
    print(f"Logs saved to {logs_csv}")


if __name__ == "__main__":
    main()

