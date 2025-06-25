# single_asset/MJD/evaluate.py

import os
import sys
import torch
import matplotlib.pyplot as plt

# Ensure project root for imports
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if root not in sys.path:
    sys.path.append(root)

from common.nets import PINNWithControls
from config import get_config


def main():
    # Load config and device
    config = get_config()
    device = config["device"]

    # Initialize model and load checkpoint
    model = PINNWithControls(
        input_dim=2,
        hidden_dim=64,
        n_hidden_layers=4,
        activation="tanh"
    ).to(device)
    ckpt_path = config["checkpoint_path"].format(epoch="final")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
    model.eval()

    # Evaluation grid
    res = config["eval_resolution"]
    W_min, W_max = config["bounds"]["W"]
    W_vals = torch.linspace(W_min, W_max, res, device=device)

    # Evaluate at two time slices: t = 0 and t = T/2
    times = [0.0, config["T"] / 2]
    for t0 in times:
        t_tensor = torch.full((res, 1), t0, device=device)
        x_eval = torch.cat([t_tensor, W_vals.unsqueeze(1)], dim=1)

        with torch.no_grad():
            logV, pi_hat, c_hat = model(x_eval)

        # Convert to numpy
        W_np = W_vals.cpu().numpy()
        pi_np = pi_hat.squeeze().cpu().numpy()
        c_np = c_hat.squeeze().cpu().numpy()

        # Plot pi*) vs W
        plt.figure()
        plt.plot(W_np, pi_np)
        plt.xlabel("Wealth W")
        plt.ylabel(r"$\pi^*(t,W)$")
        plt.title(f"Optimal Allocation, t={t0:.2f}")
        plt.grid(True)
        out_pi = f"single_asset/MJD/plots/pi_t{int(t0*100)}.png"
        os.makedirs(os.path.dirname(out_pi), exist_ok=True)
        plt.savefig(out_pi)
        plt.close()

        # Plot c*) vs W
        plt.figure()
        plt.plot(W_np, c_np)
        plt.xlabel("Wealth W")
        plt.ylabel(r"$c^*(t,W)$")
        plt.title(f"Optimal Consumption, t={t0:.2f}")
        plt.grid(True)
        out_c = f"single_asset/MJD/plots/c_t{int(t0*100)}.png"
        plt.savefig(out_c)
        plt.close()

        print(f"Saved plots for t={t0:.2f}: {out_pi}, {out_c}")

if __name__ == "__main__":
    main()
