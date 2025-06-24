# evaluate.py

import os
import sys
import math
import warnings

# Silence the PyTorch weights_only FutureWarning
warnings.filterwarnings(
    "ignore",
    message=".*weights_only=False.*",
    category=FutureWarning,
)

import torch
import numpy as np
import matplotlib.pyplot as plt

# Ensure project root is on PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from common import PINN, load_checkpoint, compute_v_w, compute_v_ww, make_meshgrid
from config import get_config


def compute_controls(model, bounds: dict, resolution: int):
    t_min, t_max = bounds["t"]
    W_min, W_max = bounds["W"]
    t_vals = np.linspace(t_min, t_max, resolution)
    W_vals = np.linspace(W_min, W_max, resolution)
    Tg, Wg = np.meshgrid(t_vals, W_vals, indexing="ij")

    coords = np.stack([Tg.ravel(), Wg.ravel()], axis=1)
    x_t = torch.tensor(coords, dtype=torch.float32, device=config["device"], requires_grad=True)

    print(f"[DEBUG] Input grid shape: {x_t.shape}")  # Should be (10000, 2)

    V_W  = compute_v_w(model, x_t)
    V_WW = compute_v_ww(model, x_t)

    # Flatten tensors for safety
    V_W  = V_W.view(-1)
    V_WW = V_WW.view(-1)
    W    = x_t[:, 1].view(-1)

    # Assumes V_W, V_WW, W all shape (N,)
    V_W   = V_W.view(-1)
    V_WW  = V_WW.view(-1)
    W     = x_t[:, 1].view(-1)

    eps = 1e-6
    V_W  = torch.clamp(V_W, min=eps)
    V_WW = torch.clamp(V_WW, min=eps)
    W    = torch.clamp(W,    min=eps)

    mu, r, sigma, gamma = config["mu"], config["r"], config["sigma"], config["gamma"]

    # Optimal control based on the HJB first order condition
    pi_star = - (mu - r) / (sigma**2) * (V_W / V_WW)
    c_star  = V_W.pow(-1.0 / gamma)


    print(f"[DEBUG] pi_star shape: {pi_star.shape}")  # Expect (10000,)

    pi_grid = pi_star.detach().cpu().numpy().reshape(resolution, resolution)
    c_grid  = c_star.detach().cpu().numpy().reshape(resolution, resolution)

    return t_vals, W_vals, pi_grid, c_grid



def plot_controls_from_grids(t_vals: np.ndarray,
                             W_vals: np.ndarray,
                             pi_grid: np.ndarray,
                             c_grid: np.ndarray,
                             save_dir: str = None):
    """
    Contour-plot π*(t,W) and c*(t,W) from precomputed grids.
    """
    T_grid, W_grid = np.meshgrid(t_vals, W_vals, indexing="ij")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    cs1 = ax1.contourf(W_grid, T_grid, pi_grid, levels=50, cmap='viridis')
    fig.colorbar(cs1, ax=ax1, label=r"$\pi^*(t,W)$")
    ax1.set(title=r"Optimal Investment $\pi^*(t,W)$", xlabel="Wealth $W$", ylabel="Time $t$")

    cs2 = ax2.contourf(W_grid, T_grid, c_grid, levels=50, cmap='plasma')
    fig.colorbar(cs2, ax=ax2, label=r"$c^*(t,W)$")
    ax2.set(title=r"Optimal Consumption $c^*(t,W)$", xlabel="Wealth $W$", ylabel="Time $t$")

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, "controls.png")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Control plots saved to: {out_path}")
    else:
        plt.show()

def compare_to_merton(model, bounds, resolution, mu, sigma, gamma, r):
    """
    Compare learned controls to analytical Merton controls.
    """
    t_vals = np.linspace(bounds["t"][0], bounds["t"][1], resolution)
    W_vals = np.linspace(bounds["W"][0], bounds["W"][1], resolution)
    Tg, Wg = np.meshgrid(t_vals, W_vals, indexing="ij")

    coords = np.stack([Tg.ravel(), Wg.ravel()], axis=1)
    device = next(model.parameters()).device
    x_t = torch.tensor(coords, dtype=torch.float32, device=device, requires_grad=True)

    V_W  = compute_v_w(model, x_t).view(-1)
    V_WW = compute_v_ww(model, x_t).view(-1)
    W    = x_t[:, 1].view(-1)

    eps = 1e-6
    V_W  = torch.clamp(V_W,  min=eps)
    V_WW = torch.clamp(V_WW, min=eps)
    W    = torch.clamp(W,    min=eps)

    # Compute using FOCs
    pi_star_learned = - (mu - r) / (sigma**2) * (V_W / V_WW)
    c_star_learned  = V_W.pow(-1.0 / gamma)

    # Compute Merton analytical values
    pi_merton = (mu - r) / (sigma**2 * gamma)
    c_merton = W.detach().cpu().numpy() ** (-1.0 / gamma)

    # Reshape for plotting (now using detach())
    pi_learned_grid = pi_star_learned.detach().cpu().numpy().reshape(resolution, resolution)
    c_learned_grid  = c_star_learned.detach().cpu().numpy().reshape(resolution, resolution)
    c_merton_grid   = c_merton.reshape(resolution, resolution)

    # Plot comparison
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    axs[0].contourf(Wg, Tg, pi_learned_grid, levels=50)
    axs[0].set_title("Learned π*(t,W)")
    axs[0].set_xlabel("W"); axs[0].set_ylabel("t")

    axs[1].contourf(Wg, Tg, c_learned_grid, levels=50)
    axs[1].set_title("Learned c*(t,W)")

    axs[2].contourf(Wg, Tg, c_merton_grid, levels=50)
    axs[2].set_title("Analytical Merton c*(W)")

    plt.tight_layout()
    plt.show()

    print(
        f"Merton π*: {pi_merton:.4f}, "
        f"Learned π* stats: mean={pi_star_learned.mean().item():.4f}, "
        f"std={pi_star_learned.std().item():.4f}"
    )

def main():
    global config  # needed inside compute_controls
    config = get_config()
    device = config["device"]

    # Load trained PINN
    model = PINN(input_dim=2, hidden_dim=64, output_dim=1, 
                 n_hidden_layers=4, activation="tanh").to(device)
    checkpoint = config["checkpoint_path"].format(epoch="final")
    load_checkpoint(model, checkpoint, device)

    # Compute and plot
    res = config["eval_resolution"]
    t_vals, W_vals, pi_grid, c_grid = compute_controls(model, config["bounds"], res)
    plots_dir = os.path.join("single_asset", "gbm", "plots")
    plot_controls_from_grids(t_vals, W_vals, pi_grid, c_grid, save_dir=plots_dir)

    compare_to_merton(model, config["bounds"], config["eval_resolution"],
                  mu=0.08, sigma=0.2, gamma=3.0, r=0.02)


if __name__ == "__main__":
    main()
