# evaluate.py
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# make sure common/ is on the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from common import (
    PINN,
    load_checkpoint, compute_v_w,
    compute_v_ww, compute_v_wvi
)
from config import get_config

def compute_controls(model, t0, W_vals, v_vals, config):
    """
    Compute optimal controls π* and c* on a grid of (W, v) at fixed time t0.
    Returns meshgrid Wg, vg and 2D arrays pi_grid, c_grid.
    """
    device = config["device"]
    mu, r = config["mu"], config["r"]
    xi, corr = config["xi"], config["corr"]
    gamma = config["gamma"]
    
    # Create meshgrid
    Wg, vg = np.meshgrid(W_vals, v_vals, indexing="ij")  # shape (res, res)
    res = Wg.shape[0]
    
    # Flatten and build input tensor (t, W, v)
    t_flat = np.full(Wg.size, t0, dtype=np.float32)
    x_np = np.stack([t_flat, Wg.ravel().astype(np.float32), vg.ravel().astype(np.float32)], axis=1)
    x = torch.tensor(x_np, requires_grad=True, device=device)
    
    # Forward: predict logV, then exp with clamp
    logV = model(x)
    logV = torch.clamp(logV, min=-20.0, max=20.0)
    V = torch.exp(logV)
    
    # Derivatives of logV
    V_W  = compute_v_w(model, x)
    V_WW = compute_v_ww(model, x)
    V_Wv = compute_v_wvi(model, x, 0)
    # (we don't need V_v for controls)
    
    # Stabilize denominators
    eps = 1e-6
    V_W  = torch.clamp(V_W,  min=eps)
    V_WW = torch.clamp(V_WW, min=eps)
    
    # Compute controls (same formula as in loss.py)
    pi_star = (((mu - r) * V_W) + (xi * corr * x[:, 2:3] * V_Wv)) / (x[:, 2:3] * V_WW)
    c_star  = V_W.pow(-1.0 / gamma)
    
    # Move to CPU numpy and reshape
    pi_np = pi_star.detach().cpu().numpy().reshape(res, res)
    c_np  = c_star.detach().cpu().numpy().reshape(res, res)
    
    return Wg, vg, pi_np, c_np

def main():
    # Load configuration and model
    config = get_config()
    device = config["device"]
    ckpt_path = config["checkpoint_path"].format(epoch="final")
    
    model = PINN(
        input_dim=3,
        hidden_dim=64,
        output_dim=1,
        n_hidden_layers=4,
        activation="tanh"
    ).to(device)
    load_checkpoint(model, ckpt_path, device=device)
    model.eval()
    
    # Evaluation grid parameters
    res = config["eval_resolution"]
    W_min, W_max = config["bounds"]["W"]
    v_min, v_max = config["bounds"]["v"]
    W_vals = np.linspace(W_min, W_max, res)
    v_vals = np.linspace(v_min, v_max, res)
    
    # Times at which to evaluate controls
    times = [0.0, config["T"] / 2.0]
    
    # Prepare output directory
    plots_dir = os.path.join("single_asset", "heston", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    for t0 in times:
        Wg, vg, pi_grid, c_grid = compute_controls(model, t0, W_vals, v_vals, config)
        
        # π* contour
        plt.figure(figsize=(6,5))
        plt.contourf(Wg, vg, pi_grid, levels=50)
        plt.colorbar(label="π*")
        plt.xlabel("Wealth W")
        plt.ylabel("Variance v")
        plt.title(f"Optimal π* at t = {t0:.2f}")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"pi_star_t{t0:.2f}.png"))
        plt.close()
        
        # c* contour
        plt.figure(figsize=(6,5))
        plt.contourf(Wg, vg, c_grid, levels=50)
        plt.colorbar(label="c*")
        plt.xlabel("Wealth W")
        plt.ylabel("Variance v")
        plt.title(f"Optimal c* at t = {t0:.2f}")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"c_star_t{t0:.2f}.png"))
        plt.close()
        
        # Print summary stats
        print(f"[t = {t0:.2f}] π*  mean={pi_grid.mean():.4f}, std={pi_grid.std():.4f}")
        print(f"[t = {t0:.2f}] c*  mean={c_grid.mean():.4f}, std={c_grid.std():.4f}")

if __name__ == "__main__":
    main()
