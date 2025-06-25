# single_asset/Bates/loss.py

import os
import sys
# ensure project root is on path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import torch
import torch.nn.functional as F

from common.derivatives import (
    compute_v_t,
    compute_v_w,
    compute_v_vi,
    compute_v_ww,
    compute_v_vivj,
    compute_v_wvi
)
from common.integrals import compute_jump_integral, bates_jump_fn
from config import get_config


def compute_loss(model, batch, data_dict=None, num_jump_samples=10):
    config = get_config()
    device = config["device"]

    # --- 1. Inputs ---
    x = batch.to(device).clone().detach().requires_grad_(True)
    t = x[:, 0:1]; W = x[:, 1:2]; v = x[:, 2:3]

    # --- 2. Forward --- 
    logV, pi_hat, c_hat = model(x)
    # clamp network outputs
    logV   = torch.clamp(logV,   min=-15.0, max=15.0)
    pi_hat = torch.clamp(pi_hat, min=-5.0,  max=5.0)
    c_hat  = torch.clamp(c_hat,  min=1e-4,  max=100.0)

    V = torch.exp(logV)

    # --- 3. Derivatives ---
    V_t  = compute_v_t(model, x)
    V_W  = torch.clamp(compute_v_w(model, x),  min=1e-3)
    V_v  = compute_v_vi(model, x, 0)
    V_WW = torch.clamp(compute_v_ww(model, x), min=1e-3)
    V_vv = compute_v_vivj(model, x, 0, 0)
    V_Wv = compute_v_wvi(model, x, 0)

    # --- 4. Parameters ---
    mu, r, gamma = config["mu"], config["r"], config["gamma"]
    rho           = config["rho"]
    kappa, theta, xi, rho_corr = (
        config["kappa"], config["theta"],
        config["xi"], config["rho_corr"]
    )
    lambda_, mu_J, sigma_J = (
        config["lambda"], config["mu_J"], config["sigma_J"]
    )

    # --- 5. Heston terms ---
    drift_W      = (r * W + pi_hat * (mu - r) - c_hat) * V_W
    drift_v      = kappa * (theta - v) * V_v
    diff_WW      = 0.5 * (pi_hat ** 2) * v * V_WW
    diff_vv      = 0.5 * (xi ** 2) * v * V_vv
    cross_Wv     = rho_corr * xi * pi_hat * v * V_Wv

    # --- 6. Jump integral (safe sampling) ---
    # clamp J inside compute_jump_integral
    jump_term = compute_jump_integral(
        model=lambda z: torch.clamp(model(z)[0], -15.0, 15.0),
        x=x, pi=pi_hat,
        config={"lambda": lambda_, "mu_J": mu_J, "sigma_J": sigma_J},
        jump_fn=bates_jump_fn,
        num_samples=num_jump_samples
    ).unsqueeze(1)

    # --- 7. Utility ---
    utility = c_hat.pow(1.0 - gamma) / (1.0 - gamma)

    # --- 8. Residual & Losses ---
    residual = (
        V_t + drift_W + drift_v
        + diff_WW + diff_vv + cross_Wv
        + jump_term + utility
        - rho * V
    )
    loss_pde = F.mse_loss(residual, torch.zeros_like(residual))

    # supervised terminal loss
    loss_data = torch.tensor(0.0, device=device)
    if data_dict and "x_terminal" in data_dict:
        x_T      = data_dict["x_terminal"].to(device)
        v_true   = data_dict["v_terminal"].to(device)
        logV_pred = torch.clamp(model(x_T)[0], min=-15.0, max=15.0)
        loss_data = F.mse_loss(logV_pred, v_true)

    # consumption regularizer
    reg_loss = 1e-3 * (c_hat ** 2).mean()

    total = (
        config["lambda_pde"] * loss_pde
        + config["lambda_data"] * loss_data
        + reg_loss
    )
    return total, loss_pde, loss_data