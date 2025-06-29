# single_asset/MJD/loss.py

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import torch
import torch.nn.functional as F

from common import compute_v_t, compute_v_w, compute_v_ww
from common import compute_jump_integral, mjd_jump_fn
from config import get_config


def compute_loss(model, batch, data_dict=None, num_jump_samples=10):
    """
    Compute the PINN loss for single-asset MJD (value + controls).

    Args:
        model: PINNWithControls mapping x -> (logV, pi, c)
        batch: Tensor (N, 2) of [t, W]
        data_dict: optional dict with keys 'x_terminal' and 'v_terminal'
        num_jump_samples: number of Monte Carlo draws for jump integral

    Notes:
        The network outputs ``pi_hat`` and ``c_hat`` are clamped to ``[-5, 5]``
        and ``[1e-3, 100]`` respectively before computing the utility term to
        prevent unstable training.

    Returns:
        total_loss, loss_pde, loss_data
    """
    config = get_config()
    device = config["device"]

    # Prepare inputs
    x = batch.to(device).clone().detach().requires_grad_(True)
    t = x[:, 0:1]
    W = x[:, 1:2]

    # Forward pass through model
    logV, pi_hat, c_hat = model(x)
    c_hat = torch.clamp(c_hat, min=1e-3, max=100.0)

    # Clamp controls for numerical stability
    pi_hat = torch.clamp(pi_hat, min=-5.0, max=5.0)
    c_hat = torch.clamp(c_hat, min=1e-3, max=100.0)

    # Clamp logV for stability
    logV = torch.clamp(logV, min=-15.0, max=15.0)
    V = torch.exp(logV)
    if torch.isnan(V).any():
        raise ValueError("NaN encountered in V")

    # Derivatives of V
    V_t = compute_v_t(model, x)
    V_W = torch.clamp(compute_v_w(model, x), min=1e-4)
    V_WW = torch.clamp(compute_v_ww(model, x), min=1e-4)
    if torch.isnan(V_W).any() or torch.isnan(V_WW).any():
        raise ValueError("NaN encountered in V_W or V_WW")

    # Model parameters
    mu, sigma = config["mu"], config["sigma"]
    r, gamma = config["r"], config["gamma"]
    rho = config["rho"]

    # PDE terms
    drift_term     = (r * W + pi_hat * (mu - r) - c_hat) * V_W
    diffusion_term = 0.5 * (pi_hat ** 2) * (sigma ** 2) * V_WW

    # Jump integral term
    jump_term = compute_jump_integral(
        model=lambda z: model(z)[0],
        x=x,
        pi=pi_hat,
        config={
            "lambda": config["lambda"],
            "mu_J": config["mu_J"],
            "sigma_J": config["sigma_J"]
        },
        jump_fn=mjd_jump_fn,
        num_samples=num_jump_samples
    ).unsqueeze(1)  # ensure shape (N, 1)
    jump_term = torch.clamp(jump_term, min=-1e4, max=1e4)

    # Utility term

    if abs(gamma - 1.0) < 1e-3:
        utility_term = torch.log(c_hat)
    else:
        utility_term = c_hat.pow(1.0 - gamma) / (1.0 - gamma)

    if torch.isnan(utility_term).any():
        raise ValueError("NaN encountered in utility_term")

    # Residual of the HJB equation
    residual = V_t + drift_term + diffusion_term + jump_term + utility_term - rho * V
    if torch.isnan(jump_term).any():
        raise ValueError("NaN encountered in jump_term")
    if torch.isnan(residual).any():
        raise ValueError("NaN encountered in residual")

    # PDE loss
    loss_pde = F.mse_loss(residual, torch.zeros_like(residual))

    # Optional supervised terminal condition loss
    loss_data = torch.tensor(0.0, device=device)
    if data_dict and "x_terminal" in data_dict:
        x_T = data_dict["x_terminal"].to(device)
        v_true = data_dict["v_terminal"].to(device)
        logV_pred = model(x_T)[0]  # only logV
        loss_data = F.mse_loss(logV_pred, v_true)

    # Regularizer to discourage extreme consumption
    reg_loss = 1e-3 * (c_hat ** 2).mean()

    # Total loss
    total_loss = (
        config["lambda_pde"] * loss_pde +
        config["lambda_data"] * loss_data +
        reg_loss
    )

    return total_loss, loss_pde, loss_data