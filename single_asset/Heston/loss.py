# loss.py for single_asset/Heston

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
from config import get_config

def compute_loss(model, batch, data_dict=None):
    """
    Compute total PINN loss for the single-asset Heston portfolio optimisation.

    Args:
        model: neural network approximating log V(t, W, v)
        batch: Tensor of shape (N, 3) with columns [t, W, v]
        data_dict: optional dict with keys:
            'x_terminal': Tensor (M, 3) of [T, W, v]
            'v_terminal': Tensor (M, 1) of log V(T, W)

    Returns:
        total_loss: weighted sum of PDE, data, and regularization losses
        loss_pde: HJB residual MSE
        loss_data: terminal MSE
    """
    # Load configuration
    config = get_config()
    device = config["device"]

    # Prepare inputs
    x = batch.to(device).clone().detach().requires_grad_(True)
    t = x[:, 0:1]
    W = x[:, 1:2]
    v = x[:, 2:3]

    # Model prediction: log V for numerical stability
    logV = model(x)
    logV = torch.clamp(logV, min=-20.0, max=20.0)
    V = torch.exp(logV)

    # Compute derivatives via autograd
    V_t  = compute_v_t(model, x)
    V_W  = compute_v_w(model, x)
    V_v  = compute_v_vi(model, x, 0)
    V_WW = compute_v_ww(model, x)
    V_vv = compute_v_vivj(model, x, 0, 0)
    V_Wv = compute_v_wvi(model, x, 0)

    # Stabilize derivatives to avoid division by zero
    V_W  = torch.clamp(V_W,  min=1e-4)
    V_WW = torch.clamp(V_WW, min=1e-4)

    # Extract model parameters
    mu    = config["mu"]
    r     = config["r"]
    gamma = config["gamma"]
    rho   = config["rho"]
    kappa = config["kappa"]
    theta = config["theta"]
    xi    = config["xi"]
    corr  = config["corr"]

    # Optimal controls (analytic)
    pi_star = ((mu - r) * V_W + xi * corr * v * V_Wv) / (v * V_WW)

    V_W  = torch.clamp(V_W,  min=1e-2, max=1e2)  # tighten this clamp
    c_star = V_W.pow(-1.0 / gamma)
    c_star = torch.clamp(c_star, max=10.0)       # restrict range of consumption



    # HJB residual terms
    drift_W   = (r * W + pi_star * (mu - r) - c_star) * V_W
    drift_v   = kappa * (theta - v) * V_v
    diff_WW   = 0.5 * (pi_star ** 2) * v * V_WW
    diff_vv   = 0.5 * (xi ** 2) * v * V_vv
    cross     = pi_star * xi * corr * v * V_Wv
    utility   = c_star.pow(1.0 - gamma) / (1.0 - gamma)

    residual = (
        V_t + drift_W + drift_v + diff_WW + diff_vv + cross + utility
        - rho * V
    )
    loss_pde = F.mse_loss(residual, torch.zeros_like(residual))

    # Supervised terminal condition loss (log V)
    loss_data = torch.tensor(0.0, device=device)
    if data_dict is not None and "x_terminal" in data_dict:
        x_term = data_dict["x_terminal"].to(device)
        v_true = data_dict["v_terminal"].to(device)
        v_pred = model(x_term)
        loss_data = F.mse_loss(v_pred, v_true)

    # Regularization on consumption to penalize extreme values
    reg_coeff = 1e-3
    reg_loss = reg_coeff * c_star.pow(2).mean()

    # Weighted total loss
    total_loss = (
        config["lambda_pde"] * loss_pde
        + config["lambda_data"] * loss_data
        + reg_loss
    )

    return total_loss, loss_pde, loss_data