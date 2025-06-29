# loss.py

import torch
import torch.nn.functional as F
from common import (
    compute_v_t, compute_v_w, compute_v_ww
)
from config import get_config


def compute_loss(model, batch, data_dict=None):
    """
    Compute total PINN loss for the single-asset GBM HJB problem.

    Args:
        model: PINN network mapping (t,W) -> V
        batch: Tensor of shape (N, 2) with columns [t, W]
        data_dict: optional dict with keys:
            'x_terminal': Tensor (M, 2) of [t=T, W]
            'v_terminal': Tensor (M, 1) of true V(T, W)

    Returns:
        total_loss: weighted sum of PDE and data losses
        loss_pde: PDE residual MSE
        loss_data: supervised terminal MSE

    Notes:
        The computed optimal controls ``pi_star`` and ``c_star`` are clamped
        to ``[-5, 5]`` and ``[1e-3, 100]`` respectively before being used in
        the utility terms.  This helps stabilise training by avoiding
        unrealistic values.
    """

    config = get_config()
    device = config["device"]

    # Move batch to device
    batch = batch.to(device)
    t = batch[:, 0:1].clone().detach().requires_grad_(True)
    W = batch[:, 1:2].clone().detach().requires_grad_(True)
    x = torch.cat([t, W], dim=1)

    # Forward pass
    logV = model(x)
    logV = torch.clamp(logV, min=-15.0, max=15.0)  # avoid exp overflow
    V = torch.exp(logV)
    if torch.isnan(V).any():
        raise ValueError("NaN encountered in V")


    # Compute derivatives
    V_t  = compute_v_t(model, x)
    V_W  = compute_v_w(model, x)
    V_WW = compute_v_ww(model, x)
    if torch.isnan(V_W).any() or torch.isnan(V_WW).any():
        raise ValueError("NaN encountered in V_W or V_WW")

    # Stabilize denominators
    eps = 1e-4
    V_WW = torch.clamp(V_WW, min=eps)
    V_W  = torch.clamp(V_W,  min=eps)

    # Compute optimal controls using the HJB first order conditions
    # dV/dW and d^2V/dW^2 can be negative so we do not clamp the sign here
    pi_star = - (config["mu"] - config["r"]) / (config["sigma"]**2) * (V_W / V_WW)

    c_star = V_W.pow(-1.0 / config["gamma"])
    c_star = torch.clamp(c_star, min=1e-3, max=100.0)

    # Clamp the controls to avoid extreme values
    pi_star = torch.clamp(pi_star, min=-5.0, max=5.0)
    c_star = torch.clamp(c_star, min=1e-3, max=100.0)

    # HJB residual
    drift_term     = (config["r"] * W + pi_star * (config["mu"] - config["r"]) - c_star) * V_W
    diffusion_term = 0.5 * (pi_star ** 2) * (config["sigma"] ** 2) * V_WW

    gamma = config["gamma"]
    if abs(gamma - 1.0) < 1e-3:
        utility_term = torch.log(c_star)
    else:
        utility_term = c_star.pow(1.0 - gamma) / (1.0 - gamma)
    if torch.isnan(utility_term).any():
        raise ValueError("NaN encountered in utility_term")
    residual = (V_t
                + drift_term
                + diffusion_term
                + utility_term
                - config["rho"] * V)
    if torch.isnan(residual).any():
        raise ValueError("NaN encountered in residual")

    # PDE loss
    loss_pde = F.mse_loss(residual, torch.zeros_like(residual))

    # Supervised terminal loss
    loss_data = torch.tensor(0.0, device=device)
    if data_dict is not None:
        if "x_terminal" in data_dict and "v_terminal" in data_dict:
            x_term = data_dict["x_terminal"].to(device)
            v_true = data_dict["v_terminal"].to(device)
            v_pred = model(x_term)
            loss_data = F.mse_loss(v_pred, v_true)

    # Total loss
    total_loss = config["lambda_pde"] * loss_pde + config["lambda_data"] * loss_data

    return total_loss, loss_pde, loss_data
