# integrals.py
"""
General utilities for computing nonlocal jump integral terms in HJB-PIDE losses.
Supports single-asset MJD, Bates (Heston+Jumps), and multi-asset jump models.
"""
import torch
from torch import Tensor, nn
from typing import Callable, Optional


def compute_jump_integral(
    model: Callable[[Tensor], Tensor],
    x: Tensor,
    pi: Tensor,
    config: dict,
    jump_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
    num_samples: int = 10
) -> Tensor:
    """
    Compute the jump integral term
      I[V](x) = λ E_J[V(t, x_jump) - V(t, x)].

    Now safe against tuple outputs and extreme jumps.
    """
    # Unpack jump parameters
    lambda_ = config.get("lambda") or config.get("lambda_jump")
    mu_J    = config["mu_J"]
    sigma_J = config["sigma_J"]

    batch_size = x.shape[0]
    device     = x.device

    # Ensure pi is shape (batch, d_assets)
    pi = pi.view(batch_size, -1)
    d_assets = pi.shape[1]

    # 1️⃣ Sample jumps J with clipping at ±4σ
    mu = torch.as_tensor(mu_J,    device=device)
    sd = torch.as_tensor(sigma_J, device=device)
    J  = torch.normal(mu, sd, size=(batch_size, num_samples, d_assets), device=device)
    J  = torch.clamp(J, min=mu - 4*sd, max=mu + 4*sd)

    # 2️⃣ Build jumped points
    x_jump = jump_fn(x, pi, J)  # shape: (batch*num_samples, input_dim)

    # 3️⃣ Evaluate logV at jumped points, extracting head[0] if tuple
    raw_jump = model(x_jump)
    logV_jump = raw_jump[0] if isinstance(raw_jump, tuple) else raw_jump
    logV_jump = torch.clamp(logV_jump, min=-15.0, max=15.0)
    V_jump    = torch.exp(logV_jump).view(batch_size, num_samples)

    # 4️⃣ Evaluate V at original points similarly
    raw = model(x)
    logV = raw[0] if isinstance(raw, tuple) else raw
    logV = torch.clamp(logV, min=-15.0, max=15.0)
    V    = torch.exp(logV).view(batch_size, 1)

    # 5️⃣ Monte-Carlo expectation
    integral = lambda_ * (V_jump.mean(dim=1, keepdim=True) - V)
    return integral.squeeze(1)


def mjd_jump_fn(x: Tensor, pi: Tensor, J: Tensor) -> Tensor:
    """
    Jump function for single-asset MJD:
        W' = W + pi * W * (exp(J) - 1)
    Args:
        x: (batch, 2) [t, W]
        pi: (batch, 1) or (batch,)
        J: (batch, num_samples, 1)
    Returns:
        x_jump: (batch*num_samples, 2)
    """
    t = x[:, 0:1]  # (batch,1)
    W = x[:, 1:2]  # (batch,1)
    pi = pi.view(-1, 1)
    # Jump multiplier
    M = torch.exp(J) - 1.0  # (batch, num_samples, 1)
    # Compute post-jump wealth
    W_jump = W.unsqueeze(1) + pi.unsqueeze(1) * W.unsqueeze(1) * M  # (batch, num_samples,1)
    # Flatten
    t_jump = t.unsqueeze(1).expand(-1, J.shape[1], -1).reshape(-1, 1)
    W_flat = W_jump.reshape(-1, 1)
    return torch.cat([t_jump, W_flat], dim=1)


def bates_jump_fn(x: Tensor, pi: Tensor, J: Tensor) -> Tensor:
    """
    Jump function for Bates model (Heston+Jumps):
    - Wealth jumps multiplicatively, volatility v unchanged.
    Args:
        x: (batch, 3) [t, W, v]
        pi: (batch, 1)
        J: (batch, num_samples, 1)
    Returns:
        x_jump: (batch*num_samples, 3)
    """
    t = x[:, 0:1]
    W = x[:, 1:2]
    v = x[:, 2:3]
    pi = pi.view(-1, 1)
    M = torch.exp(J) - 1.0  # (batch, num_samples,1)
    W_jump = W.unsqueeze(1) + pi.unsqueeze(1) * W.unsqueeze(1) * M
    # Flatten and replicate v
    t_jump = t.unsqueeze(1).expand(-1, J.shape[1], -1).reshape(-1, 1)
    W_flat = W_jump.reshape(-1, 1)
    v_flat = v.unsqueeze(1).expand(-1, J.shape[1], -1).reshape(-1, 1)
    return torch.cat([t_jump, W_flat, v_flat], dim=1)


def multi_asset_jump_fn(x: Tensor, pi: Tensor, J: Tensor) -> Tensor:
    """
    Jump function for multi-asset MJD/Bates:
        W' = W + W * sum_i(pi_i * (exp(J_i) - 1))
    Latent states in x (after W) are unchanged.
    Args:
        x: (batch, 2+z) [t, W, ...latent]
        pi: (batch, d_assets)
        J: (batch, num_samples, d_assets)
    Returns:
        x_jump: (batch*num_samples, 2+z)
    """
    t = x[:, 0:1]  # (batch,1)
    W = x[:, 1:2]  # (batch,1)
    latent = x[:, 2:]  # (batch, z)
    # Compute combined jump multiplier for wealth
    M = torch.exp(J) - 1.0  # (batch, num_samples, d)
    # pi: (batch, d) -> (batch,1,d)
    pi_exp = pi.unsqueeze(1)
    delta = (pi_exp * M).sum(dim=2, keepdim=True) * W.unsqueeze(1)  # (batch, num_samples,1)
    W_jump = W.unsqueeze(1) + delta
    # Flatten
    batch, n, _ = W_jump.shape
    t_jump = t.unsqueeze(1).expand(-1, n, -1).reshape(-1, 1)
    W_flat = W_jump.reshape(-1, 1)
    # replicate latent
    latent_flat = latent.unsqueeze(1).expand(-1, n, -1).reshape(-1, latent.shape[1])
    return torch.cat([t_jump, W_flat, latent_flat], dim=1)
