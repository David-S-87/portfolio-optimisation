# config.py

import torch
from common import uniform_sampler

def get_config():
    """
    Returns all parameters, samplers, and paths for Heston PINN training.
    """

    # -----------------------------------------
    # Model parameters (Heston dynamics + CRRA)
    # -----------------------------------------
    mu     = 0.10      # expected return
    r      = 0.02      # risk-free rate
    gamma  = 3.0       # CRRA risk aversion
    rho    = 0.05      # subjective discount rate
    T      = 1.0       # time horizon

    # Heston parameters
    kappa  = 2.0       # mean reversion speed
    theta  = 0.04      # long-run variance
    xi     = 0.3       # volatility of variance
    corr   = -0.7      # correlation rho between asset and variance

    # -----------------------------------------
    # Domain bounds for collocation points
    # -----------------------------------------
    W_min, W_max = 0.5, 2.0
    v_min, v_max = 0.01, 0.25
    bounds = {
        "t": (0.0, T),
        "W": (W_min, W_max),
        "v": (v_min, v_max)
    }

    def sampler(batch_size: int) -> torch.Tensor:
        """
        Uniformly sample (t, W, v) points from the domain.
        """
        return uniform_sampler(bounds, batch_size)

    # -----------------------------------------
    # Training hyperparameters
    # -----------------------------------------
    config = {
        "mu": mu,
        "r": r,
        "gamma": gamma,
        "rho": rho,
        "T": T,
        "kappa": kappa,
        "theta": theta,
        "xi": xi,
        "corr": corr,
        "bounds": bounds,
        "sampler": sampler,
        "batch_size": 128,
        "epochs": 5000,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "optimizer_name": "adam",
        "lambda_pde": 1.0,
        "lambda_data": 0.1,   # lighter terminal penalty to avoid domination
        "grad_clip": 1.0,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "log_every": 100,
        "save_every": 500,
        "checkpoint_path": "C:/Users/david/BathUni/MA50290_24/single_asset/heston/heston_epoch_{epoch}.pt",
        "eval_resolution": 100,
    }

    return config
