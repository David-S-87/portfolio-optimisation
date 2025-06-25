# config.py

import torch
from common import uniform_sampler

def get_config():
    """
    Returns all parameters, samplers, and paths for GBM PINN training.
    """

    # -------------------------------
    # Model parameters (GBM + CRRA utility)
    # -------------------------------
    mu = 0.10        # expected return of the risky asset
    sigma = 0.20     # volatility of the risky asset
    r = 0.02         # risk-free rate
    gamma = 3.0      # CRRA utility coefficient (risk aversion)
    rho = 0.05       # subjective discount rate
    T = 1.0          # time horizon (years)

    # -------------------------------
    # Domain bounds for collocation points
    # -------------------------------
    W_min, W_max = 0.1, 5.0
    bounds = {
        "t": (0.0, T),
        "W": (W_min, W_max),
    }

    def sampler(batch_size: int) -> torch.Tensor:
        """
        Uniformly sample 'batch_size' points (t, W) from the domain.
        """
        return uniform_sampler(bounds, batch_size)

    # -------------------------------
    # Training hyperparameters
    # -------------------------------
    config = {
        "mu": mu,
        "sigma": sigma,
        "r": r,
        "gamma": gamma,
        "rho": rho,
        "T": T,
        "bounds": bounds,
        "sampler": sampler,
        "batch_size": 128,
        "epochs": 5000,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "optimizer_name": "adam",
        "lambda_pde": 1.0,
        "lambda_data": 0.1,
        "grad_clip": 1.0,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "log_every": 100,
        "save_every": 500,
        "checkpoint_path": "single_asset/gbm/checkpoints/gbm_epoch_{epoch}.pt",
        "eval_resolution": 100,
    }

    return config
