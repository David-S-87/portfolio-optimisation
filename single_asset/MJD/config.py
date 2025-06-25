# config.py for single_asset/MJD

import torch
from common import uniform_sampler, set_seed


def get_config():
    """
    Returns configuration dictionary for single-asset Merton Jump-Diffusion (MJD) PINN training.
    """
    # -------------------------------
    # 1️⃣ Model Parameters
    # -------------------------------
    mu       = 0.10    # expected return of the risky asset
    sigma    = 0.20    # diffusion volatility
    r        = 0.02    # risk-free rate
    gamma    = 3.0     # CRRA risk aversion
    rho      = 0.05    # subjective discount rate
    T        = 1.0     # time horizon (years)

    # Jump-diffusion parameters
    lambda_  = 1.0     # jump intensity (Poisson rate)
    mu_J     = -0.10   # mean of log-jump returns
    sigma_J  = 0.30    # std dev of log-jump returns

    # -------------------------------
    # 2️⃣ Domain Bounds & Sampler
    # -------------------------------
    W_min, W_max = 0.1, 5.0
    bounds = {
        "t": (0.0, T),
        "W": (W_min, W_max),
    }

    def sampler(batch_size: int) -> torch.Tensor:
        """
        Uniformly sample (t, W) collocation points.
        """
        return uniform_sampler(bounds, batch_size)

    # -------------------------------
    # 3️⃣ Training Hyperparameters
    # -------------------------------
    config = {
        # Model parameters
        "mu": mu,
        "sigma": sigma,
        "r": r,
        "gamma": gamma,
        "rho": rho,
        "T": T,

        # Jump parameters
        "lambda": lambda_,
        "mu_J": mu_J,
        "sigma_J": sigma_J,

        # Domain
        "bounds": bounds,
        "sampler": sampler,

        # Training settings
        "batch_size": 128,
        "epochs": 5000,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "optimizer_name": "adam",
        "grad_clip": 1.0,

        # Loss weights
        "lambda_pde": 1.0,
        "lambda_data": 0.1,

        # Logging & checkpoints
        "checkpoint_path": "single_asset/mjd/checkpoints/mjd_epoch_{epoch}.pt",
        "log_every": 100,
        "save_every": 500,
        "eval_resolution": 100,
    }

    # -------------------------------
    # 4️⃣ Device & Reproducibility
    # -------------------------------
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    # For reproducibility
    set_seed(42)

    return config
