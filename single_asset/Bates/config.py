# config.py

import torch
from common.utils import uniform_sampler, set_seed


def get_config():
    """
    Returns configuration dictionary for single-asset Bates model PINN training.
    Includes Heston stochastic volatility and Merton jumps.
    """
    # -------------------------------
    # 1️⃣ Model parameters
    # -------------------------------
    # Asset dynamics
    mu      = 0.10   # expected return of risky asset
    r       = 0.02   # risk-free rate

    # CRRA utility & discounting
    gamma   = 3.0    # risk aversion
    rho     = 0.05   # subjective discount rate
    T       = 1.0    # time horizon (years)

    # Heston volatility parameters
    kappa   = 2.0    # mean reversion speed
    theta   = 0.04   # long-run variance
    xi      = 0.3    # vol of vol
    rho_corr = -0.7  # correlation between asset and variance

    # Jump parameters (Merton)
    lambda_ = 1.0    # jump intensity
    mu_J    = -0.10  # mean of log-jump returns
    sigma_J = 0.30   # std dev of log-jump returns

    # -------------------------------
    # 2️⃣ Domain bounds & sampler
    # -------------------------------
    W_min, W_max = 0.1, 5.0
    v_min, v_max = 0.01, 1.0
    bounds = {
        "t": (0.0, T),
        "W": (W_min, W_max),
        "v": (v_min, v_max),
    }

    def sampler(batch_size: int) -> torch.Tensor:
        """
        Uniformly sample collocation points in (t, W, v).
        Returns a Tensor of shape (batch_size, 3).
        """
        return uniform_sampler(bounds, batch_size)

    # -------------------------------
    # 3️⃣ Training hyperparameters
    # -------------------------------
    config = {
        # Model parameters
        "mu": mu,
        "r": r,
        "gamma": gamma,
        "rho": rho,
        "T": T,
        # Heston vol params
        "kappa": kappa,
        "theta": theta,
        "xi": xi,
        "rho_corr": rho_corr,
        # Jump params
        "lambda": lambda_,
        "mu_J": mu_J,
        "sigma_J": sigma_J,
        # Domain & sampler
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
        # Checkpoints & logging
        "checkpoint_path": "single_asset/Bates/checkpoints/bates_epoch_{epoch}.pt",
        "log_every": 100,
        "save_every": 500,
        "eval_resolution": 50,
    }

    # -------------------------------
    # 4️⃣ Device & reproducibility
    # -------------------------------
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(42)

    return config
