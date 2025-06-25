# nets.py

import torch
import torch.nn as nn


def get_activation(name: str) -> nn.Module:
    """
    Return a PyTorch activation function corresponding to the given name.

    Args:
        name: Activation name ('tanh', 'relu', 'softplus').

    Returns:
        A PyTorch activation module.
    """
    name = name.lower()
    if name == 'tanh':
        return nn.Tanh()
    elif name == 'relu':
        return nn.ReLU()
    elif name == 'softplus':
        return nn.Softplus()
    else:
        raise ValueError(f"Unsupported activation: {name}")


class PINN(nn.Module):
    """
    General-purpose Physics-Informed Neural Network for HJB problems.

    Input:
      - A tensor x of shape (batch_size, input_dim), where
          input_dim = 1 (time) + 1 (wealth) + d (model-specific latent states).
    Output:
      - A tensor of shape (batch_size, output_dim), representing V(t, W, ...).

    Args:
        input_dim (int): Number of input features.
        hidden_dim (int): Width of each hidden layer.
        output_dim (int): Output dimension (default=1 for scalar V).
        n_hidden_layers (int): Number of hidden layers.
        activation (str): Activation to use ('tanh', 'relu', 'softplus').
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 1,
        n_hidden_layers: int = 4,
        activation: str = "tanh"
    ):
        super(PINN, self).__init__()
        act_fn = get_activation(activation)
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(act_fn)

        # Hidden layers
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act_fn)

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        return self.model(x)

    def count_parameters(self) -> int:
        """
        Count the number of trainable parameters in the model.

        Returns:
            Total trainable parameter count.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PINNWithControls(nn.Module):
    """
    Extended PINN that also predicts optimal controls π and c alongside the value V.

    Architecture:
      - Shared feature extractor: MLP mapping x -> hidden representation h
      - Three heads:
          V_head: outputs scalar V(t,W,...)
          pi_head: outputs scalar π(t,W,...)
          c_head: outputs scalar c(t,W,...)

    Args:
        input_dim (int): Dimension of input features.
        hidden_dim (int): Width of hidden layers.
        n_hidden_layers (int): Number of hidden layers in feature extractor.
        activation (str): Activation function name.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        n_hidden_layers: int = 4,
        activation: str = "tanh"
    ):
        super(PINNWithControls, self).__init__()
        act_fn = get_activation(activation)
        feature_layers = []

        # Shared feature extractor
        feature_layers.append(nn.Linear(input_dim, hidden_dim))
        feature_layers.append(act_fn)
        for _ in range(n_hidden_layers):
            feature_layers.append(nn.Linear(hidden_dim, hidden_dim))
            feature_layers.append(act_fn)

        self.feature_extractor = nn.Sequential(*feature_layers)

        # Output heads: scalar outputs
        self.V_head = nn.Linear(hidden_dim, 1)
        self.pi_head = nn.Linear(hidden_dim, 1)
        self.c_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        """
        Forward pass returning value and controls.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Tuple of tensors (V, pi, c), each of shape (batch_size, 1).
        """
        h = self.feature_extractor(x)
        V = self.V_head(h)
        pi = self.pi_head(h)
        c = self.c_head(h)
        return V, pi, c

    def count_parameters(self) -> int:
        """
        Count trainable parameters across all heads and feature extractor.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
