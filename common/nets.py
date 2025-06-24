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
