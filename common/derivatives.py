# derivatives.py

import torch
from torch import Tensor

def compute_grad(
    output: Tensor,
    x: Tensor,
    idx: int
) -> Tensor:
    """
    Compute the first-order partial derivative ∂output/∂x_i.

    Args:
        output: Tensor of shape (batch_size, 1) or (batch_size,).
        x: Input tensor of shape (batch_size, input_dim) with requires_grad=True.
        idx: Index of the input dimension to differentiate with respect to.

    Returns:
        Tensor of shape (batch_size,) containing ∂output/∂x[:, idx].
    """
    # Flatten output to shape (batch_size,)
    y = output.squeeze(-1) if output.dim() > 1 else output
    # Compute gradients d(output)/d(x)
    grads = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True
    )[0]  # grads has shape (batch_size, input_dim)
    # Return the requested component
    return grads[:, idx]

def compute_grad2(
    output: Tensor,
    x: Tensor,
    idx1: int,
    idx2: int
) -> Tensor:
    """
    Compute the second-order partial derivative ∂²output/∂x_i∂x_j.

    Args:
        output: Tensor of shape (batch_size, 1) or (batch_size,).
        x: Input tensor of shape (batch_size, input_dim) with requires_grad=True.
        idx1: First derivative index.
        idx2: Second derivative index.

    Returns:
        Tensor of shape (batch_size,) containing ∂²output/∂x[:, idx1]∂x[:, idx2].
    """
    # First derivative w.r.t. idx1
    first_deriv = compute_grad(output, x, idx1)
    # Compute second derivative w.r.t. idx2
    second_grads = torch.autograd.grad(
        outputs=first_deriv,
        inputs=x,
        grad_outputs=torch.ones_like(first_deriv),
        create_graph=True,
        retain_graph=True
    )[0]
    return second_grads[:, idx2]

# ---------- Named derivative accessors ----------

def compute_v_t(model: torch.nn.Module, x: Tensor) -> Tensor:
    """
    Compute ∂V/∂t at inputs x.
    Assumes x[:, 0] corresponds to time t.
    """
    V = model(x)
    return compute_grad(V, x, idx=0)

def compute_v_w(model: torch.nn.Module, x: Tensor) -> Tensor:
    """
    Compute ∂V/∂W at inputs x.
    Assumes x[:, 1] corresponds to wealth W.
    """
    V = model(x)
    return compute_grad(V, x, idx=1)

def compute_v_vi(model: torch.nn.Module, x: Tensor, i: int) -> Tensor:
    """
    Compute ∂V/∂v^i at inputs x.
    Assumes latent state v^i is at index 2 + i.
    """
    V = model(x)
    return compute_grad(V, x, idx=2 + i)

def compute_v_ww(model: torch.nn.Module, x: Tensor) -> Tensor:
    """
    Compute ∂²V/∂W² at inputs x.
    """
    V = model(x)
    return compute_grad2(V, x, idx1=1, idx2=1)

def compute_v_wvi(model: torch.nn.Module, x: Tensor, i: int) -> Tensor:
    """
    Compute mixed partial ∂²V/∂W∂v^i at inputs x.
    """
    V = model(x)
    return compute_grad2(V, x, idx1=1, idx2=2 + i)

def compute_v_vivj(model: torch.nn.Module, x: Tensor, i: int, j: int) -> Tensor:
    """
    Compute mixed partial ∂²V/∂v^i∂v^j at inputs x.
    """
    V = model(x)
    return compute_grad2(V, x, idx1=2 + i, idx2=2 + j)
