# derivatives.py

import torch
from torch import Tensor
from typing import Union

def extract_value_output(output: Union[Tensor, tuple]) -> Tensor:
    """
    Extract the log-value (V) output from a model.
    Supports models that return either a tensor or a tuple of (logV, pi, c).
    """
    if isinstance(output, tuple):
        return output[0]  # logV
    return output

def compute_grad(output: Tensor, x: Tensor, idx: int) -> Tensor:
    """
    Compute the first-order partial derivative ∂output/∂x_i.
    """
    y = output.squeeze(-1) if output.dim() > 1 else output
    grads = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True
    )[0]
    return grads[:, idx]

def compute_grad2(output: Tensor, x: Tensor, idx1: int, idx2: int) -> Tensor:
    """
    Compute the second-order partial derivative ∂²output/∂x_i∂x_j.
    """
    first_deriv = compute_grad(output, x, idx1)
    second_grads = torch.autograd.grad(
        outputs=first_deriv,
        inputs=x,
        grad_outputs=torch.ones_like(first_deriv),
        create_graph=True,
        retain_graph=True
    )[0]
    return second_grads[:, idx2]

# ---------- Named Derivative Accessors ----------

def compute_v_t(model: torch.nn.Module, x: Tensor) -> Tensor:
    V = extract_value_output(model(x))
    return compute_grad(V, x, idx=0)

def compute_v_w(model: torch.nn.Module, x: Tensor) -> Tensor:
    V = extract_value_output(model(x))
    return compute_grad(V, x, idx=1)

def compute_v_vi(model: torch.nn.Module, x: Tensor, i: int) -> Tensor:
    V = extract_value_output(model(x))
    return compute_grad(V, x, idx=2 + i)

def compute_v_ww(model: torch.nn.Module, x: Tensor) -> Tensor:
    V = extract_value_output(model(x))
    return compute_grad2(V, x, idx1=1, idx2=1)

def compute_v_wvi(model: torch.nn.Module, x: Tensor, i: int) -> Tensor:
    V = extract_value_output(model(x))
    return compute_grad2(V, x, idx1=1, idx2=2 + i)

def compute_v_vivj(model: torch.nn.Module, x: Tensor, i: int, j: int) -> Tensor:
    V = extract_value_output(model(x))
    return compute_grad2(V, x, idx1=2 + i, idx2=2 + j)