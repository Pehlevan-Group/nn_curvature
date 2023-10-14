"""
analytic volume elements and curvature computations (for 2D inputs only)
"""

# load packages
from typing import Callable

import torch
import torch.nn as nn

# ============= collection of closed form derivatives ============
def derivative_selector(nl: str) -> Callable:
    """select the derivative function by name of non-linearity (avoid eval)"""
    if nl == "Sigmoid":
        return Sigmoid_analytic_derivative
    elif nl == "Erf":
        return Erf_analytic_derivative
    elif nl == "ReLU":
        return lambda x: (x > 0).float()
    else:
        raise NotImplementedError(
            f"derivative of nl {nl} not implemented in closed-form"
        )


def hessian_selector(nl: str) -> Callable:
    """select the hessian function by name of the non-linearity"""
    if nl == "Sigmoid":
        return Sigmoid_analytic_hessian
    elif nl == "Erf":
        return Erf_analytic_hessian
    else:
        raise NotImplementedError(
            f"derivative of nl {nl} not implemented in closed-form"
        )


def Sigmoid_analytic_derivative(x: torch.Tensor) -> torch.Tensor:
    """the analytic derivative of the sigmoid function"""
    nl = nn.Sigmoid()
    nl_result = nl(x)
    der = nl_result * (1 - nl_result)
    return der


def Sigmoid_analytic_hessian(x: torch.Tensor) -> torch.Tensor:
    """the analytic hessian of the sigmoid function"""
    nl = nn.Sigmoid()
    nl_result = nl(x)
    hes = nl_result * (1 - nl_result) * (1 - 2 * nl_result)
    return hes


def Erf_analytic_derivative(x: torch.Tensor) -> torch.Tensor:
    """the analytic derivative of the error function"""
    der = (2 / torch.pi) ** (1 / 2) * torch.exp(-x.square() / 2)
    return der


def Erf_analytic_hessian(x: torch.Tensor) -> torch.Tensor:
    """the analytic hessian of the error function"""
    hes = (2 / torch.pi) ** (1 / 2) * (-x) * torch.exp(-x.square() / 2)
    return hes


# ============== analytic metrics =================
def determinant_analytic(x: torch.Tensor, W: torch.Tensor, b: torch.Tensor, nl: str):
    """
    the analytic determinant

    :param x: input tensor (d = 2)
    :param W: weight matrix n by d
    :param b: bias vector n by 1
    :param nl: the nonlinearity, specified by a string
    """
    # prepare
    n = W.shape[0]
    # preactivation
    z = x @ W.T + b  # number of scans by n
    der_func = derivative_selector(nl)
    activated_z = der_func(z)
    activated_square = activated_z.square()

    # precompute m
    m = W[:, [0]] @ W[:, [1]].T - W[:, [1]] @ W[:, [0]].T
    m_squared = m.square()

    # O(n^2) einsum enhanced (divided by two since each added twice and diagonal are zeros)
    results = (
        torch.einsum("jk,nj,nk->n", m_squared, activated_square, activated_square) / 2
    )
    results = results / n**2
    results = torch.sqrt(results)
    return results


def ricci_analytic(x: torch.Tensor, W: torch.Tensor, b: torch.Tensor, nl: str):
    """
    the analytic ricci curvature

    :param x: input tensor (d = 2)
    :param W: weight matrix n by d
    :param b: bias vector n by 1
    :param nl: the nonlinearity, specified by a string
    """
    # prepare
    n = W.shape[0]
    z = x @ W.T + b  # number of scans by n
    der_func, hes_func = derivative_selector(nl), hessian_selector(nl)
    activated_z_der = der_func(z)
    activated_z_der_squared = activated_z_der.square()
    activated_z_hes = hes_func(z)

    # precompute m
    m = W[:, [0]] @ W[:, [1]].T - W[:, [1]] @ W[:, [0]].T

    # O(n^3), einsum enhanced
    ricci = torch.einsum(
        "jk,ij,ik,jk,ni,nj,nk,nj,nk->n",
        m,
        m,
        m,
        m,
        activated_z_der_squared,
        activated_z_der,
        activated_z_der,
        activated_z_hes,
        activated_z_hes,
    )

    # determinant returned sqrt (raised to the fourth since determinant is sqrt)
    ricci = ricci * -3 / (n**3 * determinant_analytic(x, W, b, nl) ** 4)
    return ricci
