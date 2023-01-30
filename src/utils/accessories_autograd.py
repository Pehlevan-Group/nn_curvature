"""
auxiliary functions: get curvatures 
"""

# load packages
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.autograd.functional as fnc

# constants
TOLS = 1e-8

# # takes two points in plane, eg torch.tensor([[0.5,0.5]],requires_grad=True)
# def kernel(x: torch.Tensor, xprime: torch.Tensor, mod: nn.Module) -> torch.Tensor:
#     """
#     get kernel by a given model (takes two points in plane, eg torch.tensor([[0.5,0.5]],requires_grad=True))
#     :parma x, xprime: row vectors
#     :parma mod: a model with lin1 and width attribute implmented
#     """
#     hx, hxprime = torch.sigmoid(mod.lin1(x)), torch.sigmoid(mod.lin1(xprime))

#     # implements dot-product row-wise btw each hx and hxprime
#     # takes (dim,2),(dim,2) to (dim,1)

#     # need to normalize by the width to prevent gradient blow-up with large width
#     # apparently this is the fastest way to batch dot products
#     return (1 / mod.width) * (hx * hxprime).sum(-1).unsqueeze(-1)


# def metric(x: torch.Tensor, mod: nn.Module) -> torch.Tensor:
#     """compute metric tensor"""
#     return 0.5 * fnc.hessian(
#         lambda g: kernel(g, g, mod), x, create_graph=True
#     ) - fnc.hessian(lambda g: kernel(x, g, mod), x, create_graph=True)


# def expansion(x: torch.Tensor, mod: nn.Module) -> torch.Tensor:
#     """quantify local expansion"""
#     return torch.sqrt(torch.det(metric(x, mod).squeeze()))


# # this deviates from the old one by ~.2 (order 10^3 smaller than magnitudes of the tensor)
# def christoffels(x: torch.Tensor, mod: nn.Module) -> torch.Tensor:
#     """
#     chirtoffel symbols
#     this deviates from the old one by ~.2 (order 10^3 smaller than magnitudes of the tensor)
#     """
#     met = metric(x, mod).squeeze()
#     # supports batch; returns shape (dim,2,2)... note that this returns bad inverse bc met usually not invertible... check torch.eig (met, True)
#     metinv = torch.inverse(met)
#     x.requires_grad = True

#     deriv = fnc.jacobian(lambda g: metric(g, mod), x).squeeze()

#     firstterm = torch.einsum("...in,...knj->...ijk", metinv, deriv)
#     secondterm = torch.einsum("...in,...jnk->...ijk", metinv, deriv)
#     thirdterm = torch.einsum("...in,...njk->...ijk", metinv, deriv)

#     return 0.5 * (firstterm + secondterm - thirdterm)


# def ricci(x: torch.Tensor, mod: nn.Module) -> torch.Tensor:
#     """ricci tensor"""
#     chris = christoffels(x, mod)

#     deriv = fnc.jacobian(lambda g: christoffels(g, mod), x).squeeze()

#     first = torch.einsum("...iijk->...jk", deriv)
#     second = torch.einsum("...jiki->...jk", deriv)
#     third = torch.einsum("...iip,...pjk->...jk", chris, chris)
#     fourth = torch.einsum("...ijp,...pik->...jk", chris, chris)

#     return first - second + third - fourth


# def scalcurvfromric(x: torch.Tensor, mod: nn.Module) -> torch.Tensor:
#     """ricci scalar"""
#     met = metric(x, mod).squeeze()
#     metinv = torch.inverse(met)

#     ric = ricci(x, mod)

#     return torch.einsum("...mn,...mn->", metinv, ric)


# ? why doesn't this work ?
# def regularized_inverse(met):
#     """compute regularized matrix inverse"""
#     b, m, n = met.shape
#     L, Q = torch.linalg.eigh(met)
#     L_clipeed = L.clamp(min=1e-5)
#     L_diag = torch.cat([torch.diag(1 / x) for x in L_clipeed], dim=0).reshape(b, m, n)
#     met_inverse = Q @ L_diag @ Q.transpose(-1, -2)
#     return met_inverse


# ========== batched version ===========


def batch_jacobian(f, x):
    """
    efficient jacobian computation of feature map f with respect to input x

    the output is of shape (feature_dim, batch_size, *input_dim)
    For example, if input x is (2, 10), then output is (feature_dim, 2, 10)
                 if input x is (2, 3, 32, 32), then the output is (feature_dim, 2, 3, 32, 32)
    """
    f_sum = lambda x: torch.sum(f(x), axis=0)
    return fnc.jacobian(f_sum, x, create_graph=True)


def batch_hessian(f, x):
    f_sum = lambda x: torch.sum(f(x), axis=0)
    return torch.sum(fnc.hessian(f_sum, x, create_graph=True), axis=2)


def kernel(
    x: torch.Tensor, xprime: torch.Tensor, feature_map: nn.Module
) -> torch.Tensor:
    """
    get kernel by a given model (takes two points in plane, eg torch.tensor([[0.5,0.5]],requires_grad=True))
    :parma x, xprime: row vectors
    :parma feature_map: the feature map defined by the neural network
    """
    # feature maps
    hx, hxprime = feature_map(x), feature_map(xprime)
    # implements dot-product row-wise btw each hx and hxprime
    # takes (dim,2), (dim,2) to (dim,1)

    # need to normalize by the width to prevent gradient blow-up with large width
    # apparently this is the fastest way to batch dot products
    width = hx.shape[1]  # normalized by its shape
    k = (1 / width) * (hx * hxprime).sum(-1, keepdim=True)
    return k


def metric(x: torch.Tensor, feature_map: nn.Module) -> torch.Tensor:
    """
    compute metric tensor

    :param x: the query point
    :param feature_map: the feature map defined by the neural network
    """
    # # computed through kernel
    # tensor = 0.5 * batch_hessian(
    #     lambda g: kernel(g, g, feature_map), x
    # ) - batch_hessian(lambda g: kernel(x, g, feature_map), x)

    # computed directly from feature map
    J = batch_jacobian(feature_map, x).flatten(
        start_dim=2
    )  # starting from input dimensions
    width = J.shape[0]
    met = J.permute(1, 2, 0) @ J.permute(1, 0, 2) / width  # manual normalization
    return met


def expansion(x: torch.Tensor, feature_map: nn.Module) -> torch.Tensor:
    """quantify local expansion"""
    result = torch.sqrt(torch.det(metric(x, feature_map).squeeze()))
    return result


def effective_expansion(
    x: torch.Tensor, feature_map: nn.Module, k: int = 6, thr: float = -8
) -> torch.Tensor:
    """
    to prevent underflow, we take the top eigenvalues and compute the effective expansion by the 1 / 2 log(sum of the top k eigenvalues)
    computed by singular value decomposition for efficiency

    :param k: the number of eigenvalues
    :param thr: the threshold below which to drop singular values
    """
    # from metric
    # met = metric(x, feature_map).squeeze()
    # # extract largest six eigenvalues per batch
    # eigvals_by_batch, _ = torch.lobpcg(met, k=k, largest=True)
    # effective_volume = eigvals_by_batch.log10().sum(dim=-1) / 2
    # return effective_volume

    # directly from jacobian (time efficient)
    J = batch_jacobian(feature_map, x).flatten(
        start_dim=2
    )  # flatten starting from the input dim
    width = J.shape[0]
    J = J.permute(1, 2, 0) / width ** (1 / 2)  # manual normalization
    svdvals = torch.linalg.svdvals(J)
    log_svdvals = svdvals[:, :k].log10()

    # thresholding
    mask = (log_svdvals >= thr).float()
    vol_elements_masked = log_svdvals * mask
    eff_vol_elements = vol_elements_masked.sum(dim=-1)

    return eff_vol_elements


def christoffels(x: torch.Tensor, feature_map: nn.Module) -> torch.Tensor:
    """
    chirtoffel symbols
    this deviates from the old one by ~.2 (order 10^3 smaller than magnitudes of the tensor)

    :param feature_map: the feature map defined by the neural network
    """
    met = metric(x, feature_map).squeeze()
    # supports batch; returns shape (dim,2,2)... note that this returns bad inverse
    # bc met usually not invertible...
    # * this method can not (for now be salvaged) ... It is numerially unstable whatsoever ...
    metinv = torch.linalg.pinv(
        met, hermitian=True, rtol=TOLS
    )  # set hermitian to true to discard small eigenvalues
    # x.requires_grad = True

    deriv = (
        batch_jacobian(lambda g: metric(g, feature_map), x)
        .squeeze()
        .permute(2, 1, 0, 3)
    )

    firstterm = torch.einsum("...din,...dknj->...dijk", metinv, deriv)
    secondterm = torch.einsum("...din,...djnk->...dijk", metinv, deriv)
    thirdterm = torch.einsum("...din,...dnjk->...dijk", metinv, deriv)

    result = 0.5 * (firstterm + secondterm - thirdterm)
    return result


def ricci(x: torch.Tensor, feature_map: nn.Module) -> torch.Tensor:
    """ricci tensor"""
    chris = christoffels(x, feature_map)

    deriv = (
        batch_jacobian(lambda g: christoffels(g, feature_map), x)
        .squeeze()
        .permute(3, 0, 1, 2, 4)
    )

    first = torch.einsum("...diijk->...djk", deriv)
    second = torch.einsum("...djiki->...djk", deriv)
    third = torch.einsum("...diip,...dpjk->...djk", chris, chris)
    fourth = torch.einsum("...dijp,...dpik->...djk", chris, chris)

    result = first - second + third - fourth
    return result


def scalcurvfromric(x: torch.Tensor, feature_map: nn.Module) -> torch.Tensor:
    """ricci scalar"""
    met = metric(x, feature_map).squeeze()
    # metinv = torch.inverse(met)
    metinv = torch.linalg.pinv(met, hermitian=True, rtol=TOLS)
    # metinv = regularized_inverse(met)

    ric = ricci(x, feature_map)

    result = torch.einsum("...dmn,...dmn->d", metinv, ric)
    return result


# ============= unused =============
# ##### batch functions not currently in use
# def batch_jacobian(f, x):
#     f_sum = lambda x: torch.sum(f(x), axis=0)
#     return fnc.jacobian(f_sum, x,create_graph=True)

# def batch_hessian(f, x):
#     f_sum = lambda x: torch.sum(f(x), axis=0)
#     return torch.sum(fnc.hessian(f_sum, x,create_graph=True),axis=2)
# #####
