"""
create synthetic data 
"""

# load packages
import numpy as np
from numpy.random import multivariate_normal
from numpy.random import uniform
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn

# ============== data ==============


def XORdata():
    """a simple 4 point XOR dataset"""
    X = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    X.requires_grad_(False)
    Y = torch.Tensor([0, 1, 1, 0]).view(-1, 1)
    Y.requires_grad_(False)
    return X, Y


def noisyXORdata(numsamples=400):
    """noisy rendering of XOR"""
    newnoisy = torch.zeros((numsamples, 2))
    newnoisy.requires_grad_(False)
    newnoisylabels = torch.zeros((numsamples))
    newnoisylabels.requires_grad_(False)

    for i, g in enumerate(newnoisy):
        if i < int(numsamples / 4):
            newnoisy[i] = torch.tensor(
                multivariate_normal((-1, -1), ((0.15, 0), (0, 0.15)))
            )
            newnoisylabels[i] = 0
        if i > int(numsamples / 4) - 1 and i < int(numsamples / 2):
            newnoisy[i] = torch.tensor(
                multivariate_normal((-1, 1), ((0.15, 0), (0, 0.15)))
            )
            newnoisylabels[i] = 1
        if i > int(numsamples / 2) - 1 and i < 3 * int(numsamples / 4):
            newnoisy[i] = torch.tensor(
                multivariate_normal((1, -1), ((0.15, 0), (0, 0.15)))
            )
            newnoisylabels[i] = 1
        if i > (3 * int(numsamples / 4)) - 1:
            newnoisy[i] = torch.tensor(
                multivariate_normal((1, 1), ((0.15, 0), (0, 0.15)))
            )
            newnoisylabels[i] = 0

    X, Y = unison_shuffled_copies(newnoisy, newnoisylabels)
    return X, Y


def sindata(numsamples=400, seed=42):
    np.random.seed(seed)  # for reproducibility
    sin = torch.zeros((numsamples, 2))
    sin.requires_grad_(False)
    sinlabels = torch.zeros((numsamples))
    sinlabels.requires_grad_(False)

    for i, g in enumerate(sin):
        samp = np.array((uniform(low=-1, high=1), uniform(low=-1, high=1)))
        sin[i] = torch.tensor(samp)

        # * high freq # .6*np.sin(7*samp[0]-1)
        # * low freq # .6*np.sin(3*samp[0]+1.2)
        if samp[1] > 0.6 * np.sin(7 * samp[0] - 1):
            sinlabels[i] = 1

    X, Y = unison_shuffled_copies(sin, sinlabels)
    return X, Y


def noisy_sindata_train_test(numsamples=400, seed=42, std=1, test_size=0.2):
    """noisy version of the sinusoidal data"""
    X, Y = sindata(numsamples, seed)
    # train test split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, stratify=Y, random_state=seed
    )
    # perturbe
    X_train = X_train + torch.normal(0, std, size=X_train.shape)
    return X_train, X_test, Y_train, Y_test


def sindata_full():
    """full set of data on a uniform grid in [-1.5, 1.5] * [-1.5, 1.5]"""
    # generate data
    l = torch.linspace(-1.5, 1.5, 40)
    y = torch.linspace(-1.5, 1.5, 40)
    sin = torch.cartesian_prod(l, y)
    sin.requires_grad_(False)

    # assign labless
    # * high freq # .6*np.sin(7*samp[0]-1)
    # * low freq # .6*np.sin(3*samp[0]+1.2)
    sinlabels = sin[:, 1] > 0.6 * torch.sin(7 * sin[:, 0] - 1)
    sinlabels = sinlabels.long()
    sinlabels.requires_grad_(False)

    X, Y = unison_shuffled_copies(sin, sinlabels)
    return X, Y


# ========= utils ==========


def unison_shuffled_copies(a, b):
    """concurrently shuffle two arrays"""
    assert len(a) == len(b), "arrays not of the same length"
    p = np.random.permutation(len(a))
    return a[p], b[p]
