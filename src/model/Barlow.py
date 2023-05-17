"""
Barlow twins with ResNet backbone

code adapted from https://github.com/facebookresearch/barlowtwins/blob/main/main.py
"""

# load packages
from typing import List

import torch 
import torch.nn as nn 

class BarlowTwins(nn.Module):
    def __init__(
            self, 
            backbone: nn.Module, 
            bs: int, 
            projector: List[int], 
            lambd: float,
            nl: nn.Module = nn.GELU(),
        ):
        """
        :param backbone: the feature map stage 
        :parma bs: batchsize
        :param projector: the projector dimensions
        :param nl: the nonlinearity in the projection head
        :param lambd: the multiplier for the alignment term
        """
        super().__init__()
        self.feature_map = backbone.feature_map
        self.batch_size = bs 
        self.lambd = lambd

        # projector
        sizes = [backbone.linear.in_features] + list(projector)
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nl)
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, y1, y2):
        z1 = self.projector(self.feature_map(y1))
        z2 = self.projector(self.feature_map(y2))

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.batch_size)
        # torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        n = c.shape[0]
        off_diag_c = c.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
        off_diag = off_diag_c.pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss
