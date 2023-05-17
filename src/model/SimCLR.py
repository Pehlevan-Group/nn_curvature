"""
Contrastive Learning pipeline 
1. resnet backbone + MLP, no pretraining
2. use backbone output to perform classification and assign labels. 
"""

# load packages
from typing import Tuple

import torch
import torch.nn as nn


class SimCLR(nn.Module):
    """simple contrastive learning"""

    def __init__(self, backbone: nn.Module, bs: int, nl: nn.Module = nn.GELU()):
        """
        :param backbone: the feature map stage
        :param bs: the batch size
        :param nl: the nonlinearity in the projection head
        """
        super().__init__()
        # feature map
        self.feature_map = backbone.feature_map
        # projection head
        in_feat_dim = backbone.linear.in_features
        self.proj_head = nn.Sequential(
            nn.Linear(in_feat_dim, in_feat_dim), nl, backbone.linear
        )

        # prepare
        self.mask = self._get_mask(bs)
        self.label_match = self._get_label_match(bs)

    def _get_label_match(self, bs: int) -> torch.BoolTensor:
        """
        prepare label match matrix by batch size

        :return a bool matrix with 1 indicating coming from the same image and 0 otherwise
        """
        labels = torch.cat([torch.arange(bs) for _ in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).bool()
        mask = self._get_mask(bs)
        label_match = labels[~mask].view(labels.shape[0], -1)
        return label_match

    def _get_mask(self, bs: int) -> torch.BoolTensor:
        """prepare mask by batch size"""
        mask = torch.eye(bs * 2, dtype=torch.bool, requires_grad=False)
        return mask

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """feature map + projection head"""
        features = self.feature_map(X)
        projections = self.proj_head(features)
        return projections

    def nce_loss(self, projections: torch.Tensor) -> Tuple[float]:
        """contrastive loss"""
        # get similarities
        projection_normalized = projections / projections.norm(dim=1, keepdim=True)
        similarity_matrix = projection_normalized @ projection_normalized.T

        # discard the main diagonal from both: labels and similarities matrix
        similarity_matrix = similarity_matrix[~self.mask].view(
            similarity_matrix.shape[0], -1
        )

        # select and combine multiple positives
        positives = similarity_matrix[self.label_match.bool()].view(
            self.label_match.shape[0], -1
        )

        # select only the negatives the negatives
        negatives = similarity_matrix[~self.label_match.bool()].view(
            similarity_matrix.shape[0], -1
        )

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        # logits = logits / self.args.temperature
        return logits, labels
