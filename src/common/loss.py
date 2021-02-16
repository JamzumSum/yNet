"""
A collection for custom losses. 

* author: JamzumSum
* create: 2021-1-29
"""

import torch
import torch.nn.functional as F
from torch import Tensor

transequal = lambda x: x == x.T


def focalCE(P, Y, gamma=2.0, *args, **argv):
    """
    focal loss for classification. nll_loss implement
    - P: [N, K] NOTE: not softmax-ed
    - Y: [N]    NOTE: long
    - gamma: 
    """
    pt = P
    gms = (1 - pt) ** gamma  # [N, K]
    return F.nll_loss(gms * pt.log(), Y, *args, **argv)


def focalBCE(P, Y, gamma=2.0, *args, **argv):
    """
    focal loss for classification. BCELoss implement
    - P: [N, K] NOTE: not softmax-ed when K != 1
    - Y: [N]    NOTE: long
    - gamma: 
    """
    # NOTE: softmax is troubling for both branches.
    # ADD: softmax is now forbiddened.
    Y = F.one_hot(Y, num_classes=P.size(1)).float()
    bce = F.binary_cross_entropy(P, Y, reduction="none", *args, **argv)
    pt = torch.exp(-bce)  # [N, K]
    gms = (1 - pt) ** gamma  # [N, K]
    return (gms * bce).mean()


def _pairwise_distance(embeddings, squared=False):
    """
        计算两两embedding的距离
        ------------------------------------------
        Args：
            embedding: 特征向量， 大小（N, D
            squared:   是否距离的平方，即欧式距离
        Returns：
            distances: 两两embeddings的距离矩阵，大小 （N, N）
    """
    # 矩阵相乘,得到（N, N），因为计算欧式距离|a-b|^2 = a^2 -2ab + b^2,
    # 其中 ab 可以用矩阵乘表示
    dot_product = embeddings @ embeddings.T
    # dot_product对角线部分就是 每个embedding的平方
    square_norm = dot_product.diag()
    # |a-b|^2 = a^2 - 2ab + b^2
    # tf.expand_dims(square_norm, axis=1)是（N, 1）大小的矩阵，减去 （N, N）大小的矩阵，相当于每一列操作
    distances = square_norm.unsqueeze(1) - 2 * dot_product + square_norm.unsqueeze(0)
    distances[distances < 0] = 0  # 小于0的距离置为0
    if not squared:
        distances = distances.sqrt()
    return distances


def _masked_minimum(data: Tensor, mask):
    axis_maximums = data.max(1, keepdims=True).values
    return (mask * (data - axis_maximums)).min(1, keepdims=True).values + axis_maximums


def _masked_maximum(data: Tensor, mask):
    axis_minimums = data.min(1, keepdims=True).values
    return (mask * (data - axis_minimums)).max(1, keepdims=True).values + axis_minimums


def semihard_triplet_loss(labels: Tensor, embeddings: Tensor, margin=1.0):
    """Computes the triplet loss with semi-hard negative mining.
    The loss encourages the positive distances (between a pair of embeddings with
    the same labels) to be smaller than the minimum negative distance among
    which are at least greater than the positive distance plus the margin constant
    (called semi-hard negative) in the mini-batch. If no such negative exists,
    uses the largest negative distance instead.
    See: https://arxiv.org/abs/1503.03832.
    Args:
      labels: 1-D torch.int64 `Tensor` with shape [batch_size] of
        multiclass integer labels.
      embeddings: 2-D float `Tensor` of embedding vectors. NOTE: Embeddings should
        be l2 normalized.
      margin: Float, margin term in the loss definition.
    Returns:
      triplet_loss: torch.float32 scalar.
    """
    # Reshape [N] label tensor to a [N, 1] label tensor.
    assert labels.dim() == 1
    N = labels.size(0)
    labels = labels.unsqueeze(1)  # [N, 1]
    # Build pairwise squared distance matrix.
    pdist_matrix = _pairwise_distance(embeddings, squared=True)  # [N, N]
    # Build pairwise binary adjacency matrix.
    adjacency = transequal(labels.repeat(1, N))  # [N, N]
    # Invert so we can select negatives only.
    adjacency_not = torch.logical_not(adjacency)

    # Compute the mask.
    pdist_matrix_tile = pdist_matrix.repeat(N, 1)  # [N*N, N]
    mask = pdist_matrix_tile > pdist_matrix.T.reshape(-1, 1)  # [N*N, N]
    mask = torch.logical_and(mask, adjacency_not.repeat(N, 1))  # [N*N, N]

    mask_final = (mask.sum(1, keepdims=True) > 0).reshape(N, N).T  # [N, N]

    # negatives_outside: smallest D_an where D_an > D_ap.
    negatives_outside = (
        _masked_minimum(pdist_matrix_tile, mask).reshape(N, N).T
    )  # [N, N]

    # negatives_inside: largest D_an.
    negatives_inside = _masked_maximum(pdist_matrix, adjacency_not)  # [N, 1]
    negatives_inside = negatives_inside.repeat(1, N)  # [N, N]
    semi_hard_negatives = torch.where(
        mask_final, negatives_outside, negatives_inside
    )  # [N, N]
    loss_mat = margin + pdist_matrix - semi_hard_negatives  # [N, N]
    loss_mat[loss_mat < 0] = 0
    mask_positives = adjacency.float() - torch.eye(
        N, device=embeddings.device
    )  # [N, N]

    # In lifted-struct, the authors multiply 0.5 for upper triangular
    #   in semihard, they take all positive pairs except the diagonal.
    num_positives = torch.sum(mask_positives)
    triplet_loss = torch.sum(loss_mat * mask_positives) / num_positives
    return triplet_loss


class SemiHardTripletLoss(torch.nn.Module):
    def __init__(self, margin=1.0, normalize=True):
        torch.nn.Module.__init__(self)
        self.margin = margin
        self.normalize = normalize

    def forward(self, embedding: Tensor, label: Tensor):
        if self.normalize:
            norm = torch.linalg.norm(embedding, dim=1, keepdim=True)  # [N]
            embedding = embedding / norm  # [N, D]
        return semihard_triplet_loss(label, embedding, self.margin)


def diceCoefficient(p, gt, eps=1e-5, activation="none", reduction="mean"):
    r""" computational formula：
        dice = (2 * tp) / (2 * tp + fp + fn)
    """

    if activation is None or activation == "none":
        pass
    elif activation == "sigmoid":
        p = torch.sigmoid(p)
    elif activation == "softmax2d":
        p = F.softmax(p, 1)
    else:
        raise ValueError(activation)

    N = gt.size(0)
    pflat = p.view(N, -1)
    gt_flat = gt.view(N, -1)

    TP = torch.sum(gt_flat * pflat, dim=1)
    FP = torch.sum(pflat, dim=1) - TP
    FN = torch.sum(gt_flat, dim=1) - TP
    dice = (2 * TP + eps) / (2 * TP + FP + FN + eps)
    if reduction == "sum":
        return dice.sum()
    elif reduction == "mean":
        return dice.mean()
    elif reduction == "none":
        return dice
    else:
        raise ValueError(reduction)

