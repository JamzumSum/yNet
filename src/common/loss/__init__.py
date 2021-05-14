"""
A collection for custom losses. 

* author: JamzumSum
* create: 2021-1-29
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def smoothed_label(target, smoothing=0.0, K=-1):
    # type: (Tensor, float, int) -> Tensor
    assert 0.0 <= smoothing <= 0.5
    one = 1.0 - smoothing
    zero = smoothing / (K - 1)
    onehot = F.one_hot(target, num_classes=K)
    return onehot.float().clamp_(zero, one)


def _reduct(r: Tensor, reduction: str):
    """reduce a tensor to a single value

    Args:
        r (Tensor): [any shape]
        reduction (str): `none`/`mean`/`sum`

    Raises:
        ValueError: if reduction is not valid

    Returns:
        Tensor: [single value]
    """
    if reduction == "none":
        return r
    elif reduction == "sum":
        return r.sum()
    elif reduction == "mean":
        return r.mean()
    else:
        raise ValueError(reduction)


@torch.jit.script
def focal_smooth_bce(P, Y, gamma=2.0, smooth=0.0, weight=None, reduction="mean"):
    # type: (Tensor, Tensor, float, float, Optional[Tensor], str) -> Tensor
    """
    focal bce combined with label smoothing.
        P: [N, K] NOTE: not activated, e.g. softmaxed or sigmoided.
        Y: [N]    NOTE: int
        gamma: that in focal loss. e.g. gamma=0 is just label smooth loss.
        smooth: that in label smoothing. e.g. smooth=0 is just focal loss.
    other args are like those in cross_entropy.
    """
    K = P.size(1)
    YK = smoothed_label(Y, smooth, K)
    bce = F.binary_cross_entropy_with_logits(P, YK, weight=weight, reduction="none")
    pt = torch.exp(-bce)           # [N, K]
    gms = (1 - pt) ** gamma        # [N, K]
    return _reduct(gms * bce, reduction)


@torch.jit.script
def focal_smooth_ce(P, Y, gamma=2.0, smooth=0.0, weight=None, reduction="mean"):
    # type: (Tensor, Tensor, float, float, Optional[Tensor], str) -> Tensor
    """focal ce combined with label smoothing.

    Args:
        P (Tensor): [N, K]  NOTE: NOT activated, i.e. logits
        Y (Tensor): [N]     NOTE: long
        gamma (float, optional): that in focal loss. e.g. gamma=0 is just label smooth loss. Defaults to 2.0.
        smooth (float, optional): that in label smoothing. e.g. smooth=0 is just focal loss. Defaults to 0.0.
        weight (Tensor, optional): [description]. Defaults to None.
        reduction (str, optional): [description]. Defaults to "mean".

    Returns:
        Tensor: [description]
    """
    K = P.size(1)
    YK = smoothed_label(Y, smooth, K)
    ce = -YK * P.log_softmax(1)    # [N, K]

    pt = torch.exp(-ce)            # [N, K]
    gms = (1 - pt) ** gamma        # [N, K]

    if weight is not None:
        weight = weight / torch.sum(weight)
        ce = ce * weight
    ce = (ce * gms).sum(dim=1)     # [N]
    if reduction == "mean" and weight is not None:
        batchweight = (weight * YK).sum(dim=1)
        return ce.sum() / batchweight.sum()
    else:
        return _reduct(ce, reduction)


@torch.jit.script
def diceCoefficient(p, gt, eps: float = 1e-5, reduction: str = "mean"):
    """computational formula：
        dice = (2 * tp) / (2 * tp + fp + fn)]

    Args:
        p (Tensor): [description]
        gt (Tensor): [description]
        eps (float, optional): [description]. Defaults to 1e-5.
        reduction (str, optional): [description]. Defaults to "mean".

    Returns:
        Tensor: [description]
    """
    N = gt.size(0)
    pflat = p.view(N, -1)
    gt_flat = gt.view(N, -1)

    TP = torch.sum(gt_flat * pflat, dim=1)
    FP = torch.sum(pflat, dim=1) - TP
    FN = torch.sum(gt_flat, dim=1) - TP
    dice = (2 * TP).clamp(min=eps) / (2 * TP + FP + FN).clamp(min=eps)

    return _reduct(dice, reduction)


@torch.jit.script
def confidence(X: Tensor, reduction: str = 'mean') -> Tensor:
    """Calcuate the confidence of an output tensor. The more its values are close to 0&1, the more confident it is. 
    While the more its values are close to 0.5, the less confident it is.

    Args:
        X (Tensor): [Any shape]

    Returns:
        Tensor: 0~1
    """
    X = 4 * (X - 0.5) ** 2
    X = X.flatten(1).mean(1)
    return _reduct(X, reduction)
