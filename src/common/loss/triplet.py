"""
Since my implement is not working... this is another implement from reid task.

* see: https://github.com/JDAI-CV/fast-reid/blob/master/fastreid/modeling/losses/triplet_loss.py
* author:  xingyu liao
"""

import torch
import torch.nn.functional as F
from . import _reduct


@torch.jit.script
def euclidean_dist(x, y, eps: float = 1e-12):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy - 2 * torch.matmul(x, y.t())
    dist = dist.clamp(min=eps).sqrt()          # for numerical stability
    return dist


@torch.jit.script
def cosine_dist(x, y, eps: float = 1e-12):
    x = F.normalize(x, dim=1, eps=eps)
    y = F.normalize(y, dim=1, eps=eps)
    dist = 2 - 2 * torch.mm(x, y.t())
    return dist


def softmax_weights(dist, mask, eps=1e-12):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True)
    W = torch.exp(diff) * mask / Z.clamp(min=eps) # avoid division by zero
    return W


def hard_example_mining(dist_mat, is_pos, is_neg):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pair wise distance between samples, shape [N, M]
      is_pos: positive index with shape [N, M]
      is_neg: negative index with shape [N, M]
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N]
    dist_ap, _ = torch.max(dist_mat * is_pos, dim=1)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N]
    inf = dist_mat.max() + 1
    dist_an, _ = torch.min(dist_mat * is_neg + is_pos * inf, dim=1)

    return dist_ap, dist_an


def weighted_example_mining(dist_mat, is_pos, is_neg, eps=1e-12):
    """For each anchor, find the weighted positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      is_pos:
      is_neg:
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
    """
    assert len(dist_mat.size()) == 2

    is_pos = is_pos
    is_neg = is_neg
    dist_ap = dist_mat * is_pos
    dist_an = dist_mat * is_neg

    weights_ap = softmax_weights(dist_ap, is_pos, eps)
    weights_an = softmax_weights(-dist_an, is_neg, eps)

    dist_ap = torch.sum(dist_ap * weights_ap, dim=1)
    dist_an = torch.sum(dist_an * weights_an, dim=1)

    return dist_ap, dist_an


def distance_weighted_sampling(
    dist_mat,
    embed_dim,
    batch_k,
    cutoff=0.5,
    nonzero_loss_cutoff=1.4,
    eps=1e-8,
):
    assert len(dist_mat.size()) == 2

    N = dist_mat.size(0)
    dist_mat = dist_mat.clamp(min=cutoff)
    log_weights = ((2 - embed_dim) * dist_mat.log() + ((3 - embed_dim) / 2) *
                   (1 - (dist_mat ** 2) / 4).clamp(min=eps).log())            # -log q(d)

    mask = torch.ones_like(log_weights)
    for i in range(0, N, batch_k):
        mask[i:i + batch_k, i:i + batch_k] = 0

    mask_uniform_probs = mask.float() / (N - batch_k)

    weights = torch.exp(log_weights - torch.max(log_weights)) # ? maybe shift to <= 1
    weights = weights * mask * (
        dist_mat < nonzero_loss_cutoff
    )                                                         # clip to avoid noisy samples
    weights_sum = torch.sum(weights, dim=1, keepdim=True)     # [N, 1]
    weights = weights / weights_sum.clamp(min=eps)

    a_indices = []
    p_indices = []
    n_indices = []

    for i in range(N):
        block_idx = i // batch_k
        n_indices.append(
            torch.distributions.Categorical(
                weights[i] if weights_sum[i] != 0 else mask_uniform_probs[i]
            ).sample((batch_k - 1, ))
        )

        for j in range(block_idx * batch_k, (block_idx + 1) * batch_k):
            if j != i:
                a_indices.append(i)
                p_indices.append(j)

    n_indices = torch.cat(n_indices)

    return a_indices, dist_mat[a_indices], dist_mat[p_indices], dist_mat[n_indices]


def triplet_loss(
    embedding,
    targets,
    margin: float,
    norm_feat=False,
    mining='weightedexample',
    reduction='mean'
):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'.

    Args:
        embedding (Tensor[float]): [N, D]
        targets (Tensor[long]): [N]
        margin (float): [description]
        norm_feat (bool, optional): whether the feature is normalized. if so, use cos distance. else euclidean distance.
        mining (str, optional): Defaults to 'weightedexample'.
        reduction (str, optional): Defaults to 'mean'.

    Raises:
        ValueError: if mining is not matched

    Returns:
        Tensor: [description]
    """

    if norm_feat:
        dist_mat = cosine_dist(embedding, embedding)
    else:
        dist_mat = euclidean_dist(embedding, embedding)

    # For distributed training, gather all features from different process.
    # if comm.get_world_size() > 1:
    #     all_embedding = torch.cat(GatherLayer.apply(embedding), dim=0)
    #     all_targets = concat_all_gather(targets)
    # else:
    #     all_embedding = embedding
    #     all_targets = targets

    N = dist_mat.size(0)
    is_pos = (
        targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
    )
    is_neg = (
        targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()
    )

    if mining == 'hardmining':
        dist_ap, dist_an = hard_example_mining(dist_mat, is_pos, is_neg)
    elif mining == 'weightedexample':
        dist_ap, dist_an = weighted_example_mining(dist_mat, is_pos, is_neg)
    else:
        raise ValueError(mining)

    y = dist_an.new().resize_as_(dist_an).fill_(1)

    if margin > 0:
        loss = F.margin_ranking_loss(
            dist_an, dist_ap, y, margin=margin, reduction='none'
        )
    else:
        loss = F.soft_margin_loss(dist_an - dist_ap, y, reduction='none')
        if torch.any(torch.isinf(loss)):
            loss = F.margin_ranking_loss(
                dist_an, dist_ap, y, margin=0.3, reduction='none'
            )

    return _reduct(loss, reduction)            # loss: [N]


class WeightedExampleTripletLoss(torch.nn.Module):
    def __init__(self, margin: float, normalize=True, reduction='mean'):
        super().__init__()
        self.margin = margin
        self.normalize = normalize
        self.reduction = reduction

    def forward(self, embedding, target):
        # TODO: Sync among GPUs
        if self.normalize:
            embedding = F.normalize(embedding, dim=1)
        return triplet_loss(
            embedding, target, self.margin, self.normalize, 'weightedexample',
            self.reduction
        )
