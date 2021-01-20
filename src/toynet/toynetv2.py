'''
A little improvement on ToyNetV1
Make use of the CAMs.

* author: JamzumSum
* create: 2021-1-11
'''

from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
from common.utils import freeze

from .toynetv1 import ToyNetV1, focalCE


class JointEstimator(nn.Sequential):
    '''
    Use P(Malignant) and P(BIRADs=k) to estimate joint distribution P(Malignant, BIRADs=k)
    '''
    def __init__(self, K):
        nn.Sequential(
            self, 
            nn.Conv2d(K + 1, K - 1, 1), 
            nn.Sigmoid()
        )
    def forward(self, M, B):
        '''
        M: [N, 1, H, W]
        B: [N, K, H, W]
        O: [N, K, H, W], which represents P(malignant, BIRADs=k)
        '''
        K = B.shape[1]
        sym = torch.cat([M, B], dim=1)                          # [N, K + 1, H, W]
        rel = nn.Sequential.forward(self, sym)                  # [N, K - 1, H, W]
        rel = rel * B[:, :K - 1]    # scale so that each x_i is in [0, b_i]. 

        rel_lz = rel / rel.sum(dim=1, keepdim=True) * M.expand_as(rel)      # [N, K - 1, H, W]
        mask = torch.sum(rel, dim=1, keepdim=True) <= M         # [N, K - 1, H, W]
        rel = rel * mask + (not mask) * rel_lz.unsqueeze(dim=1) # [N, K - 1, H, W]

        add = M - torch.sum(rel, dim=1, keepdim=True)           # [N, K, H, W]
        rel = torch.cat([rel, add], dim=1)                      # [N, K, H, W]

        assert rel.min() >= 0
        assert torch.all(rel <= B)
        return rel

class ToyNetV2(ToyNetV1):
    def __init__(self, ishape, K, patch_size, fc=64, b=0.5):
        ToyNetV1.__init__(self, ishape, K, patch_size, fc)
        self.Q = JointEstimator(K)
        self.b = b

    def seperatedParameters(self):
        paramM, paramB = self.backbone.seperatedParameters()
        return paramM, chain(paramB, self.Q.parameters())

    def loss(self, X, Ym, Yb=None, piter=0.):
        '''
        X: [N, ic, H, W]
        Ym: [N], long
        Yb: [N], long
        piter: current iter times / total iter times
        '''
        M, B, Pm, Pb = self.forward(X)
        Mloss = focalCE(torch.cat([1 - Pm, Pm], dim=-1), Ym, gamma=2 * piter, weight=self.mbalance)
        
        # M needs softmax for malignant/bengin exclude each other even if multiple tumors are detected.
        # But B needs no softmax in fact for BIRADs can be actually all the classes the tumors are.
        # But for simplifying the problem, BIRADs classification is treated as single-class task now.
        M = torch.softmax(M, dim=1)
        B = torch.softmax(B, dim=1)
        M = M[:, 1:2]   # get channel of class 1. [N, 1, H, W]

        Q = self.Q(M, B)                                # [N, K, H, W]
        info = Q * torch.log(Q / B / M.expand_as(B))    # [N, K, H, W]
        info = torch.asum(info, dim=(1, 2, 3))          # [N]
        info = torch.mean(info)
        warmup = self.b * torch.exp(-5 * (1 - piter))

        summary = {
            'loss/malignant CE': Mloss.detach(), 
            'likelihood/mutual info': info.detach()
        }
        if Yb is None: Bloss = 0
        else:
            Bloss = F.cross_entropy(Pb, Yb, weight=self.bbalance)
            summary['loss/BIRADs CE'] = Bloss.detach()
        return Mloss + (Bloss) - warmup * info, summary
