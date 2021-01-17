'''
A little improvement on ToyNetV1
Make use of the CAMs.

* author: JamzumSum
* create: 2021-1-11
'''

from toynetv1 import ToyNetV1, focalCE
import torch
import torch.nn as nn
import torch.nn.functional as F

class JointEstimator(nn.Sequential):
    '''
    Use P(Malignant) and P(BIRADs=k) to estimate joint distribution P(Malignant, BIRADs=k)
    '''
    def __init__(self, K):
        nn.Sequential(
            self, 
            nn.Conv2d(K + 1, K, 1), 
            nn.Sigmoid()
        )
    def forward(self, M, B):
        '''
        M: [N, 2, H, W]
        B: [N, K, H, W]
        O: [N, K, H, W], which represents P(malignant, BIRADs=k)
        '''
        M = M[:, 1:2]   # get the malignant confidence channel, [N, 1, H, W]
        return nn.Sequential.forward(self, torch.cat([M, B], dim=1))

class ToyNetV2(ToyNetV1):
    def __init__(self, ishape, K, patch_size, fc=64, a=1., b=0.5):
        ToyNetV1.__init__(self, ishape, K, patch_size, fc, a)
        self.Q = JointEstimator(K)
        self.b = b

    def loss(self, X, Ym, Yb=None, piter=0.):
        '''
        X: [N, ic, H, W]
        Ym: [N], long
        Yb: [N], long
        piter: current iter times / total iter times
        '''
        M, B, Pm, Pb = self.forward(X)
        Mloss = focalCE(torch.cat([1 - Pm, Pm], dim=-1), Ym, gamma=2 * piter, weight=self.mbalance)
        
        # M needs softmax for mlaignant/bengin exclude each other even if multi tumors are detected.
        # But B need no softmax in fact for BIRADs can be actually all the classes the tumors are.
        # But for simplifying the problem, BIRADs classification is treated as single-class task now.
        M = torch.softmax(M, dim=1)
        B = torch.softmax(B, dim=1)
        Mpos, Mneg = torch.split(M, [1, 1], dim=1)

        Qpos = self.Q(M, B)        # [N, K, H, W]
        Qneg = B - Qpos
        mal_info = Qpos * torch.log2(Qpos / Mpos / B)   # [N, K, H, W]
        ben_info = Qneg * torch.log2(Qneg / Mneg / B)   # [N, K, H, W]
        infoLoss = torch.asum(ben_info, dim=(1, 2, 3)) + torch.asum(mal_info, dim=(1, 2, 3))    # [N]
        infoLoss = torch.mean(infoLoss)
        warmup = self.b * torch.exp(-5 * (1 - piter))

        summary = {
            'loss/malignant entropy': Mloss.detach(), 
            'likelihood/mutual info': infoLoss.detach(),
            'coefficency/gaussian warm-up b': warmup
        }
        if Yb is None: Bloss = 0
        else:
            Bloss = F.cross_entropy(Pb, Yb, weight=self.bbalance)
            summary['loss/BIRADs focal'] = Bloss.detach()
        return Mloss + self.a * Bloss - warmup * infoLoss, 