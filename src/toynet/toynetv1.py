'''
A toy implement for classifying benign/malignant and BIRADs

* author: JamzumSum
* create: 2021-1-11
'''
from math import log as mathlog

import torch
import torch.nn as nn
import torch.nn.functional as F
from common.utils import class_trans, cond_trans, freeze
from common.focal import focalBCE

from .discriminator import ConsistancyDiscriminator as CD
from .unet import UNet

assert hasattr(torch, 'amax')   # make sure amax is supported

def diverseMSE(CAM, Y, K=-1, a=1., reduction='mean'):
    fence_sitter = lambda x: 0.25 - ((x - .5) ** 2).mean(dim=(2, 3))
    towards_zero = lambda x: (x ** 2).mean(dim=(2, 3))
    mse = class_trans(a * fence_sitter(CAM), towards_zero(CAM), F.one_hot(Y, num_classes=K))
    if reduction == 'mean': return mse.mean()
    elif reduction == 'sum': return mse.sum()
    else: return mse

class BIRADsUNet(UNet):
    '''
    [N, ic, H, W] -> [N, 2, H, W], [N, K, H, W]
    '''
    def __init__(self, ic, ih, iw, K, fc=64, pi=0.01):
        UNet.__init__(self, ic, ih, iw, 2, fc)
        # Set out_class=2 since we have two classes: malignant and bengin
        # Note that thought B/M are exclusive, but P(M) + P(B) != 1, 
        # for there might be inputs that contain no tumor. 
        # But for similarity we may restrict that P(M) + P(B) = 1...
        # So softmax is accepted...
        self.BDW = nn.Conv2d(fc, K, 1)
        if pi: self.initLastConvBias(pi)

    def forward(self, X):
        '''
        X: [N, ic, H, W]
        return: 
        - B/M Class Activation Mapping  [N, 2, H, W]
        - BIRADs CAM                    [N, K, H, W]
        '''
        x9, Mhead = UNet.forward(self, X)
        Bhead = torch.sigmoid(self.BDW(torch.tanh(x9)))
        return Mhead, Bhead

    def initLastConvBias(self, pi=0.01):
        '''
        Initial strategy from RetinaNet. Hopes to stablize the training.
        '''
        b = mathlog(pi / (1 - pi))
        torch.nn.init.constant_(self.DW.bias, b)
        torch.nn.init.constant_(self.BDW.bias, b)

    def seperatedParameters(self):
        paramAll = self.parameters()
        paramB = self.BDW.parameters()
        paramM = (p for p in paramAll if id(p) not in [id(i) for i in paramB])
        return paramM, paramB

class ToyNetV1(nn.Module):
    support = ('hotmap')

    def __init__(self, ishape, K, patch_size, fc=64, pi=0.01):
        nn.Module.__init__(self)
        self.K = K
        self.backbone = BIRADsUNet(*ishape, K, fc, pi)
        self.pooling = nn.AvgPool2d(patch_size)

    def seperatedParameters(self):
        m, b = self.backbone.seperatedParameters()
        return m, b

    def forward(self, X):
        '''
        X: [N, ic, H, W]
        return: 
        - B/M Class Activation Mapping  [N, 2, H, W]
        - BIRADs CAM                    [N, H, W, K]
        - B/M prediction distrib.       [N, 2]
        - BIRADs prediction distrib.    [N, K]
        '''
        Mhead, Bhead = self.backbone(X)
        Mpatches = self.pooling(Mhead)      # [N, 2, H//P, W//P]
        Bpatches = self.pooling(Bhead)      # [N, K, H//P, W//P]

        Pm = torch.amax(Mpatches, dim=(2, 3))        # [N, 2]
        Pb = torch.amax(Bpatches, dim=(2, 3))        # [N, K]
        return Mhead, Bhead, Pm, Pb

    def _loss(self, X, Ym, Yb=None, piter=0., mweight=None, bweight=None):
        '''
        Protected for classes inherit from ToyNetV1.
        return: Original result, M-branch losses, B-branch losses.
        '''
        res = self.forward(X)
        M, B, Pm, Pb = res
        # ToyNetV1 does not constrain between the two CAMs
        # But may constrain on their own values, if necessary

        Mloss = focalBCE(Pm, Ym, K=2, gamma=1 + piter, weight=mweight)
        Mpenalty = diverseMSE(M, Ym, K=2)
        zipM = (Mloss, Mpenalty)

        if Yb is None: zipB = None
        else:
            Bloss = focalBCE(Pb, Yb, K=self.K, gamma=1 + piter, weight=bweight)
            Bpenalty = diverseMSE(B, Yb, self.K, a=piter)
            zipB = (Bloss, Bpenalty)

        return res, zipM, zipB

    def lossWithResult(self, *args, **argv):
        res = self._loss(*args, **argv)
        zipM, zipB = res[1:3]
        Mloss, Mpenalty = zipM
        summary = {
            'loss/malignant focal': Mloss.detach(), 
            'penalty/CAM_malignant': Mpenalty.detach()
        }
        loss = Mloss
        penalty = Mpenalty
        if zipB:
            Bloss, Bpenalty = zipB
            loss = loss + Bloss
            penalty = penalty + 0.5 * Bpenalty
            summary['loss/BIRADs focal'] = Bloss.detach()
            summary['penalty/CAM_BIRADs'] = Bpenalty.detach()
        return res, loss + 4 * penalty, summary

    def loss(self, *args, **argv):
        '''
        X: [N, ic, H, W]
        Ym: [N], long
        Yb: [N], long
        piter: float in (0, 1)
        mweight: bengin/malignant weights
        bweight: BIRADs weights
        '''
        return self.lossWithResult(*args, **argv)[1:]

class ToyNetV1D(ToyNetV1):
    support = ('hotmap', 'discriminator')

    def __init__(self, ishape, K, *args, **argv):
        ToyNetV1.__init__(self, ishape, K, *args, **argv)
        self.D = CD(K)

    def discriminatorParameters(self):
        return self.D.parameters()

    def discriminatorLoss(self, X, Ym, Yb, piter=0.):
        N = Ym.shape[0]
        with torch.no_grad():
            _, _, Pm, Pb = self.forward(X)
        loss = self.D.loss(Pm, Pb, torch.zeros(N, 1).to(X.device))
        loss = freeze(loss, piter ** 2)
        loss = loss + self.D.loss(
            Ym.unsqueeze(1), 
            F.one_hot(Yb, num_classes=self.K).type_as(Pb), 
            torch.ones(N, 1).to(X.device)
        )
        return loss

    def _loss(self, *args, **argv):
        res, zipM, zipB = ToyNetV1._loss(self, *args, **argv)
        _, _, Pm, Pb = res
        consistency = self.D.forward(Pm, Pb).mean()
        return res, zipM, zipB, (consistency, )

    def lossWithResult(self, *args, **argv):
        '''
        return: Original result, M-branch losses, B-branch losses, consistency.
        '''
        res, loss, summary = ToyNetV1.lossWithResult(self, *args, **argv)
        # But ToyNetV1 can constrain between the probability distributions Pm & Pb :D
        consistency = res[3][0]
        summary['consistency'] = consistency.detach()
        return res, loss + (1 - consistency), summary


if __name__ == "__main__":
    x = torch.randn(2, 1, 572, 572)
    toy = ToyNetV1(
        (1, 572, 572), 
        6, 12
    )
    loss, _ = toy.loss(x, torch.zeros(2, dtype=torch.long), torch.ones(2, dtype=torch.long))
    loss.backward()
