'''
A toy implement for classifying benign/malignant and BIRADs

* author: JamzumSum
* create: 2021-1-11
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.utils import freeze

from .discriminator import ConsistancyDiscriminator as CD
from .unet import UNet

assert hasattr(torch, 'amax')   # make sure amax is supported

def focalCE(P, Y, gamma=2., *args, **argv):
    '''
    focal loss for classification. nll_loss implement
    - P: [N, K] NOTE: not softmax-ed
    - Y: [N]    NOTE: long
    - gamma: 
    '''
    pt = torch.softmax(P, dim=-1)   # [N, K]
    gms = (1 - pt) ** gamma         # [N, K]
    return F.nll_loss(gms * pt.log(), Y, *args, **argv)

def focalBCE(P, Y, gamma=2., K=-1, weight=None):
    '''
    focal loss for classification. BCELoss implement
    - P: [N, K] NOTE: not softmax-ed when K != 1
    - Y: [N]    NOTE: long
    - gamma: 
    '''
    if K == 1:
        # This is a sigmoid output that needn't softmax.
        if weight is not None: weight = weight[Y]
        bce = F.binary_cross_entropy(P.squeeze(1), Y.float(), weight=weight, reduction='none')
    else:
        # This is a 'pre-distribution' that needs softmax.
        Y = F.one_hot(Y, num_classes=K).float()
        bce = F.binary_cross_entropy_with_logits(P, Y, pos_weight=weight, reduction='none')
    pt = torch.exp(-bce)            # [N, K]
    gms = (1 - pt) ** gamma         # [N, K]
    return (gms * pt).mean()

class BIRADsUNet(UNet):
    '''
    [N, ic, H, W] -> [N, 1, H, W], [N, K, H, W]
    '''
    def __init__(self, ic, ih, iw, K, fc=64):
        UNet.__init__(self, ic, ih, iw, 1, fc)
        self.BDW = nn.Conv2d(fc, K, 1)

    def forward(self, X):
        '''
        X: [N, ic, H, W]
        return: 
        - benign/malignant Class Activation Mapping     [N, 1, H, W]
        - BIRADs CAM                    [N, H, W, K]
        '''
        x9, Mhead = UNet.forward(self, X)
        Bhead = torch.sigmoid(self.BDW(torch.tanh(x9)))
        return Mhead, Bhead

    def seperatedParameters(self):
        paramAll = self.parameters()
        paramB = self.BDW.parameters()
        paramM = (p for p in paramAll if id(p) not in [id(i) for i in paramB])
        return paramM, paramB

class ToyNetV1(nn.Module):
    hotmap = True

    def __init__(self, ishape, K, patch_size, fc=64):
        nn.Module.__init__(self)
        self.K = K
        self.backbone = BIRADsUNet(*ishape, K, fc)
        self.pooling = nn.AvgPool2d(patch_size)
        self.D = CD(K)

    def to(self, *args, **argv):
        self.mbalance = self.mbalance.to(*args, **argv)
        self.bbalance = self.bbalance.to(*args, **argv)
        super(ToyNetV1, self).to(*args, **argv)

    def seperatedParameters(self):
        m, d = self.backbone.seperatedParameters()
        return m, d, self.D.parameters()

    def forward(self, X):
        '''
        X: [N, ic, H, W]
        return: 
        - benign/malignant Class Activation Mapping     [N, 1, H, W]
        - BIRADs CAM                    [N, H, W, K]
        - malignant confidence          [N, 1]
        - BIRADs prediction vector      [N, K]
        '''
        Mhead, Bhead = self.backbone(X)
        Mpatches = self.pooling(Mhead)      # [N, 1, H//P, W//P]
        Bpatches = self.pooling(Bhead)      # [N, K, H//P, W//P]

        Pm = torch.amax(Mpatches, dim=(2, 3))        # [N, 1]
        Pb = torch.amax(Bpatches, dim=(2, 3))        # [N, K]
        return Mhead, Bhead, Pm, Pb

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

    def _loss(self, X, Ym, Yb=None, piter=0., mweight=None, bweight=None):
        '''
        Protected for classes inherit from ToyNetV1.
        return: Original result, M-branch losses, B-branch losses, consistency.
        '''
        res = self.forward(X)
        M, B, Pm, Pb = res
        # ToyNetV1 does not constrain between the two CAMs
        # But may constrain on their own values, if necessary
        penaltyfunc = lambda x: 0.25 - ((x - .5) ** 2).mean()

        Mloss = focalBCE(Pm, Ym, K=1, gamma=2 * piter, weight=mweight)
        Mpenalty = penaltyfunc(M)
        zipM = (Mloss, Mpenalty)
        # But ToyNetV1 can constrain between the probability distributions Pm & Pb.
        consistency = self.D.forward(Pm, Pb).mean()

        if Yb is None: zipB = None
        else:
            Bloss = focalBCE(Pb, Yb, K=self.K, gamma=2 * piter, weight=bweight)
            Bpenalty = penaltyfunc(B)
            zipB = (Bloss, Bpenalty)

        return res, zipM, zipB, consistency

    def lossWithResult(self, *args, **argv):
        res = self._loss(*args, **argv)
        _, zipM, zipB, consistency = res
        Mloss, Mpenalty = zipM
        summary = {
            'loss/malignant focal': Mloss.detach(), 
            'penalty/CAM_malignant': Mpenalty.detach(), 
            'consistency': consistency.detach()
        }
        loss = Mloss
        penalty = Mpenalty
        if zipB:
            Bloss, Bpenalty = zipB
            loss += Bloss
            penalty += 0.5 * Bpenalty
            summary['loss/BIRADs focal'] = Bloss.detach()
            summary['penalty/CAM_BIRADs'] = Bpenalty.detach()
        return res, loss + 4 * penalty + (1 - consistency), summary

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

if __name__ == "__main__":
    x = torch.randn(2, 1, 572, 572)
    toy = ToyNetV1(
        (1, 572, 572), 
        6, 12
    )
    loss, _ = toy.loss(x, torch.zeros(2, dtype=torch.long), torch.ones(2, dtype=torch.long))
    loss.backward()
