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
    focal loss for classification.
    - P: [N, K] NOTE: not softmax-ed
    - Y: [N]    NOTE: long
    - gamma: 
    '''
    gms = torch.pow(torch.softmax(1 - P, dim=-1), gamma)    # [N, K]
    logs = torch.log_softmax(P, dim=-1)                     # [N, K]
    return torch.nn.functional.nll_loss(gms * logs, Y, *args, **argv)

class BIRADsUNet(UNet):
    '''
    [N, ic, H, W] -> [N, 2, H, W], [N, K, H, W]
    '''
    def __init__(self, ic, ih, iw, K, fc=64):
        UNet.__init__(self, ic, ih, iw, 2, fc)
        self.BDW = nn.Conv2d(fc, K, 1)

    def forward(self, X):
        '''
        X: [N, ic, H, W]
        return: 
        - benign/malignant Class Activation Mapping     [N, 2, H, W]
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
    mbalance = torch.Tensor([0.3, 0.7])
    bbalance = torch.Tensor([0.13, 0.19, 0.16, 0.1, 0.14, 0.28])
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
        - benign/malignant Class Activation Mapping     [N, 2, H, W]
        - BIRADs CAM                    [N, H, W, K]
        - malignant confidence          [N, 2]
        - BIRADs prediction vector      [N, K]
        '''
        Mhead, Bhead = self.backbone(X)
        Mpatches = self.pooling(Mhead)      # [N, 2, H//P, W//P]
        Bpatches = self.pooling(Bhead)      # [N, K, H//P, W//P]

        Pm = torch.amax(Mpatches, dim=(2, 3))        # [N, 2]
        Pb = torch.amax(Bpatches, dim=(2, 3))        # [N, K]
        return Mhead, Bhead, Pm, Pb

    def discriminatorLoss(self, X, Ym, Yb, piter=0.):
        N = Ym.shape[0]
        with torch.no_grad():
            _, _, Pm, Pb = self.forward(X)
        loss = self.D.loss(Pm, Pb, torch.zeros(N, 1))
        loss = freeze(loss, piter)
        loss = loss + self.D.loss(
            F.one_hot(Ym, num_classes=2), 
            F.one_hot(Yb, num_classes=self.K), 
            torch.ones(N, 1)
        )
        return loss

    def loss(self, X, Ym, Yb=None, piter=0.):
        '''
        X: [N, ic, H, W]
        Ym: [N], long
        Yb: [N], long
        '''
        
        M, B, Pm, Pb = self.forward(X)      # ToyNetV1 does not constrain between the two CAMs
        Mloss = focalCE(Pm, Ym, gamma=2 * piter, weight=self.mbalance)
        Mpenalty = F.mse_loss(M - 0.5, torch.zeros_like(M))
        consistency = self.D.forward(Pm, Pb).mean()
        loss = Mloss + 0.2 * Mpenalty + 0.5 * (1 - consistency)

        summary = {
            'loss/malignant focal': Mloss.detach(), 
            'penalty/CAM_malignant': Mpenalty.detach(), 
            'consistency': consistency.detach()
        }
        if Yb is not None: 
            Bloss = focalCE(Pb, Yb, gamma=2 * piter, weight=self.bbalance)
            Bpenalty = F.mse_loss(B - 0.5, torch.zeros_like(B))
            loss = loss + Bloss + 0.2 * Bpenalty
            summary['loss/BIRADs focal'] = Bloss.detach()
            summary['penalty/CAM_BIRADs'] = Bpenalty.detach()
        return loss, summary

if __name__ == "__main__":
    x = torch.randn(2, 1, 572, 572)
    toy = ToyNetV1(
        (1, 572, 572), 
        6, 12
    )
    loss, _ = toy.loss(x, torch.zeros(2, dtype=torch.long), torch.ones(2, dtype=torch.long))
    loss.backward()
