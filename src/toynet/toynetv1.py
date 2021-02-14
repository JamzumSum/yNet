'''
A toy implement for classifying benign/malignant and BIRADs

* author: JamzumSum
* create: 2021-1-11
'''
from itertools import chain

import torch
import torch.nn as nn
from common.loss import F, focalBCE

from .discriminator import WithCD
from .unet import UNet

class BIRADsUNet(nn.Module):
    '''
    [N, ic, H, W] -> [N, 2, H, W], [N, K, H, W]
    '''
    def __init__(self, ic, K, fc=64, pi=.5, memory_trade=False):
        self.unet = UNet(
            ic=ic, oc=1, fc=fc,
            inner_res=True, memory_trade=memory_trade
        )
        self.memory_trade = memory_trade
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mfc = nn.Linear(self.unet.fc * 2 ^ self.unet.level, 2)
        self.bfc = nn.Linear(self.unet.fc * 2 ^ self.unet.level, K)

    def forward(self, X, segment=True):
        '''
        X: [N, ic, H, W]
        return: 
        - segment map           [N, 2, H, W]
        - x: bottom of unet feature. [N, fc * 2^level]
        - Pm        [N, 2]
        - Pb        [N, K]
        '''
        c, segment = self.unet(X, segment)
        x = self.pool(c).view(c.shape[:2])    # [N, fc * 2^level]
        Pm = self.mfc(x)    # [N, 2]
        Pb = self.bfc(x)    # [N, K]
        return segment, x, Pm, Pb

    def seperatedParameters(self):
        paramAll = self.parameters()
        paramB = self.bfc.parameters()
        paramM = (p for p in paramAll if id(p) not in [id(i) for i in paramB])
        return paramM, paramB

class ToyNetV1(BIRADsUNet):
    support = ('segment', )
    mweight = torch.Tensor([.4, .6])
    bweight = torch.Tensor([.1, .2, .2, .2, .2, .1])

    # for high pressure of memory, quiz is deperated.
    # Penalties roll back to mse.
    # I'm seeking ways to implement this as an augment method.
    def quiz(self, X, M, B):
        '''
        A quiz to constrain network:
        1. not to put weights on meaningless shadow area
        '''
        # padding runtime augment
        randpad = 16 * torch.randint(4, (4,))
        pad = lambda x: F.pad(x, list(randpad.tolist()), 'constant')
        PX = pad(X); PM = pad(M); PB = pad(B)
        fPM, fPB = self.forward(PX, self.memory_trade)[:2]
        Mqloss = F.mse_loss(fPM, PM)
        Bqloss = F.mse_loss(fPB, PB)
        return Mqloss, Bqloss

    def _loss(self, X, Ym, Yb=None, mask=None, piter=0.):
        '''
        Protected for classes inherit from ToyNetV1.
        return: Original result, M-branch losses, B-branch losses.
        '''
        res = self.forward(X, mask is not None)
        seg, c, Pm, Pb = res
        # ToyNetV1 does not constrain between the two CAMs
        # But may constrain on their own values, if necessary
        loss = {}
        loss['pm'] = focalBCE(Pm, Ym, K=2, gamma=1 + piter, weight=self.mweight)

        if seg is not None:
            # TODO: maybe a weighted mse?
            loss['seg'] = F.mse_loss(seg, mask)

        if Yb is not None: 
            loss['pb'] = focalBCE(Pb, Yb, K=self.K, gamma=1 + piter, weight=self.bweight)

        # TODO: apply triplet loss on c
        # loss['tri'] = F.triplet_margin_loss(*self.apn(c), margin=r)
        return res, loss

    def lossWithResult(self, *args, **argv):
        res, loss = self._loss(*args, **argv)
        Mloss = loss['pm']
        Sloss = loss['seg']
        summary = {
            'loss/malignant focal': Mloss.detach(), 
            'loss/segment mse': Sloss.detach()
        }
        loss = Mloss + Sloss
        if 'pb' in loss:
            Bloss = loss['pb']
            loss = loss + Bloss
            summary['loss/BIRADs focal'] = Bloss.detach()
        return res, loss, summary

    def loss(self, *args, **argv):
        '''
        X: [N, ic, H, W]
        Ym: [N], long
        Yb: [N], long
        piter: float in (0, 1)
        '''
        return self.lossWithResult(*args, **argv)[1:]

    @staticmethod
    def WCDVer(): return WithCD(ToyNetV1)

if __name__ == "__main__":
    x = torch.randn(2, 1, 572, 572)
    toy = ToyNetV1(1, 6, [12, 24, 48])
    loss, _ = toy.loss(x, torch.zeros(2, dtype=torch.long), torch.ones(2, dtype=torch.long))
    loss.backward()
