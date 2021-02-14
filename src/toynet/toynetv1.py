'''
A toy implement for classifying benign/malignant and BIRADs

* author: JamzumSum
* create: 2021-1-11
'''
from itertools import chain

import torch
import torch.nn as nn
from common.loss import F, focalBCE
from common.utils import unsqueeze_as

from .discriminator import WithCD
from .unet import UNet

class BIRADsUNet(nn.Module):
    '''
    [N, ic, H, W] -> [N, 2, H, W], [N, K, H, W]
    '''
    def __init__(self, ic, K, fc=64, pi=.5, memory_trade=False):
        nn.Module.__init__(self)
        self.unet = UNet(
            ic=ic, oc=1, fc=fc,
            inner_res=True, memory_trade=memory_trade
        )
        self.memory_trade = memory_trade
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mfc = nn.Linear(self.unet.fc * 2 ** self.unet.level, 2)
        self.bfc = nn.Linear(self.unet.fc * 2 ** self.unet.level, K)

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
        x = self.pool(c).squeeze(2).squeeze(2)    # [N, fc * 2^level]
        Pm = F.softmax(self.mfc(x), dim=1)    # [N, 2]
        Pb = F.softmax(self.bfc(x), dim=1)    # [N, K]
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

    @staticmethod
    def apn(c, Y, K=-1):
        '''
        c: [N, D]
        Y: [N]
        return:
            a: [1, D]
            p: [1, D]
            n: [1, D]
        '''
        distrib = torch.bincount(Y)     # [K]
        inf_safe = torch.any(distrib == 0)
        if inf_safe and (distrib != 0).sum() < 2: 
            print('Warning: Less than 2 classes in the batch. Cannot calc. triplet.')
            return

        mask = F.one_hot(Y, num_classes=K).unsqueeze(1)   # [N, 1, K]
        K = mask.shape[-1]
        ck = c.unsqueeze(-1) * mask               # [N, D, K], masked.

        mean = ck.sum(dim=0) / distrib              # [D, K]
        if inf_safe: mean[mean.isinf()] = 0
        center = torch.pow((ck - mean) * mask, 2)   # [N, D, K]
        std = center.sum(dim=0) / distrib           # [D, K]
        if inf_safe: std[std.isinf()] = 0

        acls = (std).sum(dim=0).argmax()
        arga = center[:, :, acls].sum(dim=1).argmax()
        a = c[arga].unsqueeze(0)                   # [N, D]

        center_a = torch.pow((ck - a.unsqueeze(-1)) * mask, 2).sum(dim=1)      # [N, K]
        argp = torch.argmax(center_a[:, acls])
        p = c[argp].unsqueeze(0)                   # [N, D]

        inf = center_a.max() + 1
        center_a[:, acls] = inf
        center_a[center_a == 0] = inf
        argn = center_a.min(dim=1).values.argmin()
        n = c[argn].unsqueeze(0)

        return a, p, n

    def _loss(self, X, Ym, Yb=None, mask=None, piter=0.):
        '''
        Protected for classes inherit from ToyNetV1.
        return: Original result, M-branch losses, B-branch losses.
        '''
        if self.mweight.device != X.device:
            self.mweight = self.mweight.to(X.device)
        if self.bweight.device != X.device:
            self.bweight = self.bweight.to(X.device)

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

        loss['tri'] = F.triplet_margin_loss(*self.apn(c, Ym), margin=1., swap=True)
        # TODO: triplet of BIRADs
        return res, loss

    def lossWithResult(self, *args, **argv):
        res, loss = self._loss(*args, **argv)
        Mloss = loss['pm']
        Tloss = loss['tri']
        cum_loss = Mloss + Tloss
        summary = {
            'loss/malignant focal': Mloss.detach(), 
            'loss/triplet-m': Tloss.detach()
        }
        if 'seg' in loss:
            Sloss = loss['seg']
            cum_loss = cum_loss + Sloss
            summary['loss/segment mse'] = Sloss.detach()
        if 'pb' in loss:
            Bloss = loss['pb']
            cum_loss = cum_loss + Bloss
            summary['loss/BIRADs focal'] = Bloss.detach()
        return res, cum_loss, summary

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
