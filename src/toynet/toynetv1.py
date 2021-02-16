'''
A toy implement for classifying benign/malignant and BIRADs

* author: JamzumSum
* create: 2021-1-11
'''
from itertools import chain

import torch
import torch.nn as nn
from common.loss import F, focalBCE, SemiHardTripletLoss
from common.utils import unsqueeze_as

from .discriminator import WithCD
from .unet import UNet, ConvStack2, DownConv

class BIRADsUNet(nn.Module):
    '''
    [N, ic, H, W] -> [N, 2, H, W], [N, K, H, W]
    '''
    def __init__(self, ic, K, fc=64, ylevel=1, pi=.5, memory_trade=False):
        nn.Module.__init__(self)
        self.unet = UNet(
            ic=ic, oc=1, fc=fc,
            inner_res=True, memory_trade=memory_trade
        )
        self.memory_trade = memory_trade
        self.ylevel = ylevel

        cc = self.unet.fc * 2 ** self.unet.level
        mls = []
        for _ in range(ylevel):
            mls.append(DownConv(cc))
            conv = ConvStack2(mls[-1].oc, oc=cc * 2, res=True)
            mls.append(conv)
            cc = mls[-1].oc

        self.ypath = nn.Sequential(*mls)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mfc = nn.Linear(cc, 2)
        self.bfc = nn.Linear(cc, K)

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
        if self.ylevel: c = self.ypath(c)
        x = self.pool(c).squeeze(2).squeeze(2)    # [N, fc * 2^(level + ylevel)]
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
    sh_triloss = SemiHardTripletLoss()

    @staticmethod
    def apn(c, Y, K=-1):
        '''
        c: [N, D]
        Y: [N]
        return:
            a: [1, D]
            p: [num_p, D]. num_p is the number of all postive samples except the anchor.
            n: [1, D]
        '''
        distrib = torch.bincount(Y)     # [K]
        N = c.size(0)
        K = distrib.size(0)
        inf_safe = torch.any(distrib == 0)
        if inf_safe and (distrib != 0).sum() < 2: 
            print('Warning: Less than 2 classes in the batch. Cannot calculate triplet. Skipped.')
            return

        mask = F.one_hot(Y, num_classes=K).unsqueeze(1)   # [N, 1, K]
        K = mask.shape[-1]
        ck = c.unsqueeze(-1) * mask             # [N, D, K], masked.

        mean = ck.sum(dim=0) / distrib          # [D, K]
        if inf_safe: mean[mean.isinf()] = 0
        center = torch.pow((ck - mean) * mask, 2).sum(dim=1)   # [N, K]
        std = center.sum(dim=0) / distrib       # [K]
        if inf_safe: std[std.isinf()] = 0

        acls = std.argmax()
        arga = center[:, acls].argmax()
        a = c[arga].unsqueeze(0)                # [N, D]

        center_a = torch.pow((ck - a.unsqueeze(-1)) * mask, 2).sum(dim=1)      # [N, K]
        argp = Y == acls
        argp[arga] = False
        p = c[argp][: max(1, N // K)]
        dp = center_a[:, acls].max()

        inf = center_a.max() + 1
        center_a[:, acls] = inf
        center_a[center_a == 0] = inf
        center_a[center_a <= dp] = inf
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
        seg, embed, Pm, Pb = res
        # ToyNetV1 does not constrain between the two CAMs
        # But may constrain on their own values, if necessary
        loss = {}
        
        loss['pm'] = focalBCE(Pm, Ym, gamma=1 + piter, weight=self.mweight)

        if seg is not None:
            loss['seg'] = ((seg - mask) ** 2 * (mask + 1)).mean()

        if Yb is not None: 
            loss['pb'] = focalBCE(Pb, Yb, gamma=1 + piter, weight=self.bweight)

        # mcount = torch.bincount(Ym)
        # if len(mcount) == 2 and mcount[0] > 0:
        #     loss['tm'] = self.sh_triloss(embed, Ym)
        # TODO: triplet of BIRADs
        return res, loss

    def lossWithResult(self, *args, **argv):
        res, loss = self._loss(*args, **argv)
        itemdic = {
            'pm': 'm-CE',
            'tm': 'm-triplet',
            'seg': 'segment-mse',
            'pb': 'b-CE',
            # 'tb': 'b-triplet'
        }
        cum_loss = 0
        summary = {}
        for k, v in itemdic.items():
            if k not in loss: continue
            cum_loss = cum_loss + loss[k]
            summary['loss/' + v] = loss[k].detach()
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
