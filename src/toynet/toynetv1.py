"""
A toy implement for classifying benign/malignant and BIRADs

* author: JamzumSum
* create: 2021-1-11
"""
from itertools import chain

import torch
import torch.nn as nn
from common.loss import F, focal_smooth_loss
from common.utils import unsqueeze_as, freeze

from .discriminator import WithCD
from .unet import ConvStack2, DownConv, UNet


class BIRADsUNet(nn.Module):
    """
    [N, ic, H, W] -> [N, 2, H, W], [N, K, H, W]
    """

    def __init__(
        self,
        in_channel,
        K,
        width=64,
        ulevel=4,
        ylevel=1,
        memory_trade=False,
        zero_init_residual=True,
    ):
        nn.Module.__init__(self)
        self.unet = UNet(
            ic=in_channel + 1,
            oc=1,
            fc=width,
            level=ulevel,
            inner_res=True,
            memory_trade=memory_trade,
        )
        self.memory_trade = memory_trade
        self.ylevel = ylevel

        cc = self.unet.fc * 2 ** self.unet.level
        mls = []
        for i in range(ylevel << 1):
            if i & 1:
                mls.append(ConvStack2(cc, oc=cc * 2, res=True))
            else:
                mls.append(DownConv(cc))
            cc = mls[-1].oc

        self.ypath = nn.Sequential(*mls)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mfc = nn.Linear(cc, 2)
        self.bfc = nn.Linear(cc, K)
        self.initParameter()

    def initParameter(self, zero_init_residual=False):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if isinstance(m, DownConv):
                    nn.init.constant_(m.weight, 0.25)
                else:
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, ConvStack2):
                    nn.init.constant_(m.CB[1].weight, 0)

    def forward(self, X, mask=None, segment=True, classify=True):
        """
        X: [N, ic, H, W]
        mask: [N, 1, H, W]
            NOTE: Do NOT pass mask when segment=True for it's a LEAK of ground-truth.
            `mask` is only a guidance for classifying path(ypath).
        return: 
            segment map           [N, 2, H, W]
            x: bottom of unet feature. [N, fc * 2^level]
            Pm        [N, 2]
            Pb        [N, K]
        """
        N = X.size(0)
        assert segment or classify, "hello? :D"
        if mask is None:
            mask = torch.zeros((N, 1, *X.shape[2:]), device=X.device)
        else:
            assert not segment, "ground-truth leaking!"

        X = torch.cat((X, mask), dim=1)
        c, segment = self.unet(X, segment)
        if self.ylevel:
            c = self.ypath(c)
        x = self.pool(c).squeeze(2).squeeze(2)  # [N, fc * 2^(level + ylevel)]
        if classify:
            Pm = F.softmax(self.mfc(x), dim=1)  # [N, 2]
            Pb = F.softmax(self.bfc(x), dim=1)  # [N, K]
            return segment, x, Pm, Pb
        else:
            return segment, x, None, None

    def parameter_groups(self, weight_decay: dict):
        paramAll = self.parameters()
        paramB = tuple(self.bfc.parameters())
        paramM = (p for p in paramAll if id(p) not in [id(i) for i in paramB])
        need_decay = []
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                need_decay.append(id(m.weight))
            elif isinstance(m, nn.Linear):
                need_decay.append(id(m.weight))

        paramdic = {}
        for branch, param in {"M": paramM, "B": paramB}.items():
            if weight_decay[branch]:
                paramdic[branch] = (i for i in param if id(i) in need_decay)
                paramdic[branch + "_no_decay"] = (
                    i for i in param if id(i) not in need_decay
                )
            else:
                paramdic[branch] = param
        return paramdic


class ToyNetV1(BIRADsUNet):
    support = ("segment",)
    mweight = torch.Tensor([0.4, 0.6])
    bweight = torch.Tensor([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])

    def _loss(self, X, Ym, Yb=None, mask=None, piter=0.0):
        """
        Protected for classes inherit from ToyNetV1.
        return: Original result, M-branch losses, B-branch losses.
        """
        if self.mweight.device != X.device:
            self.mweight = self.mweight.to(X.device)
        if self.bweight.device != X.device:
            self.bweight = self.bweight.to(X.device)

        res = self.forward(X, segment=mask is not None)
        seg, embed, Pm, Pb = res
        guide_pm = guide_pb = None
        # ToyNetV1 does not constrain between the two CAMs
        # But may constrain on their own values, if necessary
        loss = {}

        if torch.rand(1) < 0.5 + piter:
            # allow mask guidance
            if torch.rand(1) < piter ** 4:
                # use segment result instead of ground-truth
                _, _, guide_pm, guide_pb = self.forward(X, mask=seg, segment=False)
            elif mask is not None:
                # use ground-truth as guidance
                _, _, guide_pm, guide_pb = self.forward(X, mask=mask, segment=False)
            

        loss["pm"] = focal_smooth_loss(Pm, Ym, gamma=1 + piter, weight=self.mweight)
        if guide_pm is not None:
            gmloss = focal_smooth_loss(guide_pm, Ym, gamma=1 + piter, weight=self.mweight)
            loss['pm'] = (loss['pm'] + gmloss) / 2
            # guidance consistancy loss
            loss["gcm"] = freeze(
                F.kl_div(Pm.log(), guide_pm, reduction="batchmean"),
                gmloss.detach().clamp(0, 1),
            )
        
        if seg is not None:
            loss["seg"] = ((seg - mask) ** 2 * (mask + 1)).mean()

        if Yb is not None:
            loss["pb"] = focal_smooth_loss(Pb, Yb, gamma=1 + piter, weight=self.bweight)
            if guide_pb is not None:
                gbloss = focal_smooth_loss(guide_pm, Ym, gamma=1 + piter, weight=self.mweight)
                loss['pb'] = (loss['pb'] + gbloss) / 2
                loss["gcb"] = freeze(
                    F.kl_div(Pb.log(), guide_pb, reduction="batchmean"),
                    gbloss.detach().clamp(0, 1),
                )

        # mcount = torch.bincount(Ym)
        # if len(mcount) == 2 and mcount[0] > 0:
        #     loss['tm'] = self.sh_triloss(embed, Ym)
        # TODO: triplet of BIRADs
        return res, loss

    def lossWithResult(self, *args, **argv):
        res, loss = self._loss(*args, **argv)
        itemdic = {
            "pm": "m/CE",
            "tm": "m/triplet",
            "seg": "segment-mse",
            "pb": "b/CE",
            'gcm': 'm/mask-guidance-consist',
            'gcb': 'b/mask-guidance-consist',
            # 'tb': 'b/triplet'
        }
        cum_loss = 0
        summary = {}
        for k, v in itemdic.items():
            if k not in loss:
                continue
            cum_loss = cum_loss + loss[k]
            summary["loss/" + v] = loss[k].detach()
        return res, cum_loss, summary

    def loss(self, *args, **argv):
        """
        X: [N, ic, H, W]
        Ym: [N], long
        Yb: [N], long
        piter: float in (0, 1)
        """
        return self.lossWithResult(*args, **argv)[1:]

    @staticmethod
    def WCDVer():
        return WithCD(ToyNetV1)


if __name__ == "__main__":
    x = torch.randn(2, 1, 572, 572)
    toy = ToyNetV1(1, 6, [12, 24, 48])
    loss, _ = toy.loss(
        x, torch.zeros(2, dtype=torch.long), torch.ones(2, dtype=torch.long)
    )
    loss.backward()
