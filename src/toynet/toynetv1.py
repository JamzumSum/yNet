"""
A toy implement for classifying benign/malignant and BIRADs

* author: JamzumSum
* create: 2021-1-11
"""
from collections import defaultdict
from itertools import chain

import torch
import torch.nn as nn
from common import freeze, unsqueeze_as, spatial_softmax
from common.decorators import NoGrad
from common.loss import F, focal_smooth_bce, focal_smooth_ce
from common.loss.triplet import WeightedExampleTripletLoss
from common.support import *
from common.utils import BoundingBoxCrop
from misc import CoefficientScheduler as CSG

from .discriminator import WithCD
from .unet import ConvStack2, DownConv, UNet, ChannelNorm

yes = lambda p: torch.rand(()) < p


class YNet(nn.Module, SegmentSupported, SelfInitialed):
    """
    YNet: image[N, 1, H, W] ->  segment[N, 1, H, W], embedding[N, D], D = 2^(ul+yl+1)
    """

    def __init__(
        self,
        in_channel,
        width=64,
        ulevel=4,
        ylevels: list = None,
        memory_trade=False,
        residual=True,
        zero_init_residual=True,
        norm="batchnorm",
        pad_mode="zero",
    ):
        nn.Module.__init__(self)

        self.memory_trade = memory_trade
        self.pad_mode = pad_mode
        self.norm = norm
        self.residual = residual
        if ylevels is None:
            self.ylevel = 0
            ylevels = []
        else:
            self.ylevel = len(ylevels)

        self.unet = UNet(
            ic=in_channel,
            oc=1,
            fc=width,
            level=ulevel,
            residual=residual,
            memory_trade=memory_trade,
            norm=norm,
        )
        cc = self.unet.oc * 2  # 2x channel for 2 placeholders
        ylayers = []
        for ylevel in ylevels:
            for i in range(ylevel):
                if i % ylevel:
                    ylayers.append(ConvStack2(cc, cc, self.residual, self.norm))
                else:
                    ylayers.append(ConvStack2(cc, 2 * cc, self.residual, self.norm))
                    ylayers.append(DownConv(2 * cc))
                cc = ylayers[-1].oc
        self.yoc = cc
        self.ypath = nn.Sequential(*ylayers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.selfInit()

    @property
    def max_depth(self):
        return self.unet.level + self.ylevel

    def selfInit(self, zero_init_residual=False):
        for m in self.modules():
            if isinstance(m, SelfInitialed):
                if m is self:
                    continue
                m.selfInit()
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, ConvStack2):
                    nn.init.constant_(m.CB[1].weight, 0)

    @NoGrad
    def pad_placeholder(self, X, seg, c, mask, pad_mode=None):
        # NOTE: whether the entire block requires no grad is under consideration.
        # under current condition it will be just a classification basis for fc.
        pad_mode = pad_mode or self.pad_mode
        if pad_mode == "segment":
            # 0312: Any tests failed again
            # fmt: off
            if mask is None: mask = seg
            # softmax ensures the min of attentioned image >= 1/H*W*e
            return c + self.unet(X * spatial_softmax(mask), False)
            # fmt: on
        elif pad_mode == "zero":
            return torch.zeros_like(c)
        else:
            raise ValueError(pad_mode)

    def forward(self, X, mask=None, segment=True, classify=True, pad_mode=None):
        """
        args: 
            X: [N, ic, H, W]
            mask: optional [N, 1, H, W]
        flag:
            segment: if true, the whole unet will be inferenced to generate a segment map.
            classify: if true, the ypath is inferenced to get classification result.
            logit: skip softmax for some losses that needn't that.
            pad_mode: how to generate feature for the second plcaeholder. 
                `segment`: If map not given, force segment current X. Then use the segment map to generate feature.
                `zero`: Set the second placeholder as 0. A little like dropout.
                `copy`: Set the second placeholder same as the first. Risk of overfitting.
        return: 
            segment map  [N, 2, H, W]. return this alone if `segment but not classify`
            x: bottom of unet feature. [N, fc * 2^ul]
            Pm        [N, 2]
            Pb        [N, K]
        """
        if mask is None or segment:
            c, seg = self.unet(X)
        else:
            c = self.unet(X, False)

        if not classify:
            return seg

        cg = self.pad_placeholder(X, seg, c, mask, pad_mode)

        # NOTE: Two placeholders c & cg
        # TODO: what about a weight balancing the two placeholders?
        #       Then the linear size can be reduced as 0.5x :D
        c = torch.cat((c, cg), dim=1)  # [N, fc * 2^(ul + 1), H, W]

        if self.ylevel:
            c = self.ypath(c)
        ft = self.pool(c)[..., 0, 0]  # [N, fc * 2^(ul + yl + 1)]
        # empirically, D = fc * 2^(ul + yl + 1) >= 128
        return seg, ft


class BIRADsUNet(YNet, MultiBranch):
    def __init__(self, in_channel, K, *args, norm="batchnorm", **kwargs):
        YNet.__init__(self, in_channel, *args, **kwargs)
        cc = self.yoc
        self.mfc = nn.Linear(cc, 2)
        self.bfc = nn.Linear(cc, K)
        self.sigma = nn.Softmax(dim=1)
        # fmt: off
        self.norm_layer = {
            "batchnorm": nn.BatchNorm1d, 
            "groupnorm": ChannelNorm
        }[norm](cc)
        # fmt: on

    def branch_weight(self, weight_decay: dict):
        """
        args:
            weight_decay: all keys should be in self.branches
        exmaple: 
            weight_decay: {
                'M': True, 
                'B': True
            }
        """
        paramAll = self.parameters()
        paramB = tuple(self.bfc.parameters())
        paramM = tuple(p for p in paramAll if id(p) not in [id(i) for i in paramB])
        paramdic = {"M": paramM, "B": paramB}
        # a param dict when not filter by `weight_decay`
        if not any(weight_decay.values()):
            return paramdic

        decay_weight_ge = defaultdict(
            lambda: lambda m: [],
            {nn.Conv2d: lambda m: [id(m.weight)], nn.Linear: lambda m: [id(m.weight)]},
        )
        need_decay = (decay_weight_ge[type(m)](m) for m in self.modules())
        need_decay = sum(need_decay, [])

        for branch, param in paramdic.copy().items():
            if weight_decay[branch]:
                paramdic[branch + "_no_decay"] = (
                    i for i in param if id(i) not in need_decay
                )
                paramdic[branch] = [i for i in param if id(i) in need_decay]
            else:
                paramdic[branch] = param
        return paramdic

    @property
    def branches(self):
        return ("M", "B")

    def forward(
        self, X, mask=None, segment=True, classify=True, logit=False, pad_mode=None
    ):
        # BNNeck below.
        # See: A Strong Baseline and Batch Normalization Neck for Deep Person Re-identification
        # Use ft to calculate triplet, etc.; use fi to classify.
        seg, ft = YNet.forward(self, X, mask, segment, classify, pad_mode)
        fi = self.norm_layer(ft)

        lm = self.mfc(fi)  # [N, 2]
        lb = self.bfc(fi)  # [N, K]
        if logit:
            return seg, ft, lm, lb

        Pm = self.sigma(lm)
        Pb = self.sigma(lb)
        return seg, ft, Pm, Pb


class ToyNetV1(BIRADsUNet):
    __version__ = (1, 0)

    def __init__(self, in_channel, *args, margin=0.3, **kwargs):
        super().__init__(in_channel, *args, **kwargs, pad_mode="segment")
        # TODO: weights should be set by config...
        self.register_buffer("mweight", torch.Tensor([0.4, 0.6]))
        self.register_buffer("bweight", torch.Tensor([0.1, 0.2, 0.2, 0.2, 0.2, 0.1]))

        # triplet-ify using cosine distance by default(normalize=True)
        self.triplet = (
            WeightedExampleTripletLoss(margin=margin, normalize=False)
            if margin > 0
            else None
        )

    def _loss(self, cmgr: CSG, X, Ym, Yb=None, mask=None):
        """
        Protected for classes inherit from ToyNetV1.
        return: Original result, M-branch losses, B-branch losses.
        """
        piter = cmgr.piter
        probe_self_guide = cmgr.get("probe_self_guide", piter ** 4)

        # allow mask guidance
        if yes(probe_self_guide):
            # use segment result instead of ground-truth
            res = self.forward(X, segment=mask is not None, logit=True)
        elif mask is not None:
            # use ground-truth as guidance
            res = self.forward(X, mask=mask, logit=True)
        else:
            res = self.forward(X, segment=False, logit=True, pad_mode="zero")

        # ToyNetV1 does not constrain between the two CAMs
        # But may constrain on their own values, if necessary
        def lossInner(seg, embed, lm, lb):
            loss = {"pm": F.cross_entropy(lm, Ym, weight=self.mweight)}

            if mask is not None and seg is not None:
                weight_focus_pos = cmgr.get("weight_focus_pos", 1 - piter ** 2)
                loss["seg"] = ((seg - mask) ** 2 * (weight_focus_pos * mask + 1)).mean()

            if Yb is not None:
                gamma_b = cmgr.get("gamma_b", piter + 1)
                loss["pb"] = focal_smooth_bce(
                    lb, Yb, gamma=gamma_b, weight=self.bweight
                )

            if self.triplet:
                mcount = torch.bincount(Ym)
                if len(mcount) == 2 and mcount[0] > 0:
                    loss["tm"] = self.triplet(embed, Ym)
                # TODO: triplet of BIRADs
            return loss

        loss = lossInner(*res)
        return res, loss

    def lossWithResult(self, cmgr: CSG, *args, **argv):
        res, loss = self._loss(cmgr, *args, **argv)
        itemdic = {
            "pm": "m/CE",
            "tm": "m/triplet",
            "seg": "segment-mse",
            "pb": "b/CE",
            # 'tb': 'b/triplet'
        }
        cum_loss = 0
        summary = {}
        for k, v in itemdic.items():
            if k not in loss:
                continue
            # multi-task loss fusion
            cum_loss = cum_loss + loss[k] * cmgr.get("task_" + k, 1)
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

    @classmethod
    def WCDVer(cls):
        return WithCD(cls)
