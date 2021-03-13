"""
A toy implement for classifying benign/malignant and BIRADs

* author: JamzumSum
* create: 2021-1-11
"""
from collections import defaultdict
from itertools import chain

import torch
import torch.nn as nn
from common import freeze, spatial_softmax, unsqueeze_as
from common.decorators import CheckpointSupport, NoGrad
from common.layers import MLP
from common.loss import F, focal_smooth_bce, focal_smooth_ce
from common.loss.triplet import WeightedExampleTripletLoss
from common.support import *
from misc import CoefficientScheduler as CSG

from .discriminator import WithCD
from .unet import ChannelNorm, ConvStack2, DownConv, UNet

yes = lambda p: torch.rand(()) < p


class YNet(nn.Module, SegmentSupported, SelfInitialed):
    """
    Generate embedding and segment of an image.
    YNet: image[N, 1, H, W] ->  segment[N, 1, H, W], embedding[N, D], D = fc * 2^(ul+yl)
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
    ):
        nn.Module.__init__(self)

        self.memory_trade = memory_trade
        self.norm = norm
        self.residual = residual
        if ylevels is None:
            ylevels = []
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
        cc = self.unet.oc
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

    def forward(self, X, segment=True, classify=True):
        """
        args: 
            X: [N, ic, H, W]
            mask: optional [N, 1, H, W]
        flag:
            segment: if true, the whole unet will be inferenced to generate a segment map.
            classify: if true, the ypath is inferenced to get classification result.
            logit: skip softmax for some losses that needn't that.
        return: 
            segment map  [N, 2, H, W]. return this alone if `segment but not classify`
            x: bottom of unet feature. [N, fc * 2^ul]
            Pm        [N, 2]
            Pb        [N, K]
        """
        assert segment or classify, "hello?"
        if segment:
            c, seg = self.unet(X)
        else:
            c = self.unet(X, False)

        if not classify:
            return seg

        if self.ylevel:
            c = self.ypath(c)
        ft = self.pool(c)[..., 0, 0]  # [N, D], D = fc * 2^(ul + yl)
        return seg, ft


class BIRADsYNet(YNet, MultiBranch):
    def __init__(self, in_channel, K, *args, zdim=2048, norm="batchnorm", **kwargs):
        YNet.__init__(self, in_channel, *args, **kwargs)
        cc = self.yoc
        self.zdim = zdim
        self.mfc = nn.Linear(cc, 2)
        self.bfc = nn.Linear(cc, K)
        self.sigma = nn.Softmax(dim=1)
        self.proj_mlp = MLP(cc, zdim, [2048, 2048], True, False)  # L3

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

    def forward(self, X, segment=True, classify=True, logit=False):
        # BNNeck below.
        # See: A Strong Baseline and Batch Normalization Neck for Deep Person Re-identification
        # Use ft to calculate triplet, etc.; use fi to classify.
        if classify:
            seg, ft = YNet.forward(self, X, segment, classify)
        else:
            return YNet.forward(self, X, segment, classify)

        fi = self.proj_mlp(ft)  # [N, Z], empirically, Z >= 128

        lm = self.mfc(fi)  # [N, 2]
        lb = self.bfc(fi)  # [N, K]
        if logit:
            return seg, ft, fi, lm, lb

        Pm = self.sigma(lm)
        Pb = self.sigma(lb)
        return seg, ft, fi, Pm, Pb


class ToyNetV1(BIRADsYNet):
    """
    ToyNetV1 does not constrain between the two CAMs, 
    But may constrain on their own values, if necessary.
    """

    __version__ = (1, 1)

    def __init__(self, in_channel, *args, margin=0.3, **kwargs):
        super().__init__(in_channel, *args, **kwargs)
        # TODO: weights should be set by config...
        self.register_buffer("mweight", torch.Tensor([0.4, 0.6]))
        self.register_buffer("bweight", torch.Tensor([0.1, 0.2, 0.2, 0.2, 0.2, 0.1]))

        # triplet-ify using cosine distance by default(normalize=True)
        self.triplet = (
            WeightedExampleTripletLoss(margin=margin, normalize=False)
            if margin > 0
            else None
        )
        # siamise loss
        self.fuse_layer = nn.Conv2d(2, 1, 1, bias=False)
        self.pred_mlp = MLP(self.zdim, self.zdim, [512], False, False)  # L2

    @staticmethod
    def neg_cos_sim(p, z):
        z = z.detach()  # stop-grad
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return -(p * z).sum(dim=1).mean()

    def second_view(self, X, seg):
        """
        return the embedding of an augmented view
        """
        # softmax ensures the min of attentioned image >= 1/H*W*e
        fused = self.fuse_layer(torch.cat((X, spatial_softmax(seg)), dim=1))
        fi2 = self.forward(fused, segment=False, classify=True, logit=True)[2]
        return fi2

    def _loss(self, cmgr: CSG, X, Ym, Yb=None, mask=None):
        """
        Protected for classes inherit from ToyNetV1.
        """
        piter = cmgr.piter
        res = self.forward(X, segment=True, classify=True, logit=True)
        seg, ft, fi, lm, lb = res
        fi2 = self.second_view(X, seg.detach())

        D = lambda p, z: self.neg_cos_sim(self.pred_mlp(p), z)
        loss = {
            "pm": F.cross_entropy(lm, Ym, weight=self.mweight),
            "sim": (D(fi, fi2) + D(fi2, fi)) / 2,
        }

        if mask is not None and seg is not None:
            weight_focus_pos = cmgr.get("weight_focus_pos", 1 - piter ** 2)
            loss["seg"] = ((seg - mask) ** 2 * (weight_focus_pos * mask + 1)).mean()

        if Yb is not None:
            gamma_b = cmgr.get("gamma_b", piter + 1)
            loss["pb"] = focal_smooth_bce(lb, Yb, gamma=gamma_b, weight=self.bweight)

        if self.triplet:
            # TODO: check batch-meta
            loss["tm"] = self.triplet(ft, Ym)
            # TODO: triplet of BIRADs

        return res, loss

    def lossWithResult(self, cmgr: CSG, *args, **argv):
        res, loss = self._loss(cmgr, *args, **argv)
        itemdic = {
            "pm": "m/CE",
            "tm": "m/triplet",
            "sim": "siamise/neg_cos_similarity",
            "seg": "segment-mse",
            "pb": "b/CE",
            "tb": "b/triplet",
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
