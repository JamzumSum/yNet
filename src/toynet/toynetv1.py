"""
A toy implement for classifying benign/malignant and BIRADs

* author: JamzumSum
* create: 2021-1-11
"""
from collections import defaultdict
from itertools import chain

import torch
import torch.nn as nn
from common import freeze, unsqueeze_as
from common.decorators import NoGrad
from common.loss import F, focal_smooth_bce, focal_smooth_ce
from common.loss.triplet import WeightedExampleTripletLoss
from common.support import *
from common.utils import BoundingBoxCrop, morph_close
from misc import CoefficientScheduler as CSG

from .discriminator import WithCD
from .unet import ConvStack2, DownConv, UNet

yes = lambda p: torch.rand(()) < p


class BIRADsUNet(nn.Module, SegmentSupported, SelfInitialed, MultiBranch):
    """
    [N, ic, H, W] -> [N, 2, H, W], [N, K, H, W]
    """

    norm = "groupnorm"

    def __init__(
        self,
        in_channel,
        K,
        width=64,
        ulevel=4,
        ylevel=1,
        dropout=0.2,
        memory_trade=False,
        zero_init_residual=True,
    ):
        nn.Module.__init__(self)
        self.unet = UNet(
            ic=in_channel,
            oc=1,
            fc=width,
            level=ulevel,
            inner_res=True,
            memory_trade=memory_trade,
            norm="groupnorm",
        )
        self.memory_trade = memory_trade
        self.ylevel = ylevel

        cc = self.unet.oc * 2  # 2x channel for 2 placeholders
        mls = []
        for i in range(ylevel << 1):
            if i & 1:
                mls.append(ConvStack2(cc, oc=cc * 2, res=True, norm=self.norm))
            else:
                mls.append(DownConv(cc))
            cc = mls[-1].oc

        self.ypath = nn.Sequential(*mls)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mfc = nn.Linear(cc, 2)
        self.bfc = nn.Linear(cc, K)
        self.final_norm = nn.GroupNorm(max(1, cc // 16), cc)
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

    def forward(
        self, X, mask=None, segment=True, classify=True, logit=False, pad_mode="zero"
    ):
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
                `roi`: Crop images according to segment and pass them to unet to get another placeholder. 
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

        # TODO: move these to class method
        def from_segment(X, seg, c, mask):  # All test failed. probability collapsed.
            with torch.no_grad():
                if mask is None:
                    mask = morph_close(seg)
                X = X * mask.clamp(0.3, 1)
            return self.unet(X, False)

        def roi_extract(X, seg, c, mask):
            raise NotImplementedError
            with torch.no_grad():
                if mask is None:
                    mask = morph_close(seg)
                Xl = BoundingBoxCrop(0.5, 2 ** self.max_depth)(mask, X)  # N * [C, h, w]
            # TODO BUG: enter inference mode when using BN
            # or use GroupNorm if possible...
            return torch.cat([self.unet(xi.unsqueeze(0), False) for xi in Xl])

        def zero_padding(X, seg, c, mask):
            # 0304: Test passed. Prove that placeholders are working.
            return torch.zeros_like(c)

        def copy_it(X, seg, c, mask):
            return c.copy()

        cg = {
            "segment": from_segment,
            "zero": zero_padding,
            "copy": copy_it,
            "roi": roi_extract,
        }[pad_mode](X, seg, c, mask)

        # NOTE: Two placeholders c & cg
        # TODO: what about a weight balancing the two placeholders?
        #       Then the linear size can be reduced as 0.5x :D
        c = torch.cat((c, cg), dim=1)  # [N, fc * 2^(ul + 1), H, W]

        # fmt: off
        if self.ylevel: c = self.ypath(c)
        c = self.pool(c)[..., 0, 0]    # [N, fc * 2^(ul + yl + 1)]
                                       # empirically, D = fc * 2^(ul + yl + 1) >= 128

        x = self.final_norm(c)
        # TODO: is fc's biases neccesary?
        lm = self.mfc(x)  # [N, 2]
        lb = self.bfc(x)  # [N, K]
        if logit: return seg, c, lm, lb
        # fmt: on

        Pm = F.softmax(lm, dim=1)
        Pb = F.softmax(lb, dim=1)
        return seg, c, Pm, Pb

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


class ToyNetV1(BIRADsUNet):
    __version__ = (1, 0)

    def __init__(self, in_channel, *args, margin=0.3, **kwargs):
        super().__init__(in_channel, *args, **kwargs)
        # TODO: weights should be set by config...
        self.register_buffer("mweight", torch.Tensor([0.4, 0.6]))
        self.register_buffer("bweight", torch.Tensor([0.1, 0.2, 0.2, 0.2, 0.2, 0.1]))

        self.triplet = WeightedExampleTripletLoss(margin=margin) if margin > 0 else None

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
            res = self.forward(X, segment=False, logit=True)

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
