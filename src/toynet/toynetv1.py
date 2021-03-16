"""
A toy implement for classifying benign/malignant and BIRADs

* author: JamzumSum
* create: 2021-1-11
"""
from collections import defaultdict
from itertools import chain

import torch
import torch.nn as nn
from common import freeze
from common.decorators import CheckpointSupport, NoGrad
from common.layers import MLP
from common.loss.triplet import WeightedExampleTripletLoss
from common.support import *
from misc import CoefficientScheduler as CSG

from .lossbase import *
from .ynet import YNet


class BIRADsYNet(YNet, MultiBranch):
    def __init__(self, in_channel, K, *args, zdim=2048, norm="batchnorm", **kwargs):
        YNet.__init__(self, in_channel, *args, **kwargs)
        cc = self.yoc
        self.zdim = zdim
        self.sigma = nn.Softmax(dim=1)
        self.proj_mlp = MLP(cc, zdim, [2048, 2048], False, False)  # L3
        self.norm_layer = nn.BatchNorm1d(zdim)
        self.mfc = nn.Linear(zdim, 2)
        self.bfc = nn.Linear(zdim, K)

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

        ft = self.proj_mlp(ft)  # [N, Z], empirically, Z >= 128
        fi = self.norm_layer(ft)

        lm = self.mfc(fi)  # [N, 2]
        lb = self.bfc(fi)  # [N, K]
        if logit:
            return seg, ft, fi, lm, lb

        Pm = self.sigma(lm)
        Pb = self.sigma(lb)
        return seg, ft, fi, Pm, Pb


class ToyNetV1(BIRADsYNet):
    """
    ToyNetV1 does not deal too much with BIRADs task. 
    It just apply an usual CE supervise on it.
    """

    __version__ = (1, 2)

    def __init__(self, cmgr: CSG, in_channel, *args, margin=0.3, **kwargs):
        super().__init__(in_channel, *args, **kwargs)
        self.cmgr = cmgr
        self.siamise = SiamiseBase(cmgr, self, self.zdim)
        self.triplet = TripletBase(cmgr, margin, False)
        self.ce = CEBase(cmgr, self)
        self.segmse = MSESegBase(cmgr)

    def loss(self, meta: dict, X, Ym, Yb=None, mask=None):
        """
        return the result asis and a loss dict.
        """
        res = self.forward(X, segment=True, classify=True, logit=True)
        seg, ft, fi, lm, lb = res

        loss = self.ce(lm, lb, Ym, Yb)
        loss.update(self.siamise(X, seg, fi))
        loss.update(self.segmse(seg, mask))

        if meta['balanced']:
            loss.update(self.triplet(ft, Ym))
            # TODO: triplet of BIRADs

        return res, loss

    def multiTaskLoss(self, loss: dict):
        cum_loss = 0
        for k, v in loss.items():
            cum_loss = cum_loss + v * self.cmgr.get("task_" + k, 1)
        return cum_loss

    @staticmethod
    def lossSummary(loss: dict):
        itemdic = {
            "pm": "m/CE",
            "tm": "m/triplet",
            "sim": "siamise/neg_cos_similarity",
            "seg": "segment-mse",
            "pb": "b/CE",
            "tb": "b/triplet",
        }
        return {"loss/" + v: loss[k].detach() for k, v in itemdic.items() if k in loss}
