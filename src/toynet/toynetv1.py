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
from common.decorators import NoGrad
from common.support import *
from data.augment.online import RandomAffine
from misc import CheckpointSupport as CPS
from misc import CoefficientScheduler as CSG
from misc.decorators import autoPropertyClass, noexcept

from .lossbase import *
from .ynet import YNet

first = lambda it: next(iter(it))


class BIRADsYNet(YNet, MultiBranch):
    def __init__(
        self,
        cps: CPS,
        in_channel,
        K,
        width=64,
        ulevel=4,
        *,
        norm="batchnorm",
        **kwargs
    ):
        YNet.__init__(self, cps, in_channel, width, ulevel, **kwargs)
        self.sigma = nn.Softmax(dim=1)
        self.norm_layer = nn.BatchNorm1d(self.yoc)
        self.mfc = nn.Linear(self.yoc, 2)
        self.bfc = nn.Linear(self.yoc, K)

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
        paramB = list(self.bfc.parameters())
        paramM = [p for p in paramAll if id(p) not in [id(i) for i in paramB]]
        paramdic = {"M": paramM, "B": paramB}
        # a param dict when not filter by `weight_decay`
        if not any(weight_decay.values()):
            return paramdic

        decay_weight_ge = defaultdict(
            lambda: lambda m: [],
            {
                nn.Conv2d: lambda m: [id(m.weight)],
                nn.Linear: lambda m: [id(m.weight)]
            },
        )
        need_decay = (decay_weight_ge[type(m)](m) for m in self.modules())
        need_decay = sum(need_decay, [])

        for branch, param in paramdic.copy().items():
            if weight_decay[branch]:
                paramdic[branch +
                         "_no_decay"] = [i for i in param if id(i) not in need_decay]
                paramdic[branch] = [i for i in param if id(i) in need_decay]
            else:
                paramdic[branch] = param
        return paramdic

    @property
    def branches(self):
        return ("M", "B")

    def forward(self, X, segment=True, classify=True, logit=False) -> dict:
        # BNNeck below.
        # See: A Strong Baseline and Batch Normalization Neck for Deep Person Re-identification
        # Use ft to calculate triplet, etc.; use fi to classify.
        r = YNet.forward(self, X, segment, classify)
        if not classify:
            return r

        ft = r["ft"]

        fi = self.norm_layer(ft)   # [N, Z], empirically, Z >= 128
        r["fi"] = fi

        lm = self.mfc(fi)  # [N, 2]
        lb = self.bfc(fi)  # [N, K]
        r["lm"] = lm
        r["lb"] = lb
        if logit: return r

        Pm = self.sigma(lm)
        Pb = self.sigma(lb)
        r["pm"] = Pm
        r["pb"] = Pb
        return r


@autoPropertyClass
class ToyNetV1(nn.Module, SegmentSupported, MultiBranch):
    """
    ToyNetV1 does not deal too much with BIRADs task. 
    It just apply an usual CE supervise on it.
    """

    cmgr: CSG
    __version__ = (1, 2)

    def __init__(self, cmgr: CSG, cps: CPS, in_channel, *args, margin=0.3, **kwargs):
        nn.Module.__init__(self)
        self.ynet = BIRADsYNet(cps, in_channel, *args, **kwargs)
        # online augment
        aug_conf = kwargs.get('aug_conf', {})
        self.aug = RandomAffine(
            aug_conf.get('degrees', 10),
            aug_conf.get('translate', .2),
            aug_conf.get('scale', (0.8, 1.1)),
        )
        # loss bases
        self.triplet = TripletBase(cmgr, margin, False)
        self.ce = CEBase(cmgr, self)
        self.segmse = MSESegBase(cmgr)
        self.siamese = SiameseBase(cmgr, self, self.ynet.yoc, self.segmse)

    @noexcept
    def forward(self, *args, **kwargs):
        return self.ynet.forward(*args, **kwargs)

    @noexcept
    def loss(self, meta: dict, X, Ym, Yb=None, mask=None) -> tuple:
        """return the result asis and a loss dict.

        Args:
            meta (dict): batch meta
            X (Tensor): [description]
            Ym (Tensor): [description]
            Yb (Tensor, optional): [description]. Defaults to None.
            mask (Tensor, optional): [description]. Defaults to None.

        Returns:
            tuple: [description]
        """
        need_seg = mask is not None

        aX, amask = self.aug(X, mask) if need_seg else (self.aug(X), None)
        r: dict = self.ynet(X, segment=need_seg, classify=True, logit=True)
        r2 = self.ynet(aX, segment=need_seg, classify=True, logit=True)

        loss = self.ce(r["lm"], r["lb"], Ym, Yb)

        # loss |= self.siamese(X, r['fi'], r2["fi"])
        if need_seg:
            loss |= self.segmse(r["seg"], mask)
            loss['seg_aug'] = self.segmse(r2['seg'], amask, value_only=True)

        if meta["balanced"]:
            loss |= self.triplet(r["ft"], Ym)
            # TODO: triplet of BIRADs

        return r, self.reduceLoss(loss, meta['augindices'])

    def reduceLoss(self, loss: dict, aug_indices, w: float = 1 / 3) -> dict:
        """[summary]

        Args:
            loss (dict): [description]
            aug_indices (Tensor): [description]
            w (float, optional): [description]. Defaults to 1/3.

        Returns:
            dict: [description]
        """
        device = first(loss.values()).device
        aug_mask = torch.tensor(aug_indices, dtype=torch.float, device=device)
        batch_weight = 1 - (1 - w) * aug_mask
        # TODO
        return {k: (batch_weight * v).sum() for k, v in loss.items()}

    def multiTaskLoss(self, loss: dict):
        """[summary]

        Args:
            loss (dict): [description]

        Returns:
            Tensor: [description]
        """
        cum_loss = 0
        for k, v in loss.items():
            cum_loss = cum_loss + v * self.cmgr.get("task." + k, 1)
        return cum_loss

    @staticmethod
    def lossSummary(loss: dict) -> dict:
        """[summary]

        Args:
            loss (dict): [description]

        Returns:
            dict: [description]
        """
        itemdic = {
            "pm": "m/CE",
            "tm": "m/triplet",
            "sim": "siamise/neg_cos_similarity",
            "seg": "segment/mse",
            "pb": "b/CE",
            "tb": "b/triplet",
            "seg_aug": 'segment/mse_aug'
        }
        return {"loss/" + v: loss[k].detach() for k, v in itemdic.items() if k in loss}

    @property
    def branches(self):
        return self.ynet.branches

    def branch_weight(self, weight_decay: dict):
        d = self.ynet.branch_weight(weight_decay)
        p = self.siamese.parameters()
        d['M'] += p
        return d
