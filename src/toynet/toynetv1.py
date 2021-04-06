"""
A toy implement for classifying benign/malignant and BIRADs

* author: JamzumSum
* create: 2021-1-11
"""
import torch
import torch.nn as nn
from common.optimizer import get_need_decay, split_upto_decay
from common.support import *
from data.augment.online import RandomAffine
from misc import CheckpointSupport as CPS
from misc import CoefficientScheduler as CSG
from misc.decorators import autoPropertyClass

from .lossbase import MultiTask
from .lossbase.loss import *
from .ynet import YNet

first = lambda it: next(iter(it))
value_only = lambda d: first(d.values())


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
        YNet.__init__(self, cps, in_channel, width, ulevel, norm=norm, **kwargs)
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
        paramB = list(self.bfc.parameters())
        paramM = [p for p in self.parameters() if id(p) not in [id(i) for i in paramB]]
        paramdic = {"M": paramM, "B": paramB}
        # a param dict when not filter by `weight_decay`
        if not any(weight_decay.values()):
            return paramdic

        return split_upto_decay(get_need_decay(self.modules()), paramdic, weight_decay)

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

        fi = self.norm_layer(ft)   # [N, D], empirically, D >= 128
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


class ToyNetV1(nn.Module, SegmentSupported, MultiBranch, MultiTask):
    """
    ToyNetV1 does not deal too much with BIRADs task. 
    It just apply an usual CE supervise on it.
    """
    def __init__(
        self,
        cmgr: CSG,
        cps: CPS,
        in_channel,
        *args,
        zdim=2048,
        aug_weight=0.3333,
        aug_conf=None,
        smooth=0.,
        **kwargs
    ):
        nn.Module.__init__(self)
        MultiTask.__init__(self, cmgr, aug_weight)
        self.cps = cps
        self.ynet = BIRADsYNet(cps, in_channel, *args, **kwargs)
        # online augment
        if aug_conf is None: aug_conf = {}
        self.aug = RandomAffine(
            aug_conf.get('degrees', 0),
            aug_conf.get('translate', .2),
            aug_conf.get('scale', (0.8, 1.1)),
        )
        # loss bases
        self.triplet = TripletBase(cmgr, True)
        self.ce = CEBase(cmgr, smooth)
        self.seg = MSESegBase(cmgr)
        self.siamese = SiameseBase(cmgr, self.ynet.yoc, zdim)

        isenable = lambda task: {
            True: cmgr.get(f"task.{task}", 1) != 0,
            False: False,
            None: True
        }[cmgr.isConstant(f'task.{task}')]

        assert isenable('pm')
        self.enable_seg = isinstance(self, SegmentSupported) and isenable('seg')
        self.enable_sa = isinstance(self, SegmentSupported) and isenable('seg_aug')
        self.enable_siam = isenable('sim')

        if not self.enable_seg: self.seg.enable = False
        if not isenable('tm'): self.triplet.enable = False
        if not self.enable_siam: self.siamese.enable = False

    def forward(self, *args, **kwargs):
        return self.ynet.forward(*args, **kwargs)

    def __loss__(self, meta: dict, X, Ym, Yb=None, mask=None, reduce=True) -> tuple:
        """return the result asis and a loss dict.

        Args:
            meta (dict): batch meta
            X (Tensor): image
            Ym (Tensor): [description]
            Yb (Tensor, optional): [description]. Defaults to None.
            mask (Tensor, optional): ground-truth. Defaults to None.
            reduce (bool, optional): whether to reduce loss dict. Defaults to True.

        Returns:
            tuple: resultdic, lossdic
        """
        need_seg = self.enable_seg and mask is not None
        need_sa = need_seg and self.enable_sa

        r: dict = self.ynet(X, segment=need_seg, classify=True, logit=True)
        if need_sa or self.enable_siam:
            aX, amask = self.aug(X, mask) if need_sa else (self.aug(X), None)
            r2 = self.ynet(aX, segment=need_sa, classify=self.enable_siam, logit=True)

        loss = self.ce(r["lm"], r["lb"], Ym, Yb)
        loss |= self.seg(r.get('seg', None), mask)

        if self.enable_siam:
            loss |= self.siamese(r['fi'], r2['fi'])

        if need_sa:
            loss['seg_aug'] = value_only(self.seg(r2['seg'], amask))

        if meta["balanced"]:
            loss |= self.triplet(r["ft"], Ym)

        if reduce: loss = self.reduceLoss(loss, meta['augindices'])
        return r, loss

    @property
    def branches(self):
        return self.ynet.branches

    def branch_weight(self, weight_decay: dict):
        d = self.ynet.branch_weight(weight_decay)
        p = self.siamese.parameters()
        iorf = lambda s: s in d and s
        d[iorf('M_no_decay') or iorf('M')].extend(p)
        return d
