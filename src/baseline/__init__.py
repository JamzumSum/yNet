import torch.nn as nn
import torchvision.models.resnet as resnet
from common.optimizer import get_need_decay, split_upto_decay
from common.support import MultiBranch
from misc import CheckpointSupport as CPS
from misc import CoefficientScheduler as CSG
from toynet.lossbase import CEBase, MultiTask


class SimBack(nn.Module, MultiTask):
    '''
    A simple Backbone. 
    '''
    package = resnet

    def __init__(self, cmgr: CSG, cps: CPS, model: str, aug_weight=0.3333, smooth=0.):
        nn.Module.__init__(self)
        MultiTask.__init__(self, cmgr, aug_weight)
        self.sigma = nn.Softmax(dim=1)
        self.mbranch = self.getBackbone(model, 2)

        self.cps = cps
        self.ce = CEBase(cmgr, smooth)

    @classmethod
    def getBackbone(cls, arch: str, num_classes: int) -> nn.Module:
        return getattr(cls.package, arch)(num_classes=num_classes)

    def forward(self, X, logit=False):
        X = X.repeat(1, 3, 1, 1)   # [N, 3, H, W]
        d = {}
        d['lm'] = self.mbranch(X)
        if logit: return d

        d['pm'] = self.sigma(d['lm'])
        return d

    def __loss__(self, meta, X, Ym, reduce=True, *args, **argv):
        r = self.forward(X, logit=True)
        lm = r['lm']
        loss = self.ce(lm, None, Ym, None)
        if reduce: loss = self.reduceLoss(loss, meta['augindices'])
        return r, loss


class ParalBack(SimBack, MultiBranch):
    '''
    A naive holder of two parallel backbones. 
    '''
    def __init__(self, cmgr: CSG, cps: CPS, K, model: tuple):
        if isinstance(model, str): model = (model, model)
        SimBack.__init__(self, cmgr, cps, model[0])
        self.bbranch = self.getBackbone(model[1], K)

    def forward(self, X, logit=False):
        X = X.repeat(1, 3, 1, 1)   # [N, 3, H, W]
        d = SimBack.forward(self, X, logit)
        d['lb'] = self.bbranch(X)
        if logit: return d

        d['pb'] = self.sigma(d['lb'])
        return d

    def __loss__(self, meta, X, Ym, Yb=None, reduce=True, *args, **argv):
        r = self.forward(X, logit=False)
        lm = r['lm']
        lb = r['lb']
        loss = self.ce(lm, lb, Ym, Yb)
        if reduce: loss = self.reduceLoss(loss, meta['augindices'])
        return r, loss

    @property
    def branches(self):
        return ('M', 'B')

    def branch_weight(self, weight_decay: dict):
        paramdic = {"M": self.mbranch.parameters(), "B": self.bbranch.parameters()}
        # a param dict when not filter by `weight_decay`
        if not any(weight_decay.values()):
            return paramdic

        return split_upto_decay(get_need_decay(self.modules()), paramdic, weight_decay)
