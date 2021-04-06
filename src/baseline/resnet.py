import torch
import torch.nn as nn
import torchvision.models.resnet as resnet
from common.optimizer import get_need_decay, split_upto_decay
from common.support import MultiBranch
from misc import CheckpointSupport as CPS
from misc import CoefficientScheduler as CSG
from toynet.lossbase import CEBase, MultiTask


class SimRes(torch.nn.Module, MultiTask):
    '''
    A simple ResNet. 
    Support: resnet18, resnet34, resnet50, resnet101, resnet152
    '''
    def __init__(self, cmgr: CSG, cps: CPS, model='resnet34', aug_weight=0.3333):
        torch.nn.Module.__init__(self)
        MultiTask.__init__(self, cmgr, aug_weight)
        self.sigma = nn.Softmax(dim=1)
        self.mbranch = getattr(resnet, model)(num_classes=2)

        self.cps = cps
        self.ce = CEBase(cmgr, self)

    def forward(self, X, logit=False):
        if X.size(1) == 1:
            X = X.repeat(1, 3, 1, 1)           # [N, 3, H, W]
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


class Resx2(SimRes, MultiBranch):
    '''
    A naive holder of two ResNet. 
    Support: resnet18, resnet34, resnet50, resnet101, resnet152
    '''
    def __init__(self, cmgr: CSG, cps: CPS, K, model='resnet34'):
        if isinstance(model, str): model = (model, model)
        SimRes.__init__(self, cmgr, cps, model[0])
        self.bbranch = getattr(resnet, model[1])(num_classes=K)

    def forward(self, X, logit=False):
        if X.size(1) == 1:
            X = X.repeat(1, 3, 1, 1)           # [N, 3, H, W]
        d = SimRes.forward(self, X, logit)
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
