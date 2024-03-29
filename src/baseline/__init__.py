import torch.nn as nn
import torchvision.models.resnet as resnet
from common.optimizer import get_arg_name, get_need_decay, split_upto_decay
from common.support import MultiBranch
from data.augment.online import RandomAffine
from misc import CheckpointSupport as CPS
from misc import CoefficientScheduler as CSG
from torch import Tensor
from toynet.lossbase import CEBase, MultiTask
from toynet.lossbase.loss import ConsistencyBase, SiameseBase, TripletBase


class BNNeckHook:
    __slots__ = ('ft', 'fi', 'bn')

    def __init__(self, bn: nn.BatchNorm1d):
        self.bn = bn

    def __call__(self, module, input):
        self.ft = input[0]
        self.fi = self.bn(self.ft)
        return self.fi


class SimBack(nn.Module, MultiTask):
    '''
    A simple Backbone. 
    '''
    package = resnet

    def __init__(
        self,
        cmgr: CSG,
        cps: CPS,
        K: int,
        model: str,
        *,
        pretrain_path: dict = None,
        aug_conf=None,
        aug_weight=0.3333,
        zdim=2048,
        smooth=0.,
        **kwargs
    ):
        nn.Module.__init__(self)
        MultiTask.__init__(self, cmgr, aug_weight)
        self.cps = cps

        self.sigma = nn.Softmax(dim=1)
        self.mbranch = self.getBackbone(model, 2, **kwargs)
        self.bn = nn.BatchNorm1d(self._getFC().in_features)
        self.fi_mhook = self.fiHook(self.bn)

        # online augment
        if aug_conf is None: aug_conf = {}
        self.aug = RandomAffine(
            0,
            aug_conf.get('translate', .2),
            aug_conf.get('scale', (0.8, 1.1)),
        )

        self.ce = CEBase(cmgr, K, smooth)
        self.triplet = TripletBase(cmgr, True)

        isenable = lambda task: (not cmgr.isConstant(f'task.{task}')
                                 ) or cmgr.get(f"task.{task}", 1) != 0

        assert isenable('pm')
        self.enable_siam = isenable('sim')

        if not isenable('tm'): self.triplet.enable = False
        if self.enable_siam:
            self.siamese = SiameseBase(cmgr, self._getFC().in_features, zdim)

    def _getFC(self, model=None) -> nn.Linear:
        if model is None: model = self.mbranch
        fc = {
            'resnet': 'fc',
            'densenet': 'classifier'
        }[self.package.__name__[len(self.package.__package__) + 1:]]
        return model._modules[fc]

    def fiHook(self, bn):
        fibuf = BNNeckHook(bn)
        self._getFC().register_forward_pre_hook(fibuf)
        return fibuf

    def getBackbone(self, arch: str, num_classes: int, **kwargs) -> nn.Module:
        return getattr(self.package, arch)(num_classes=num_classes, **kwargs)

    def forward(self, X: Tensor, logit=False):
        X = X.repeat(1, 3, 1, 1)   # [N, 3, H, W]
        d = {}
        d['lm'] = self.mbranch(X)
        d['ft'] = self.fi_mhook.ft
        d['fi'] = self.fi_mhook.fi
        if logit: return d

        d['pm'] = self.sigma(d['lm'])
        return d

    def __loss__(self, meta, X, Ym, reduce=True, *args, **argv):
        r = self.forward(X)
        if self.enable_siam:
            aX = self.aug(X)
            r2 = self.forward(aX, logit=True)

        loss = self.ce(r['lm'], None, Ym, None)

        if meta["balanced"]:
            loss |= self.triplet(r['ft'], Ym)

        if self.enable_siam:
            loss |= self.siamese(r['fi'], r2['fi'])

        if reduce: loss = self.reduceLoss(loss, meta['augindices'])
        return r, loss


class ParalBack(SimBack, MultiBranch):
    '''
    A naive holder of two parallel backbones. 
    '''
    def __init__(self, cmgr: CSG, cps: CPS, K, model: tuple, **kwargs):
        if isinstance(model, str): model = (model, model)
        SimBack.__init__(self, cmgr, cps, K, model[0], **kwargs)

        aname = get_arg_name(SimBack.__init__)[1:]
        self.bbranch = self.getBackbone(
            model[1], K, **{k: v
                            for k, v in kwargs.items() if k not in aname}
        )

        self.dis = ConsistencyBase(cmgr, K)
        
        isenable = lambda task: (not cmgr.isConstant(f'task.{task}')
                                 ) or cmgr.get(f"task.{task}", 1) != 0

        assert isenable('pb')

    def forward(self, X, logit=False):
        d = SimBack.forward(self, X, logit)
        d['lb'] = self.bbranch(X.repeat(1, 3, 1, 1))
        if logit: return d

        d['pb'] = self.sigma(d['lb'])
        return d

    def __loss__(self, meta, X, Ym, Yb=None, reduce=True, *args, **argv):
        r, loss = super().__loss__(meta, X, Ym, False, *args, **argv)
        loss |= self.ce(r['lm'], r['lb'], Ym, Yb)
        loss |= self.dis(r['pm'], r['pb'])
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
