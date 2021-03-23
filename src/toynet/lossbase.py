"""
For simplify and cleanify code in toynetv*, 
Loss bases will hold all layers and loss modules for calculating losses. 
But the bases are not module.

* author: JamzumSum
* create: 2021-3-15
"""
from abc import ABC, abstractmethod
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
from common.decorators import autoPropertyClass
from common.layers import MLP
from common.loss import F, focal_smooth_bce, focal_smooth_ce
from common.loss.triplet import WeightedExampleTripletLoss
from misc import CoefficientScheduler as CSG

__all__ = ["LossBase", "SiameseBase", "TripletBase", "CEBase", "MSESegBase"]
first = lambda it: next(iter(it))


class LossBase(ABC):
    def __init__(self, cmgr: CSG):
        self.cmgr = cmgr

    def __call__(self, *args, value_only=False, **kwargs):
        d = self.__loss__(*args, **kwargs)
        if value_only:
            assert len(d) == 1
            return first(d.values())

    @abstractmethod
    def __loss__(self, *args, **kwargs) -> dict:
        pass


class SiameseBase(LossBase):
    def __init__(self, cmgr, holder, zdim: int, msebase=None, **aug_conf):
        super().__init__(cmgr)
        self.p = holder
        self.mse = msebase or MSESegBase(cmgr)
        holder.pred_mlp = MLP(zdim, zdim, [512], False, False) # L2

    @staticmethod
    def neg_cos_sim(p, z):
        z = z.detach()     # stop-grad
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return -(p * z).sum(dim=1).mean()

    def __loss__(self, X, fi, fi2):
        # negative cosine similarity with pred_mlp.
        # D is to be used as a symmetrized loss.
        D = lambda p, z: self.neg_cos_sim(self.p.pred_mlp(p), z)
        return {"sim": (D(fi, fi2) + D(fi2, fi)) / 2}

    def parameters(self):
        return self.p.pred_mlp.parameters()


class TripletBase(LossBase):
    def __init__(self, cmgr, margin=0.3, normalize=True):
        super().__init__(cmgr)
        self.enable = margin > 0
        self.triplet = WeightedExampleTripletLoss(margin, normalize)

    def __loss__(self, ft, Ym):
        return {"tm": self.triplet.forward(ft, Ym)} if self.enable else {}


class CEBase(LossBase):
    def __init__(self, cmgr, holder: nn.Module):
        super().__init__(cmgr)
        # TODO: weights should be set by config...
        holder.register_buffer("mweight", torch.Tensor([0.4, 0.6]))
        holder.register_buffer("bweight", torch.Tensor([0.1, 0.2, 0.2, 0.2, 0.2, 0.1]))
        self.p = holder

    def __loss__(self, lm, lb, ym, yb=None):
        d = {"pm": F.cross_entropy(lm, ym, weight=self.p.mweight)}
        if yb is not None:
            gamma_b = self.cmgr.get("gamma_b", self.cmgr.piter + 1)
            d["pb"] = focal_smooth_bce(lb, yb, gamma=gamma_b, weight=self.p.bweight)
        return d


class MSESegBase(LossBase):
    def __init__(self, cmgr):
        super().__init__(cmgr)

    def __loss__(self, seg=None, mask=None):
        if mask is None or seg is None:
            return {}
        weight_focus_pos = self.cmgr.get("weight_focus_pos", 1 - self.cmgr.piter ** 2)
        return {"seg": ((seg - mask) ** 2 * (weight_focus_pos * mask + 1)).mean()}
