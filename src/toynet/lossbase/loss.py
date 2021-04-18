"""
For simplify and cleanify code in toynetv*, 
Loss bases will hold all layers and loss modules for calculating losses. 
But the bases are not module.

* author: JamzumSum
* create: 2021-3-15
"""
from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
from common import freeze
from common.layers import MLP
from common.loss import F, focal_smooth_bce, focal_smooth_ce
from common.loss.triplet import WeightedExampleTripletLoss
from misc import CoefficientScheduler as CSG

_S2T = dict[str, torch.Tensor]

__all__ = ["SiameseBase", "TripletBase", "CEBase", "MSESegBase", "IdentityBase"]


class LossBase(ABC, nn.Module):
    enable = True

    def __init__(self, cmgr: CSG):
        nn.Module.__init__(self)
        self.cmgr = cmgr

    def __call__(self, *args, **kwds) -> _S2T:
        return super().__call__(*args, **kwds) if self.enable else {}


class IdentityBase(LossBase):
    def __init__(self, cmgr: CSG, *args, **kwargs):
        super().__init__(cmgr)

    def forward(self, *args, **kwargs):
        return {}


class SiameseBase(LossBase):
    def __init__(self, cmgr, D, zdim: int = 2048):
        super().__init__(cmgr)
        self.proj_mlp = MLP(D, zdim, [2048, 2048])    # L3
        self.pred_mlp = MLP(zdim, zdim, [512], False) # L2

    @staticmethod
    def neg_cos_sim(p, z):
        z = z.detach()     # stop-grad
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return -(p * z).sum(dim=1)

    def forward(self, fi, fi2):
        """calculate negative cosine similarity of two feature.

        Args:
            fi (Tensor[float]): embedding 1
            fi2 (Tensor[float]): embedding 2

        Returns:
            Tensor[float]: symmetrized negative cosine similarity
        """
        z1 = self.proj_mlp(fi)
        z2 = self.proj_mlp(fi2)
        # negative cosine similarity with pred_mlp.
        # D is to be used as a symmetrized loss.
        D = lambda p, z: self.neg_cos_sim(self.pred_mlp(p), z)
        return {"sim": (D(z1, z2) + D(z2, z1)) / 2}


class TripletBase(LossBase):
    def __init__(self, cmgr, normalize=True):
        super().__init__(cmgr)
        margin = self.cmgr.get('margin', 0.3)
        self.enable = margin != 0 or not self.cmgr.isConstant('margin')
        self.triplet = WeightedExampleTripletLoss(margin, normalize, 'none')

    def forward(self, ft, Ym):
        self.triplet.margin = self.cmgr.get('margin', self.triplet.margin)
        return {"tm": self.triplet.forward(ft, Ym)} if self.enable else {}


class CEBase(LossBase):
    def __init__(self, cmgr, smooth=0.1):
        super().__init__(cmgr)
        # TODO: weights should be set by config...
        self.register_buffer("mweight", torch.Tensor([0.4, 0.6]))
        self.register_buffer("bweight", torch.Tensor([0.1, 0.2, 0.2, 0.2, 0.2, 0.1]))
        self.smooth = smooth

    def forward(self, lm, lb, ym, yb=None):
        d = {
            "pm": focal_smooth_ce(
                lm, ym, 0, self.smooth, weight=self.mweight, reduction='none'
            )
        }
        if yb is None: return d

        d["pb"] = focal_smooth_ce(
            freeze(lb, 1),                           # NOTE: lb is detached here.
            yb,
            gamma=self.cmgr.get("gamma_b", 'x + 1'),
            weight=self.bweight,
            reduction='none'
        )
        return d


class MSESegBase(LossBase):
    """[summary]

        Args:
            seg (Tensor[float], optional): [description]. Defaults to None.
            mask (Tensor[float], optional): [description]. Defaults to None.

        Returns:
            Tensor: [description]
        """
    def forward(self, seg=None, mask=None):
        if mask is None or seg is None:
            return {}
        weight_focus_pos = self.cmgr.get("weight_focus_pos", '1 - x ** 2')
        weight = weight_focus_pos * mask + 1
        return {"seg": ((seg - mask) ** 2 * weight).mean(dim=(1, 2, 3))}
