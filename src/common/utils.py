from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import unsqueeze_as
from .decorators import NoGrad


@torch.jit.script
def gray2JET(x, thresh=0.5):
    # type: (Tensor, float) -> Tensor
    """
    - x: [..., H, W],       NOTE: float 0~1
    - O: [..., 3, H, W],    NOTE: BGR, float 0~1
    """
    x = 255 * x
    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)
    B = [
        128 + 4 * x,
        255 * ones,
        255 * ones,
        254 * ones,
        638 - 4 * x,
        ones,
        zeros,
        zeros,
    ]
    G = [
        zeros,
        zeros,
        4 * x - 128,
        255 * ones,
        255 * ones,
        255 * ones,
        892 - 4 * x,
        zeros,
    ]
    R = [
        zeros,
        zeros,
        zeros,
        2 * ones,
        4 * x - 382,
        254 * ones,
        255 * ones,
        1148 - 4 * x,
    ]
    cond = [
        x < 31,
        x == 32,
        (33 <= x) * (x <= 95),
        x == 96,
        (97 <= x) * (x <= 158),
        x == 159,
        (160 <= x) * (x <= 223),
        224 <= x,
    ]
    cond = torch.stack(cond)  # [8, :]
    B = torch.sum(torch.stack(B) * cond, dim=0)
    G = torch.sum(torch.stack(G) * cond, dim=0)
    R = torch.sum(torch.stack(R) * cond, dim=0)
    O = torch.stack([R, G, B], dim=-3) / 255
    return unsqueeze_as(x > thresh * 255, O, 1) * O


class ConfusionMatrix:
    def __init__(self, K=None, smooth=1e-8):
        if K:
            self._m = torch.zeros(K, K, dtype=torch.int)
        self.K = K
        self.eps = smooth

    @property
    def initiated(self):
        return hasattr(self, "_m")

    @property
    def N(self):
        return self._m.sum() if self.initiated else None

    def add(self, P, Y):
        """P&Y: [N]"""
        INTTYPE = (torch.int, torch.int64, torch.int16)
        assert P.dtype in INTTYPE
        assert Y.dtype in INTTYPE
        if self.K:
            K = self.K
        else:
            self.K = K = int(Y.max())

        if not self.initiated:
            self._m = torch.zeros(K, K, dtype=torch.int, device=P.device)
        if self._m.device != P.device:
            self._m = self._m.to(P.device)
        if Y.device != P.device:
            Y = Y.to(P.device)

        for i, j in product(range(K), range(K)):
            self._m[i, j] += ((P == i) * (Y == j)).sum()

    def accuracy(self):
        acc = self._m.diag().sum()
        return acc / self._m.sum()

    def err(self, *args, **kwargs):
        return 1 - self.accuracy(*args, **kwargs)

    def precision(self, reduction="none"):
        acc = self._m.diag()
        prc = acc / (self._m.sum(dim=1) + self.eps)
        if reduction == "mean":
            return prc.mean()
        else:
            return prc

    def recall(self, reduction="none"):
        acc = self._m.diag()
        rec = acc / (
            self._m.sum()
            - self._m.sum(dim=0)
            - self._m.sum(dim=1)
            + self._m.diag()
            + self.eps
        )
        if reduction == "mean":
            return rec.mean()
        else:
            return rec

    def fscore(self, beta=1, reduction="mean"):
        P = self.precision()
        R = self.recall()
        f = (1 + beta * beta) * (P * R) / (beta * beta * P + R + self.eps)
        if reduction == "mean":
            return f.mean()
        else:
            return f

    def mat(self):
        return self._m / self._m.max()


class SEBlock(nn.Sequential):
    def __init__(self, L, hs=128):
        nn.Sequential.__init__(
            self,
            nn.Linear(L, hs),
            nn.ReLU(),
            nn.Linear(hs, L),
            nn.Softmax(dim=-1)
            # use softmax instead of sigmoid here since the attention-ed channels are sumed,
            # while the sum might be greater than 1 if sum of the attention vector is not restricted.
        )
        nn.init.constant_(self[2].bias, 1 / L)

    def forward(self, X):
        """
        X: [N, K, H, W, L]
        O: [N, K, H, W]
        """
        X = X.permute(4, 0, 1, 2, 3)  # [L, N, K, H, W]
        Xp = F.adaptive_avg_pool2d(X, (1, 1))  # [L, N, K, 1, 1]
        Xp = Xp.permute(1, 2, 3, 4, 0)  # [N, K, 1, 1, L]
        Xp = nn.Sequential.forward(self, Xp).permute(4, 0, 1, 2, 3)  # [L, N, K, 1, 1]
        return (X * Xp).sum(dim=0)


class PyramidPooling(nn.Module):
    """
    Use pyramid pooling instead of max-pooling to make sure more elements in CAM can be backward. 
    Otherwise only the patch with maximum average confidence has grad while patches and small.
    Moreover, the size of patches are fixed so is hard to select. Multi-scaled patches are suitable.
    """

    def __init__(self, patch_sizes, hs=128):
        nn.Module.__init__(self)
        if any(i & 1 for i in patch_sizes):
            print(
                """Warning: At least one value in `patch_sizes` is odd. 
            Channel-wise align may behave incorrectly."""
            )
        self.patch_sizes = sorted(patch_sizes)
        self.atn = SEBlock(self.L, hs)

    @property
    def L(self):
        return len(self.patch_sizes)

    def forward(self, X):
        """
        X: [N, C, H, W]
        O: [N, K, 2 * H//P_0 -1, 2 * W//P_0 - 1]
        """
        # set stride as P/2, so that patches overlaps each other
        # hopes to counterbalance the lack-representating of edge pixels of a patch.
        ls = [
            F.avg_pool2d(X, patch_size, patch_size // 2)
            for patch_size in self.patch_sizes
        ]
        base = ls.pop(0)  # [N, K, H//P0, W//P0]
        ls = [F.interpolate(i, base.shape[-2:], mode="nearest") for i in ls]
        ls.insert(0, base)
        ls = torch.stack(ls, dim=-1)  # [N, K, H//P0, W//P0, L]
        return self.atn(ls)


def morph_close(X, kernel=3, iteration=1):
    # type: (Tensor, int, int) -> Tensor
    assert kernel & 1
    mp = torch.nn.MaxPool2d(kernel, 1, (kernel - 1) // 2)

    for _ in range(iteration):
        X = -mp.forward(-mp.forward(X))

    return X


class BoundingBoxCrop(nn.Module):
    def __init__(self, threshold_rate, unit_scale=1):
        super().__init__()
        self.tr = threshold_rate
        self.register_buffer(
            "unit",
            torch.IntTensor(
                (unit_scale, unit_scale) if isinstance(unit_scale, int) else unit_scale
            ),
        )

    @staticmethod
    def wh(box):
        return box[..., 2:] - box[..., :2]

    @staticmethod
    def crop(X, box):
        """
        Arg:
            X: [H, W]
            box: [4]
        return:
            [h, w]
        """
        return X[box[1] : box[3], box[0] : box[2]]

    def unitwh(self, box):
        """
        args:
            box: [N*C, 4]. (x_min, y_min, x_max, y_max)
        return:
            wh: [N*C, 2]
        """
        wh = self.wh(box)  # [N*C, 2]
        wh = (wh / self.unit).ceil()
        wh[wh == 0] = 1
        wh = wh * self.unit
        return wh.int()

    def finebox(self, BX, box):
        """
        args:
            BX: [N*C, H, W]
            box/ubox: [N*C, 4]. (x_min, y_min, x_max, y_max)
        return:
            finebox: [N*C, 4]
        """
        uwh = self.unitwh(box)
        wh = self.wh(box)
        extra_wh = uwh - wh  # [N*C, 2]
        extra_wh[extra_wh < 0] = 0
        extra_box = box  # [N*C, 4]
        extra_box[:, :2] -= extra_wh
        extra_box[:, 2:] += extra_wh
        extra_box[extra_box < 0] = 0

        finebox = []
        for xi, bi, uwhi in zip(BX, extra_box, uwh):
            extra_crop = self.crop(xi, bi)  # [H, W]
            ker = (min(extra_crop.size(0), uwhi[1]), min(extra_crop.size(1), uwhi[0]))
            xmap = F.avg_pool2d(extra_crop.unsqueeze(0), ker, 1).squeeze(0)
            x_min = bi[0] + xmap.max(dim=0).values.argmax()
            y_min = bi[1] + xmap.max(dim=1).values.argmax()
            x_max = x_min + uwhi[0]
            y_max = y_min + uwhi[1]
            finebox.append(torch.stack((x_min, y_min, x_max, y_max)))
        return torch.stack(finebox)

    def selectCrop(self, X, box, thresh) -> list:
        """
        box: [N*C, 4]. (x_min, y_min, x_max, y_max)
        """
        BX = X
        BX[BX >= thresh] = 1.0
        box = self.finebox(BX, box)
        return [self.crop(xi, bi) for xi, bi in zip(X, box)]

    def selectThresh(self, X):
        # TODO
        return X.min() + (X.max() - X.min()) * self.tr

    @NoGrad
    def forward(self, X):
        """
        X: [N, C, H, W]
        return:
            N * [C * [h, w]]
        """
        N, C, H, W = X.shape
        X = X.reshape(N * C, H, W)
        thresh = self.selectThresh(X)
        BX = X >= thresh
        y = torch.empty(1, H, dtype=torch.int)
        y[0, :] = torch.arange(H, dtype=torch.int)
        x = torch.empty(1, W, dtype=torch.int)
        x[0, :] = torch.arange(W, dtype=torch.int)

        x = BX.sum(dim=-2).bool() * x  # [N*C, W]
        y = BX.sum(dim=-1).bool() * y  # [N*C, H]

        def mine0(t, dim):
            inf = t.max() + 1
            t[t == 0] = inf
            return torch.argmin(t, dim)

        y_max = torch.argmax(y, dim=-1)  # [N*C]
        y_min = mine0(y, dim=-1)
        x_max = torch.argmax(x, dim=-1)
        x_min = mine0(x, dim=-1)

        box = torch.stack((x_min, y_min, x_max, y_max), dim=-1)  # [N*C, 4]
        crop = self.selectCrop(X, box, thresh)
        return [crop[i : i + C] for i in range(0, N * C, C)]

