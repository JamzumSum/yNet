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
            X: [C, H, W]
            box: [4]
        return:
            [C, h, w]
        """
        return X[:, box[1] : box[3], box[0] : box[2]]

    def unitwh(self, box):
        """
        args:
            box: [N, 4]. (x_min, y_min, x_max, y_max)
        return:
            wh: [N, 2]
        """
        wh = self.wh(box)  # [N, 2]
        wh = (wh / self.unit).ceil()
        wh[wh == 0] = 1
        wh = wh * self.unit
        return wh.int()

    def finebox(self, X, box, thresh):
        """
        args:
            BX: [N, H, W]
            box/ubox: [N, 4]. (x_min, y_min, x_max, y_max)
        return:
            finebox: [N, 4]
        """
        BX = X
        BX[BX >= thresh] = 1.0

        uwh = self.unitwh(box)
        wh = self.wh(box)
        extra_wh = uwh - wh  # [N, 2]
        extra_wh[extra_wh < 0] = 0
        extra_box = box  # [N, 4]
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

    def selectThresh(self, X):
        # TODO
        return X.min() + (X.max() - X.min()) * self.tr

    @NoGrad
    def forward(self, mask, X=None):
        """
        aegs:
            mask: [N, 1, H, W]
            X: [N, C, H, W]
        return:
            N * [C, h, w]
        """
        _, _, H, W = mask.shape
        thresh = self.selectThresh(mask)
        BM = mask.squeeze(1) >= thresh  # [N, H, W]

        y = torch.empty(1, H, dtype=torch.int)
        y[0, :] = torch.arange(H, dtype=torch.int)
        x = torch.empty(1, W, dtype=torch.int)
        x[0, :] = torch.arange(W, dtype=torch.int)

        x = BM.sum(dim=-2).bool() * x  # [N, W]
        y = BM.sum(dim=-1).bool() * y  # [N, H]

        def mine0(t, dim):
            inf = t.max() + 1
            t[t == 0] = inf
            return torch.argmin(t, dim)

        y_max = torch.argmax(y, dim=-1)  # [N]
        y_min = mine0(y, dim=-1)
        x_max = torch.argmax(x, dim=-1)
        x_min = mine0(x, dim=-1)

        box = torch.stack((x_min, y_min, x_max, y_max), dim=-1)  # [N, 4]
        box = self.finebox(mask, box, thresh)

        return [self.crop(xi, bi) for xi, bi in zip(mask if X is None else X, box)]

