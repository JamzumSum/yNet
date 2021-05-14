from itertools import product

import torch
from torch._C import Size
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

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
    cond = torch.stack(cond)       # [8, :]
    B = torch.sum(torch.stack(B) * cond, dim=0)
    G = torch.sum(torch.stack(G) * cond, dim=0)
    R = torch.sum(torch.stack(R) * cond, dim=0)
    O = torch.stack([R, G, B], dim=-3) / 255
    return unsqueeze_as(x >= thresh * 255, O, 1) * O


def morph_close(X, kernel=3, iteration=1):
    # type: (Tensor, int, int) -> Tensor
    assert kernel & 1
    mp = torch.nn.MaxPool2d(kernel, 1, (kernel - 1) // 2)

    for _ in range(iteration):
        X = -mp.forward(-mp.forward(X))

    return X


class BoundingBox(nn.Module):
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.thresh = threshold

    def forward(self, mask: Tensor) -> Tensor:
        """calculate bounding box of a mask.

        Args:
            mask (Tensor): [N, 1, H, W]

        Returns:
            Tensor: [N, 4] (y_min, x_min, y_max, x_max)
        """
        assert mask.dim() == 4
        mask = mask.squeeze(1)                       # [N, H, W]
        N, H, W = mask.shape
        wmat = mask.max(dim=1).values >= self.thresh # [N, W]
        hmat = mask.max(dim=2).values >= self.thresh # [N, H]
        wmat = [torch.nonzero(i) for i in wmat]      # N * [?, 1]
        hmat = [torch.nonzero(i) for i in hmat]      # N * [?, 1]

        xmin, xmax, ymin, ymax = [], [], [], []
        for i in wmat:
            if i.shape[0]:
                xmin.append(i.min())
                xmax.append(i.max() + 1)
            else:
                xmin.append(torch.zeros((), dtype=i.dtype))
                xmax.append(torch.empty((), dtype=i.dtype).fill_(W))
        for i in hmat:
            if i.shape[0]:
                ymin.append(i.min())
                ymax.append(i.max() + 1)
            else:
                ymin.append(torch.zeros((), dtype=i.dtype))
                ymax.append(torch.empty((), dtype=i.dtype).fill_(H))
        
        xmin = torch.stack(xmin)
        xmax = torch.stack(xmax)
        ymin = torch.stack(ymin)
        ymax = torch.stack(ymax)

        yxmin = torch.stack((ymin, xmin), dim=-1)
        yxmax = torch.stack((ymax, xmax), dim=-1)
        return torch.cat((yxmin, yxmax), dim=-1)


class CropUpsample(nn.Module):
    def __init__(self, size: tuple, threshold: float = 0.5, mode='bilinear'):
        super().__init__()
        self._bbox = BoundingBox(threshold)
        self._up = nn.Upsample(size, mode=mode, align_corners=False)

    def crop(self, X, bbox) -> list[Tensor]:
        """crop X according to bbox

        Args:
            X (Tensor): [N, C, H, W]
            bbox (Tensor): [N, 4]

        Returns:
            list[Tensor]: croped tensors
        """

        # bug: crop may change the shape of an image, even if the crop size is correct.
        return [
            i[..., y_min:y_max, x_min:x_max].unsqueeze(0)
            for i, (y_min, x_min, y_max, x_max) in zip(X, self.pad(bbox))
        ]

    def pad(self, bbox: Tensor) -> Tensor:
        """pad X to target shape

        Args:
            bbox (Tensor): [N, 4]

        Returns:
            Tensor: [N, 4]
        """
        assert bbox.dim() == 2, bbox.shape

        L = self._up.size[0] / self._up.size[1] if isinstance(
            self._up.size, (tuple, list)
        ) else 1
        hw = self.hw(bbox)         # [N, 2]
        HW = torch.stack([hw[:, 1] * L, hw[:, 0] / L], dim=-1).round_().int()
        rest = (HW - hw).relu_()
        rest1 = torch.stack((rest // 2, bbox[:, :2]), dim=0).min(dim=0).values
        rest2 = rest - rest1
        return torch.cat([bbox[:, :2] - rest1, bbox[:, 2:] + rest2], dim=-1)

    def forward(self, X, mask: Tensor) -> Tensor:
        """[summary]

        Args:
            X (Tensor): feature map [N, *, H, W]
            mask (Tensor): [N, 1, H, W]
 
        Returns:
            Tensor: [description]
        """
        self._bbox_buf = self._bbox(mask)
        croped = self.crop(X, self._bbox_buf)
        return torch.cat([self._up(i) for i in croped], dim=0)

    @property
    def bbox_buf(self) -> Tensor:
        return self._bbox_buf

    @staticmethod
    def hw(bbox):
        """calculate HW of a bbox

        Args:
            bbox (Tensor): [N, 4]

        Returns:
            Tensor: [N, 2]
        """
        return bbox[:, 2:] - bbox[:, :2]
