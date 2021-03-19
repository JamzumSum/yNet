from typing import Iterable

import torch
import torch.nn.functional as F
from common.decorators import d3support
from torchvision.transforms import RandomAffine as RA
from common.decorators import autoPropertyClass
from . import affine


class RandomAffine(torch.nn.Module):
    def __init__(self, degrees, translate=None, scale=None):
        isnum = lambda x: isinstance(x, (int, float))
        if isnum(degrees):
            degrees = abs(degrees)
            degrees = (-degrees, degrees)
        if isnum(translate):
            assert translate >= 0
            translate = (translate, translate)
        if isnum(scale):
            assert scale > 0
            scale = (scale, 1) if scale < 1 else (1, scale)

        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        super().__init__()

    def forward(self, *img):
        if len(img) == 0:
            raise ValueError
        elif len(img) == 1:
            return self._affine(img[0])

        N = [i.size(0) for i in img]
        img = torch.cat(img, 0)
        r = self._affine(img)
        return torch.split(r, N)

    def _affine(self, img):
        WH = [img.shape[-1], img.shape[-2]]
        angle, (dx, dy), scale, _ = RA.get_params(
            self.degrees,
            self.translate,
            self.scale,
            None,
            WH,
        )
        return affine(img, dx, dy, scale, angle)
