from typing import Iterable

import torch
import torch.nn.functional as F
from common.decorators import d3support
from torchvision import transforms

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

    @staticmethod
    def get_params(degrees, translate, scale_ranges):
        """Get parameters for affine transformation

        Returns:
            params to be passed to the affine transformation
        """
        angle = float(
            torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item()
        )
        if translate is not None:
            max_dx = float(translate[0])
            max_dy = float(translate[1])
            tx = torch.empty(1).uniform_(-max_dx, max_dx).item()
            ty = torch.empty(1).uniform_(-max_dy, max_dy).item()
            translations = (tx, ty)
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = float(
                torch.empty(1).uniform_(scale_ranges[0], scale_ranges[1]).item()
            )
        else:
            scale = 1.0
        return angle, translations, scale

    def _affine(self, img):
        angle, (dx, dy), scale = self.get_params(
            self.degrees,
            self.translate,
            self.scale,
        )
        return affine(img, dx, dy, scale, angle)


class RandomSimple(transforms.Compose):
    def __init__(self, size=512):
        super().__init__([
            transforms.RandomResizedCrop(size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
        ])

    def __call__(self, *img):
        if len(img) == 0:
            raise ValueError
        elif len(img) == 1:
            return super().__call__(img[0])

        N = [i.size(0) for i in img]
        img = torch.cat(img, 0)
        r = super().__call__(img)
        return torch.split(r, N)
