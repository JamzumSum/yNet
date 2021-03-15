import bisect
from abc import ABC, abstractmethod, abstractstaticmethod
from functools import wraps
from random import choice

import torch
from common.support import DeviceAwareness

from ..dataset import Distributed, DistributedConcatSet
from . import elastic, getGaussianFilter, scale, translate

first = lambda it: next(iter(it))


class VirtualDataset(Distributed, ABC):
    def argwhere(self, *args, **kwargs):
        raise NotImplementedError("Virual datasets cannot be query.")


def just_image(func):
    @wraps(func)
    def deformer(item: dict, *args, **kwargs):
        if "mask" in item:
            N = item["X"].size(0)
            X = torch.cat((item["X"], item["mask"]), dim=0)
        else:
            X = item["X"]

        X = func(X, *args, **kwargs)

        if "mask" in item:
            item["mask"] = X[N:]
            X = X[:N]
        item["X"] = X
        return item

    return deformer


class AugmentSet(VirtualDataset, DeviceAwareness, ABC):
    # BUG: Use cuda to augment data is necessary to some extent but conflicts with multiprocessing
    # in most times. I've not found a method to adopt the both.
    # see warning in https://pytorch.org/docs/stable/data.html#multi-process-data-loading
    def __init__(
        self,
        dataset: Distributed,
        distrib_title: str,
        aim_size=None,
        *,
        suffix: str,
        device=None,
    ):
        self.dataset = dataset
        self.suffix = suffix
        DeviceAwareness.__init__(self, device)
        assert not all(
            isinstance(i, dict) for i in dataset.meta.values()
        ), "Augment source set must be uniform."

        self.distrib_title = distrib_title
        Distributed.__init__(self, dataset.statTitle)

        if aim_size is None:
            avg = self.dataset.distribution.max().item()
        else:
            avg = aim_size // self.K(distrib_title)

        org_distrib = self.dataset.getDistribution(distrib_title)
        my_distrib = avg * torch.ones_like(org_distrib, dtype=torch.int) - org_distrib

        if torch.any(my_distrib < 0):
            print(
                "At least one class has samples more than %d // %d = %d."
                "These classes wont be sampled when augumenting."
                % (aim_size, self.K(distrib_title), avg)
            )
            my_distrib.clamp_(min=0)

        self._distrib = {distrib_title: my_distrib}
        for i in dataset.statTitle:
            if i == distrib_title:
                continue
            j = dataset.joint(distrib_title, i)
            j = j / j.sum(dim=1, keepdim=True)
            self._distrib[i] = (my_distrib.unsqueeze(1) * j).sum(dim=0).round_()

        self._indices = [
            dataset.argwhere(lambda d: d == i, distrib_title)
            for i in range(self.K(distrib_title))
        ]
        self.cum_distrib = self.cumsum(my_distrib)

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for l in sequence:
            r.append(l + s)
            s += l
        return r

    def __getitem__(self, i):
        k = bisect.bisect_right(self.cum_distrib, i)
        indices = self._indices[k]
        if self._fetch:
            x = self.deformation(self.dataset[choice(indices)])
        else:
            x = self.dataset[choice(indices)]
        x["meta"].aug = self.suffix
        return x

    def __len__(self):
        return self._distrib[self.distrib_title].sum().item()

    def K(self, title):
        return self.dataset.K(title)

    def getDistribution(self, title):
        return self._distrib[title]

    @abstractmethod
    def deformation(self, item):
        pass

    @property
    def meta(self) -> dict:
        return self.dataset.meta

    @abstractstaticmethod
    def deformItem(self, X):
        pass


class ElasticAugmentSet(AugmentSet):
    """
    Elastic deformation w/o random affine.
    """

    def __init__(self, *args, sigma=4, alpha: tuple = 34, **kwargs):
        """
        - kernel: size of gaussian kernel. int or tuple/list
        - sigma: sigma of gaussian filter.
        - alpha: mean and std of coefficient of elastic deformation.
        """
        AugmentSet.__init__(self, *args, **kwargs, suffix="els")
        if isinstance(alpha, (int, float)):
            std = max(1, abs(34 - alpha))
            alpha = (alpha, std)
        self.alpha = alpha
        self.filter, self.padding = getGaussianFilter(sigma)
        self.filter = self.filter.to(self.device)

    @staticmethod
    @just_image
    def deformItem(X, filter, padding, alpha_gaussian_param: tuple):
        assert len(alpha_gaussian_param) == 2
        mean, std = alpha_gaussian_param
        a = torch.randn(()) * std + mean
        return elastic(X, filter, padding, a.item())

    def deformation(self, item):
        return self.deformItem(item, self.filter, self.padding, self.alpha)


class FlipAugmentSet(AugmentSet):
    def __init__(self, *args, dim: list, **kwargs):
        super().__init__(*args, **kwargs, suffix="flip")
        if isinstance(dim, int):
            dim = [dim]
        self.dim = dim

    @staticmethod
    @just_image
    def deformItem(X, dim: list):
        return X.flip(dim=choice(dim))

    def deformation(self, item):
        return self.deformItem(item, self.dim)


class TransAugmentSet(AugmentSet):
    def __init__(self, *args, dxrange: tuple, dyrange: tuple = None, **kwargs):
        super().__init__(*args, **kwargs, suffix="trans")
        if isinstance(dxrange, (int, float)):
            dxrange = (-dxrange, dxrange)
        if dyrange is None:
            dyrange = dxrange
        elif isinstance(dyrange, (int, float)):
            dyrange = (0 - dyrange, dyrange)
        self.dxrange = dxrange
        self.dyrange = dyrange

    @staticmethod
    @just_image
    def deformItem(X, dxrange: tuple, dyrange: tuple):
        assert len(dxrange) == 2
        assert len(dyrange) == 2
        tx = torch.empty(()).uniform_(*dxrange).item()
        ty = torch.empty(()).uniform_(*dyrange).item()
        return translate(X, tx, ty)

    def deformation(self, item):
        return self.deformItem(item, self.dxrange, self.dyrange)


class ScaleAugmentSet(AugmentSet):
    def __init__(self, *args, scalerange: tuple, **kwargs):
        super().__init__(*args, **kwargs, suffix="scale")
        scalerange = tuple(scalerange)
        assert all(i > 0 for i in scalerange)
        self.scalerange = scalerange

    @staticmethod
    @just_image
    def deformItem(X, scalerange):
        assert len(scalerange) == 2
        sc = torch.empty(()).uniform_(*scalerange).item()
        return scale(X, sc)

    def deformation(self, item):
        return self.deformItem(item, self.scalerange)


def augmentWith(
    # fmt: off
    dataset: Distributed, aug_class, distrib_title: str, aim_size: int,
    device=None, tag=None, *args, **argv
    # fmt: on
):
    meta = dataset.meta
    if not all(isinstance(i, dict) for i in meta.values()):
        if tag:
            tag = str(tag)
        augset = aug_class(
            dataset, distrib_title, aim_size, *args, **argv, device=device
        )
        return DistributedConcatSet(
            [dataset, augset], tag=[tag, tag + "_aug"] if tag else None,
        )

    # TODO: How to estimate?
    estim_size = lambda D: round(len(D) / len(dataset) * aim_size)
    return DistributedConcatSet(
        (
            augmentWith(
                D, aug_class, distrib_title, estim_size(D), device, tag, *args, **argv
            )
            for tag, D in dataset.taged_datasets
        ),
        tag=[str(i) + "&aug" for i in dataset.tag],
    )
