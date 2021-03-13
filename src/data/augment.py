import bisect
from abc import ABC, abstractclassmethod, abstractproperty
from random import choice

import torch
import torch.nn.functional as F
from common import unsqueeze_as
from common.support import DeviceAwareness
from cv2 import getGaussianKernel

from .dataset import Distributed, DistributedConcatSet

first = lambda it: next(iter(it))


class VirtualDataset(Distributed, ABC):
    def argwhere(self, *args, **kwargs):
        raise NotImplementedError("Virual datasets cannot be query.")


class AugmentSet(VirtualDataset, DeviceAwareness, ABC):
    # BUG: Use cuda to augment data is necessary to some extent but conflicts with multiprocessing
    # in most times. I've not found a method to adopt the both.
    # see warning in https://pytorch.org/docs/stable/data.html#multi-process-data-loading
    def __init__(
        self, dataset: Distributed, distrib_title: str, aim_size=None, device=None
    ):
        self.dataset = dataset
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
        x["meta"].aug = self.pid_suffix
        return x

    def __len__(self):
        return self._distrib[self.distrib_title].sum().item()

    def K(self, title):
        return self.dataset.K(title)

    def getDistribution(self, title):
        return self._distrib[title]

    @abstractclassmethod
    def deformation(self, item):
        pass

    @property
    def meta(self) -> dict:
        return self.dataset.meta

    @abstractproperty
    def pid_suffix(self) -> str:
        return "aug"


# @torch.jit.script
def elastic(X, kernel, padding, alpha=34.0):
    # type: (Tensor, Tensor, int, float) -> Tensor
    """
    X: [C, H, W]
    """
    X = X.unsqueeze(0)
    H, W = X.shape[-2:]

    dx = torch.rand(X.shape[-2:], device=kernel.device) * 2 - 1
    dy = torch.rand(X.shape[-2:], device=kernel.device) * 2 - 1

    xgrid = torch.arange(W, device=dx.device).repeat(H, 1)
    ygrid = torch.arange(H, device=dy.device).repeat(W, 1).T
    with torch.no_grad():
        dx = alpha * F.conv2d(
            unsqueeze_as(dx, X, 0), kernel, bias=None, padding=padding
        )
        dy = alpha * F.conv2d(
            unsqueeze_as(dy, X, 0), kernel, bias=None, padding=padding
        )
    H /= 2
    W /= 2
    dx = (dx + xgrid - W) / W
    dy = (dy + ygrid - H) / H
    grid = torch.stack((dx.squeeze(1), dy.squeeze(1)), dim=-1)
    return F.grid_sample(
        X, grid, padding_mode="reflection", align_corners=False
    ).squeeze(0)


class ElasticAugmentSet(AugmentSet):
    """
    Elastic deformation w/o random affine.
    """

    def __init__(
        self,
        dataset: Distributed,
        distrib_title: str,
        aim_size=None,
        device=None,
        sigma=4,
        alpha=34,
    ):
        """
        - kernel: size of gaussian kernel. int or tuple/list
        - sigma: sigma of gaussian filter.
        - alpha: coefficient of elastic deformation.
        """
        AugmentSet.__init__(self, dataset, distrib_title, aim_size, device)
        self.alpha = alpha
        self.filter, self.padding = self.getFilter(sigma)
        self.filter = self.filter.to(self.device)

    @staticmethod
    def deformItem(item, filter, padding, alpha=34):
        if "mask" in item:
            N = item["X"].size(0)
            X = torch.cat((item["X"], item["mask"]), dim=0)
        else:
            X = item["X"]

        X = elastic(X, filter, padding, alpha)

        if "mask" in item:
            item["mask"] = X[N:]
            X = X[:N]
        item["X"] = X
        return item

    def deformation(self, item):
        return self.deformItem(item, self.filter, self.padding, self.alpha)

    @property
    def pid_suffix(self):
        return "els"

    @staticmethod
    def getFilter(sigma):
        padding = int(4 * sigma + 0.5)
        kernel = 2 * padding + 1
        kernel = getGaussianKernel(kernel, sigma).astype("float32")
        kernel = kernel @ kernel.T
        kernel = torch.from_numpy(kernel)
        kernel = kernel.unsqueeze_(0).unsqueeze_(0)
        return kernel, padding


def augmentWith(
    dataset: Distributed,
    aug_class,
    distrib_title,
    aim_size,
    device=None,
    tag=None,
    *args,
    **argv
):
    meta = dataset.meta
    if not all(isinstance(i, dict) for i in meta.values()):
        if tag:
            tag = str(tag)
        return DistributedConcatSet(
            [
                dataset,
                aug_class(dataset, distrib_title, aim_size, device, *args, **argv),
            ],
            tag=[tag, tag + "_aug"] if tag else None,
        )

    return DistributedConcatSet(
        [
            augmentWith(
                D,
                aug_class,
                distrib_title,
                round(len(D) / len(dataset) * aim_size),
                device,
                tag,
                *args,
                **argv
            )
            for tag, D in dataset.taged_datasets
        ],
        tag=[str(i) + "&aug" for i in dataset.tag],
    )
