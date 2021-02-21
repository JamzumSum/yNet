from abc import ABC, abstractclassmethod
from random import choice

import torch
import torch.nn.functional as F
from common.decorators import NoGrad
from common.utils import unsqueeze_as
from cv2 import getGaussianKernel

from .dataset import Distributed, DistributedConcatSet

first = lambda it: next(iter(it))

class VirtualDataset(Distributed, ABC):
    def argwhere(self, cond, title=None):
        raise NotImplementedError('Virual datasets cannot be query.')

class AugmentSet(VirtualDataset, ABC):
    def __init__(self, dataset: Distributed, distrib_title: str, aim_size=None):
        self.dataset = dataset
        assert not all(isinstance(i, dict) for i in dataset.meta.values()), "Augment source set must be uniform."

        self.distrib_title = distrib_title
        Distributed.__init__(self, dataset.statTitle)

        if aim_size is None:
            avg = self.dataset.distribution.max().item()
        else:
            avg = aim_size // self.K

        org_distrib = self.dataset.getDistribution(distrib_title)
        my_distrib = avg * torch.ones_like(org_distrib, dtype=torch.int) - org_distrib

        if torch.any(my_distrib < 0):
            print('At least a class has samples more than %d // %d = %d.'
            'These classes wont be sampled when augumenting.' % (aim_size, self.K, avg))
            my_distrib.clamp_(0, avg)

        self.distrib = {distrib_title: my_distrib}
        for i in dataset.statTitle:
            if i == distrib_title: continue
            j = dataset.joint(distrib_title, i)
            j = j / j.sum(dim=1, keepdim=True)
            self.distrib[i] = (my_distrib.unsqueeze(1) * j).sum(dim=0).round_()

        self._cat = torch.distributions.Categorical(my_distrib / len(self))
        self._indices = [dataset.argwhere(lambda d: d == i, distrib_title) for i in range(self.K)]

    def __getitem__(self, i):
        indices = self._indices[self._cat.sample().item()]
        return self.deformation(
            self.dataset.__getitem__(choice(indices))
        )

    def __len__(self): 
        return self.distrib[self.distrib_title].sum().item()

    @property
    def K(self): return len(self.dataset.distribution)

    def getDistribution(self, title):
        return self.distrib[title]

    @abstractclassmethod
    def deformation(self, item): pass
    
    @property
    def meta(self): return self.dataset.meta


@torch.jit.script
def elastic(X, kernel, padding, alpha=34.):
    # type: (Tensor, Tensor, int, float) -> Tensor
    '''
    X: [C, H, W]
    '''
    X = X.unsqueeze(0)
    H, W = X.shape[-2:]

    dx = torch.rand(X.shape[-2:], device=kernel.device) * 2 - 1
    dy = torch.rand(X.shape[-2:], device=kernel.device) * 2 - 1
    
    xgrid = torch.arange(W, device=dx.device).repeat(H, 1)
    ygrid = torch.arange(H, device=dy.device).repeat(W, 1).T
    with torch.no_grad():
        dx = alpha * F.conv2d(unsqueeze_as(dx, X, 0), kernel, bias=None, padding=padding)
        dy = alpha * F.conv2d(unsqueeze_as(dy, X, 0), kernel, bias=None, padding=padding)
    H /= 2; W /= 2
    dx = (dx + xgrid - W) / W
    dy = (dy + ygrid - H) / H
    grid = torch.stack((dx.squeeze(1), dy.squeeze(1)), dim=-1)
    return F.grid_sample(X, grid, padding_mode='reflection', align_corners=False).squeeze(0)


class ElasticAugmentSet(AugmentSet):
    '''
    Elastic deformation w/o random affine.
    '''
    def __init__(
        self, dataset: Distributed, distrib_title: str, aim_size=None, 
        sigma=4, alpha=34
    ):
        """
        - kernel: size of gaussian kernel. int or tuple/list
        - sigma: sigma of gaussian filter.
        - alpha: coefficient of elastic deformation.
        """
        AugmentSet.__init__(self, dataset, distrib_title, aim_size)
        self.alpha = alpha
        self.filter, self.padding = self.getFilter(sigma)
        self.gpu = torch.cuda.is_available()
        if self.gpu:
            self.filter = self.filter.cuda()

    def deformation(self, item):
        if self.gpu:
            item['X'] = item['X'].cuda()
            if 'mask' in item:
                item['mask'] = item['mask'].cuda()
        if 'mask' in item: 
            N = item['X'].size(0)
            X = torch.cat((item['X'], item['mask']), dim=0)
        else: X = item['X']

        X = elastic(X, self.filter, self.padding, self.alpha)

        if 'mask' in item:
            item['mask'] = X[N:]
            X = X[:N]
        item['X'] = X
        return item

    @staticmethod
    def getFilter(sigma):
        padding = int(4 * sigma + 0.5)
        kernel = 2 * padding + 1
        kernel = getGaussianKernel(kernel, sigma).astype('float32')
        kernel = kernel @ kernel.T
        kernel = torch.from_numpy(kernel)
        kernel = kernel.unsqueeze_(0).unsqueeze_(0)
        return kernel, padding
        

def augmentWith(dataset: Distributed, aug_class, distrib_title, aim_size, tag=None, *args, **argv):
    meta = dataset.meta
    if not all(isinstance(i, dict) for i in meta.values()):
        if tag: tag = str(tag)
        return DistributedConcatSet(
            [dataset, aug_class(dataset, distrib_title, aim_size, *args, **argv)], 
            tag=[tag, tag + '_aug'] if tag else None
        )

    return DistributedConcatSet(
        [augmentWith(D, aug_class, distrib_title, round(len(D) / len(dataset) * aim_size), tag, *args, **argv) for tag, D in dataset.taged_datasets], 
        tag=[str(i) + '&aug' for i in dataset.tag]
    )
