from abc import ABC, abstractclassmethod

import torch
import torch.nn.functional as F

from .dataset import Distributed
from cv2 import getGaussianKernel
from common.decorators import NoGrad
from common.utils import unsqueeze_as

first = lambda it: next(iter(it))

class VirtualDataset(Distributed, ABC):
    def argwhere(self, cond, title=None):
        raise NotImplementedError('Virual datasets cannot be query.')

class AugmentSet(VirtualDataset, ABC):
    def __init__(self, dataset: Distributed, distrib_title: str, aim_size=None):
        self.dataset = dataset
        self.distrib_title = distrib_title
        Distributed.__init__(self, [distrib_title])

        if aim_size is None:
            avg = self.dataset.distribution().max().item()
        else:
            avg = aim_size // self.K
        org_distrib = self.dataset.getDistribution(distrib_title)
        my_distrib = avg * torch.ones_like(org_distrib, dtype=torch.int) - org_distrib
        if torch.any(my_distrib < 0):
            print('At least a class has samples more than %d. \
            These classes wont be sampled when augumenting.' % avg)
            my_distrib.clamp_(0, avg)

        self.distrib = {distrib_title: my_distrib}
        self._cat = torch.distributions.Categorical(my_distrib / len(self))

    def __len__(self): 
        return self.distrib[self.distrib_title].sum().item()

    @property
    def K(self): return len(self.dataset.distribution)

    def getDistribution(self, title):
        if title == self.distrib_title: return self.distrib[title]
        else: return self.dataset.getDistribution(title)

    @abstractclassmethod
    def deformation(self, item): pass

    def __getitem__(self, i):
        return self.deformation(
            self.dataset.__getitem__(self._cat.sample())
        )

class ElasticAugmentSet(AugmentSet):
    '''
    Elastic deformation w/o random affine.
    '''
    def __init__(
        self, dataset: Distributed, distrib_title: str, aim_size=None, 
        kernel=21, sigma=4, alpha=34
    ):
        """
        - kernel: size of gaussian kernel. int or tuple/list
        - sigma: sigma of gaussian filter.
        - alpha: coefficient of elastic deformation.
        """
        AugmentSet.__init__(self, dataset, distrib_title, aim_size)
        self.alpha = alpha
        if isinstance(kernel, int): kernel = (kernel, kernel)
        assert kernel[0] & 1, "kernel must be odd"
        assert kernel[1] & 1, "kernel must be odd"

        self.filter = torch.nn.Conv2d(1, 1, kernel, padding=(kernel[0] // 2, kernel[1] // 2), bias=False)
        self.filter.forward = NoGrad(self.filter.forward)
        kernel = getGaussianKernel(kernel[0], sigma) @ getGaussianKernel(kernel[1], sigma).T
        kernel = torch.from_numpy(kernel).type_as(self.filter.weight)
        kernel = unsqueeze_as(kernel, self.filter.weight, 0)
        with torch.no_grad():
            self.filter.weight = torch.nn.Parameter(kernel, requires_grad=False)

    def deformation(self, item):
        item['X'] = self.elastic(item['X'])
        return item

    def elastic(self, X):
        '''
        X: [C, H, W]
        '''
        X = X.unsqueeze(0)
        H, W = X.shape[-2:]
        uniform = torch.distributions.Uniform(-1, 1)
        dx = uniform.sample(X.shape[-2:])
        dy = uniform.sample(X.shape[-2:])
        
        xgrid = torch.arange(W).repeat(H, 1)
        ygrid = torch.arange(H).repeat(W, 1).T
        with torch.no_grad():
            dx = self.alpha * self.filter(unsqueeze_as(dx, X, 0))
            dy = self.alpha * self.filter(unsqueeze_as(dy, X, 0))
        H /= 2; W /= 2
        dx = (dx + xgrid - W) / W
        dy = (dy + ygrid - H) / H
        grid = torch.stack((dx.squeeze(1), dy.squeeze(1)), dim=-1)
        return F.grid_sample(X, grid, padding_mode='reflection', align_corners=False).squeeze(0)
