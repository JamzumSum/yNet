from collections import defaultdict
from itertools import chain
from random import shuffle as shuffle_, choices

import torch
from torch.utils.data import (ConcatDataset, DataLoader, Sampler,
                              SequentialSampler, RandomSampler)
from torch.utils.data._utils.collate import default_collate


class ChainSubsetRandomSampler(Sampler[int]):
    def __init__(self, dataset: ConcatDataset):
        self.sampler = [RandomSampler(i) for i in dataset.datasets]

    def __iter__(self):
        shuffle_(self.sampler)
        return chain(*tuple(iter(i) for i in self.sampler))

    def __len__(self):
        return sum(len(i) for i in self.sampler)

class Fix3Loader(DataLoader):
    def __init__(self, dataset: ConcatDataset, batch_size=1, shuffle=False, device=None, **otherconf):
        DataLoader.__init__(
            self, dataset, batch_size, **otherconf,
            collate_fn=lambda x: Fix3Loader.fix3(self, x), 
            sampler=(ChainSubsetRandomSampler if shuffle else SequentialSampler)(dataset)
        )
        self.device = device

    def augmentFromBatch(self, x, N):
        aN = N - len(x)
        if aN <= 0: return x

        augment = lambda x: x + torch.randn_like(x) / 100
        ax = choices(x, k=aN)
        ax = [(augment(i[0]), *i[1:],) for i in ax]
        return x + ax

    def fix3(self, x):
        '''
        1. fix num of return vals to 3
        2. device transfering
        3. make sure only one shape in the batch.
        4. augment the batch if drop_last. For caller may expect any length of batch is fixed.
        '''
        N = len(x)
        shapestat = defaultdict(int)
        for i in x: shapestat[i[0].shape] += 1
        mode = max(shapestat.items(), key=lambda i: i[1])[0]
        x = [i for i in x if i[0].shape == mode]
        if self.drop_last:
            x = self.augmentFromBatch(x, N)

        x = default_collate(x)
        x = list(x) + [None] * (3 - len(x))
        if self.device: 
            x = [None if i is None else i.to(self.device) for i in x]
        return x
