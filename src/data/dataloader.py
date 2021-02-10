from collections import defaultdict
from itertools import chain
from random import shuffle as shuffle_, choices

import torch
from torch.utils.data import (ConcatDataset, DataLoader, Sampler,
                              SequentialSampler, RandomSampler)


class ChainSubsetRandomSampler(Sampler[int]):
    def __init__(self, dataset: ConcatDataset):
        self.sampler = [RandomSampler(i) for i in dataset.datasets]

    def __iter__(self):
        shuffle_(self.sampler)
        return chain(*tuple(iter(i) for i in self.sampler))

    def __len__(self):
        return sum(len(i) for i in self.sampler)

class FixLoader(DataLoader):
    title = ('X', 'Ym', 'Yb'
    )
    def __init__(self, dataset: ConcatDataset, batch_size=1, shuffle=False, device=None, **otherconf):
        DataLoader.__init__(
            self, dataset, batch_size, **otherconf,
            collate_fn=self.fixCollate, 
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

    def fixCollate(self, x):
        '''
        1. fix num of return vals
        2. device transfering
        3. make sure only one shape in the batch.
        4. augment the batch if drop_last. For caller may expect any length of batch is fixed.
        '''
        N = len(x)
        hashf = lambda i: hash((i['X'].shape, 'Yb' in i, ))

        hashstat = defaultdict(int)
        for i in x: hashstat[hashf(i)] += 1
        mode = max(hashstat.items(), key=lambda i: i[1])[0]
        x = [i for i in x if hashf(i) == mode]

        if self.drop_last:
            x = self.augmentFromBatch(x, N)

        x = {k: torch.stack([i[k] for i in x]) for k in x[0]}
        if self.device: 
            for k, v in x.items(): x[k] = v.to(self.device)
        return x
