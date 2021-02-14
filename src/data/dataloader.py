from collections import defaultdict
from itertools import chain
from random import shuffle as shuffle_, choices

import torch
from torch.utils.data import (ConcatDataset, DataLoader, Sampler,
                              SequentialSampler, RandomSampler)

merge_list = lambda l: sum(l, [])

class CumulativeRandomSampler(RandomSampler):
    def __init__(self, add, *args, **kwargs):
        self.a = add
        RandomSampler.__init__(self, *args, **kwargs)
    def __iter__(self):
        return (self.a + i for i in RandomSampler.__iter__(self))

class ChainSubsetRandomSampler(Sampler[int]):

    def __init__(self, dataset: ConcatDataset):
        self.sampler = self.getSamplers(dataset)

    @staticmethod
    def getSamplers(dataset, start=0):
        meta: dict = dataset.meta
        if not all(isinstance(i, dict) for i in meta.values()) or len(meta) == 1:
            return [CumulativeRandomSampler(start, dataset)]

        hashf = lambda m: (m['shape'], 'Yb' in m['title'], 'mask' in m['title'], )
        # dataset: ConcatDataset
        if len(set(hashf(i) for i in meta.values())) == 1:
            return [CumulativeRandomSampler(start, dataset)]
        indices = dataset.cumulative_sizes[:-1]
        indices.insert(0, 0)
        return merge_list([
            ChainSubsetRandomSampler.getSamplers(D, s + start) \
            for s, D in zip(indices, dataset.datasets)
        ])

    def __iter__(self):
        shuffle_(self.sampler)
        return chain(*tuple(iter(i) for i in self.sampler))

    def __len__(self):
        return sum(len(i) for i in self.sampler)

class FixLoader(DataLoader):
    title = ('X', 'Ym', 'Yb', 'mask')

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
        3. make sure only one shape and annotation type in the batch.
        4. augment the batch if drop_last. For caller may expect length of any batch is fixed.
        '''
        N = len(x)
        hashf = lambda i: (i['X'].shape, 'Yb' in i, 'mask' in i, )

        hashstat = defaultdict(list)
        for i in x: hashstat[hashf(i)].append(i)
        x = max(hashstat.values(), key=len)

        if self.drop_last:
            x = self.augmentFromBatch(x, N)

        x = {k: torch.stack([i[k] for i in x]) for k in x[0]}
        if self.device: 
            for k, v in x.items(): x[k] = v.to(self.device)
        return x
