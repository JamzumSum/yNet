from collections import defaultdict
from itertools import chain
from random import choices
from random import shuffle as shuffle_

import torch
import torch.multiprocessing as mp
from common import deep_collate
from common.support import DeviceAwareness
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    RandomSampler,
    Sampler,
    SequentialSampler,
)

from .augment import ElasticAugmentSet

mp.set_start_method("spawn", True)
merge_list = lambda l: sum(l, [])


class CumulativeRandomSampler(RandomSampler):
    def __init__(self, add, *args, **kwargs):
        self.a = add
        RandomSampler.__init__(self, *args, **kwargs)

    def __iter__(self):
        return (self.a + i for i in super().__iter__())


class ChainSubsetRandomSampler(Sampler[int]):
    def __init__(self, dataset: ConcatDataset, setwise_shuffle=False):
        self.sampler = self.getSamplers(dataset)
        self.shuffle = setwise_shuffle

    @staticmethod
    def getSamplers(dataset, start=0):
        meta: dict = dataset.meta
        if not all(isinstance(i, dict) for i in meta.values()) or len(meta) == 1:
            return [CumulativeRandomSampler(start, dataset)]

        hashf = lambda m: (m["shape"], "Yb" in m["title"], "mask" in m["title"],)
        # dataset: ConcatDataset
        if len(set(hashf(i) for i in meta.values())) == 1:
            return [CumulativeRandomSampler(start, dataset)]
        indices = dataset.cumulative_sizes[:-1]
        indices.insert(0, 0)
        return merge_list(
            ChainSubsetRandomSampler.getSamplers(D, s + start)
            for s, D in zip(indices, dataset.datasets)
        )

    def __iter__(self):
        if self.shuffle:
            shuffle_(self.sampler)
        return chain(*tuple(iter(i) for i in self.sampler))

    def __len__(self):
        return sum(len(i) for i in self.sampler)


class FixLoader(DataLoader, DeviceAwareness):
    title = ("X", "Ym", "Yb", "mask")

    def __init__(
        self,
        dataset: ConcatDataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        device=None,
        spawn=False,
        **otherconf
    ):
        if spawn: raise NotImplementedError
        DeviceAwareness.__init__(self, device)
        sampler_cls = ChainSubsetRandomSampler if shuffle else SequentialSampler
        DataLoader.__init__(
            self,
            dataset,
            batch_size,
            **otherconf,
            pin_memory=False,
            drop_last=drop_last,
            collate_fn=self.fixCollate,
            # TODO: maybe batch_sampler in the future
            sampler=sampler_cls(dataset),
            num_workers=mp.cpu_count() if spawn else 0,
        )
        if drop_last:
            self.filter, self.padding = ElasticAugmentSet.getFilter(4)
            self.fiter = self.filter.to(self.device)

    def augmentFromBatch(self, x, N):
        aN = N - len(x)
        if aN <= 0:
            return x

        ax = [
            ElasticAugmentSet.deformItem(i, self.filter, self.padding)
            for i in choices(x, k=aN)
        ]
        return x + ax

    def fixCollate(self, x):
        """
        1. fix num of return vals
        2. make sure only one shape and annotation type in the batch.
        3. augment the batch if drop_last. For caller may expect length of any batch is fixed.
        """
        N = len(x)
        hashf = lambda i: (i["X"].shape, "Yb" in i, "mask" in i,)

        hashstat = defaultdict(list)
        for i in x:
            hashstat[hashf(i)].append(i)
        x = max(hashstat.values(), key=len)

        if self.drop_last:
            x = self.augmentFromBatch(x, N)

        x = deep_collate(x, True)
        x.setdefault("Yb", None)
        x.setdefault("mask", None)
        return x
