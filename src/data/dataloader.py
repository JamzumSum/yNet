from collections import defaultdict
from itertools import chain
from random import choices
from random import shuffle as shuffle_

import torch
import torch.multiprocessing as mp
from common import deep_collate
from common.support import DeviceAwareness
from torch.utils.data import (BatchSampler, ConcatDataset, DataLoader,
                              RandomSampler, Sampler, SequentialSampler)

from data.dataset import Distributed

mp.set_start_method("spawn", True)
merge_list = lambda l: sum(l, [])


class CumulativeRandomSampler(RandomSampler):
    def __init__(self, add, *args, **kwargs):
        self.a = add
        super().__init__(*args, **kwargs)

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


class DistributedSampler(BatchSampler, DeviceAwareness):
    """
    Ensure amounts of each class in a batch to be the same.

    Hence `batch_size` denotes num_per_class in a batch. 
    The final batchsize is `batch_size * k`.
    Samples of the same class will be arranged continuously.
    """

    def __init__(
        self,
        dataset: Distributed,
        distrib_title: str,
        batchsize_k: int,
        device=None,
        shuffle=False,
    ):
        sampler_cls = ChainSubsetRandomSampler if shuffle else SequentialSampler
        self.batchsize_k = batchsize_k
        self.shuffle = shuffle
        self.dataset = dataset
        self.distrib_title = distrib_title
        self.k = self.dataset.K(distrib_title)
        DeviceAwareness.__init__(self, device)
        BatchSampler.__init__(self, sampler_cls(dataset), batchsize_k * self.k, True)

        bfdic = defaultdict(lambda: defaultdict(int))

        def loopmeta(x):
            nonlocal bfdic
            bfdic[x["meta"].batchflag][x[distrib_title].item()] += 1

        with dataset.no_fetch():
            dataset.argwhere(loopmeta)
        self._l = sum(min(d.values()) // batchsize_k for d in bfdic.values())

    def makebatch(self, hashdic: dict):
        batch = []
        for stack in hashdic:
            for _ in range(self.batchsize_k):
                batch.append(stack.pop())
        return batch

    def __iter__(self):
        kstack = lambda: [[] for _ in range(self.k)]
        tdic = defaultdict(kstack)
        for i in self.sampler:
            with self.dataset.no_fetch():
                x = self.dataset[i]
            hd = tdic[x["meta"].batchflag]
            hd[x[self.distrib_title]].append(i)
            if all(len(stack) > self.batchsize_k for stack in hd):
                yield self.makebatch(hd)

    def __len__(self):
        return self._l


def fixCollate(x):
    """
    1. make sure only one shape and annotation type in the batch.
    2. add a meta of the batch.
    """
    hashstat = defaultdict(list)
    for i in x:
        hashstat[i.pop("meta").batchflag].append(i)
    bf, x = max(hashstat.items(), key=lambda t: len(t[1]))

    x = deep_collate(x, True, ["meta"])
    x.setdefault("Yb", None)
    x.setdefault("mask", None)
    x["meta"] = {
        "batchflag": bf,
        'balanced': True
    }
    return x


class FixBatchLoader(DataLoader, DeviceAwareness):
    def __init__(
        self,
        dataset: ConcatDataset,
        distrib_title: str,
        batchsize_k=1,
        shuffle=False,
        device=None,
        spawn=False,
        **otherconf
    ):
        if spawn:
            raise NotImplementedError
        assert "drop_last" not in otherconf

        DeviceAwareness.__init__(self, device)
        DataLoader.__init__(
            self,
            dataset,
            **otherconf,
            pin_memory=False,
            collate_fn=fixCollate,
            num_workers=mp.cpu_count() if spawn else 0,
            batch_sampler=DistributedSampler(
                dataset, distrib_title, batchsize_k, device, shuffle
            ),
        )


class FixLoader(DataLoader, DeviceAwareness):
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
        if spawn:
            raise NotImplementedError
        DeviceAwareness.__init__(self, device)
        sampler_cls = ChainSubsetRandomSampler if shuffle else SequentialSampler
        DataLoader.__init__(
            self,
            dataset,
            batch_size,
            **otherconf,
            pin_memory=False,
            drop_last=drop_last,
            collate_fn=fixCollate,
            sampler=sampler_cls(dataset),
            num_workers=mp.cpu_count() if spawn else 0,
        )
