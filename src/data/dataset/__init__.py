"""
Dataset for BIRADs images & annotations. Supports multi-scale images.

* author: JamzumSum
* create: 2021-1-13
"""
import bisect
import os
from abc import ABC, abstractclassmethod
from collections import defaultdict
from itertools import product

import torch
from misc.indexserial import IndexLoader
from torch import default_generator as ggen  # type: ignore
from torch.utils.data import ConcatDataset, Dataset, Subset, random_split

first = lambda it: next(iter(it))
unique = lambda it: list(set(it))


class Distributed(Dataset, ABC):
    def __init__(self, statTitle: list):
        Dataset.__init__(self)
        self.statTitle = statTitle
        self._fetch = True

    def getDistribution(self, title: str):
        if title not in self.statTitle:
            return

        z = torch.zeros((self.K(title),), dtype=torch.long)
        for d in self:
            z[d[title]] += 1
        return z

    def K(self, title):
        return max(self, lambda d: d[title])

    def argwhere(self, cond, title=None, indices=None):
        if title is None:
            ge = lambda i: i
        else:
            ge = lambda i: i[title]

        if indices is None:
            indices = range(len(self))
        gen = ((i, self[i]) for i in indices)

        return [i for i, d in gen if cond(ge(d))]

    @property
    def distribution(self):
        return self.getDistribution("Ym")

    def joint(self, title1: str, title2: str):
        K1 = len(self.getDistribution(title1))
        K2 = len(self.getDistribution(title2))
        m = torch.empty((K1, K2), dtype=torch.long)
        for i in range(K1):
            with self.no_fetch():
                arg1 = self.argwhere(lambda d: d == i, title1)
                for j in range(K2):
                    m[i, j] = len(self.argwhere(lambda d: d == j, title2, arg1))
        return m

    def no_fetch(self):
        p = self

        class NoFetchContext:
            def __enter__(self):
                p._fetch = False

            def __exit__(self, *args, **kwargs):
                p._fetch = True

        return NoFetchContext()


class DistributedSubset(Distributed, Subset):
    def __init__(self, dataset: Distributed, indices):
        Subset.__init__(self, dataset, indices)
        Distributed.__init__(self, dataset.statTitle)
        self._distrib = {}

    def getDistribution(self, title):
        if title not in self._distrib:
            # cache it
            self._distrib[title] = Distributed.getDistribution(self, title)
        return self._distrib[title]

    def K(self, title):
        return len(self.meta["classname"][title])

    def joint(self, title1: str, title2: str):
        K1 = len(self.getDistribution(title1))
        K2 = len(self.getDistribution(title2))
        m = torch.empty((K1, K2), dtype=torch.long)
        for i in range(K1):
            with self.no_fetch():
                arg1 = self.dataset.argwhere(
                    lambda d: d == i, title1, indices=self.indices
                )
                for j in range(K2):
                    m[i, j] = len(self.dataset.argwhere(lambda d: d == j, title2, arg1))
        return m

    @property
    def meta(self):
        return self.dataset.meta

    def no_fetch(self):
        p = self
        class NoFetchContext:
            def __enter__(self):
                self.ct = p.dataset.no_fetch()
                self.ct.__enter__()
            def __exit__(self, *args, **kwargs):
                self.ct.__exit__(*args, **kwargs)
        return NoFetchContext()


class DistributedConcatSet(Distributed, ConcatDataset):
    def __init__(self, datasets, tag=None):
        statTitle = unique(sum([i.statTitle for i in datasets], []))
        self.tag = tag
        ConcatDataset.__init__(self, datasets)
        Distributed.__init__(self, statTitle)

    def K(self, title):
        if title not in self.statTitle:
            return
        for i in self.datasets:
            K = i.K(title)
            if K is not None:
                return K

    def getDistribution(self, title):
        stats = (i.getDistribution(title) for i in self.datasets)
        stats = [i for i in stats if i is not None]
        if not stats:
            return
        if len(stats) == len(self.datasets):
            return torch.stack(stats).sum(dim=0)
        else:
            stats = torch.stack(stats).sum(dim=0)
            stats = stats / stats.sum()
            return torch.round_(len(self) * stats)

    def joint(self, title1, title2):
        return sum(i.joint(title1, title2) for i in self.datasets)

    def argwhere(self, cond, title=None, indices=None):
        if indices is None:
            with self.no_fetch():
                return Distributed.argwhere(self, cond, title, indices)

        res = []
        cm = [0] + self.cumulative_sizes[:-1]
        idxs = [
            [i - l for i in indices if l <= i < r]
            for l, r in zip(cm, self.cumulative_sizes)
        ]

        for i, D in enumerate(self.datasets):
            r = D.argwhere(cond, title, idxs[i])
            for k, v in enumerate(r):
                r[k] = cm[i] + v
            res += r
        return res

    @property
    def taged_datasets(self):
        if self.tag:
            return zip(self.tag, self.datasets)
        else:
            return enumerate(self.datasets)

    @property
    def meta(self):
        if len(self.datasets) == 1:
            return first(self.datasets).meta
        else:
            m = first(self.datasets).meta
            if all(m == i.meta for i in self.datasets):
                return m
            return {tag: D.meta for tag, D in self.taged_datasets}

    def no_fetch(self):
        p = self

        class NoFetchContext:
            def __init__(self):
                self.ct = []

            def __enter__(self):
                for i in p.datasets:
                    ct = i.no_fetch()
                    ct.__enter__()
                    self.ct.append(ct)

            def __exit__(self, *args, **kwargs):
                for i in self.ct:
                    i.__exit__(*args, **kwargs)

        return NoFetchContext()


def classSpecSplit(
    dataset: Distributed, t, v, distrib_title, generator=ggen, tag=None,
):
    """
    Split tensors in the condition that:
        for each class that is given, the size of items are kept as the given number(cls_vnum)
    NOTE: for each class, items for validation must be less than those for training.

    generator: generate seed for random spliting. 
    Useful on continue-training to get an uncahnaged set split.

    return: (trainset, validation set)
    """
    if t < v:
        print("勇气可嘉.")
        return classSpecSplit(dataset, v, t, distrib_title, generator, tag)[::-1]
    vr = v / (v + t)

    tcs = []
    vcs = []
    ttag = []
    vtag = []
    if isinstance(dataset, DistributedConcatSet):
        for tag, D in dataset.taged_datasets:
            trn, val = classSpecSplit(D, t, v, distrib_title, generator, tag)
            if trn:
                tcs.append(trn)
                ttag.append("{}_t".format(tag))
            if val:
                vcs.append(val)
                vtag.append("{}_v".format(tag))
        return DistributedConcatSet(tcs, ttag), DistributedConcatSet(vcs, vtag)

    distrib = dataset.distribution
    classname = dataset.meta["classname"][distrib_title]

    for clsidx, num in enumerate(distrib):
        num = int(num)
        if num == 0:
            print(
                'Class "%s" in dataset %s is missing. Skipped.'
                % (classname[clsidx], str(tag))
            )
            continue
        elif num == 1:
            print(
                'Class "%s" in dataset %s has only 1 sample. Add to trainset.'
                % (classname[clsidx], str(tag))
            )
            tcs.append(DistributedSubset(dataset, torch.arg))
            val = None
        else:
            vnum = max(1, round(vr * num))
            with dataset.no_fetch():
                indices = dataset.argwhere(lambda d: d == clsidx, distrib_title)
            extract = DistributedSubset(dataset, indices)
            # NOTE: fix for continue training
            seed = int(
                torch.empty((), dtype=torch.int64).random_(generator=generator).item()
            )
            gen = torch.Generator().manual_seed(seed)
            trn, val = random_split(extract, [num - vnum, vnum], generator=gen)
            tcs.append(DistributedSubset(extract, trn.indices))
            vcs.append(DistributedSubset(extract, val.indices))
            if tag:
                ttag.append("%s_t/%s" % (tag, classname[clsidx]))
                vtag.append("%s_v/%s" % (tag, classname[clsidx]))

    return DistributedConcatSet(tcs, ttag), DistributedConcatSet(vcs, vtag)
