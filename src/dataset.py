'''
Dataset for BIRADs images & annotations. Supports multi-scale images.

* author: JamzumSum
* create: 2021-1-13
'''
from collections import defaultdict
from itertools import chain

import torch
from torch.utils.data import (ConcatDataset, DataLoader, Subset, TensorDataset,
                              random_split)


def count(T, K):
    d = torch.zeros(K)
    c = torch.bincount(T)
    d[:len(c)] = c
    return d

class DistributedDataset(TensorDataset):
    def __init__(self, *tensors, group=None, countOn=0):
        TensorDataset.__init__(self, *tensors)
        self.countOn = countOn
        self.setgroup(group)
    
    def getDistribution(self, countOn=0):
        return count(self.tensors[countOn if countOn < 0 else countOn + 1], self.K[countOn])

    def __getattribute__(self, name):
        try: return TensorDataset.__getattribute__(self, name)
        except AttributeError: return getattr(self._group, name)

    def setgroup(self, group):
        self._group = group     

    @property
    def distribution(self): return self.getDistribution()

class DistributedDatasetList(ConcatDataset):
    def __init__(self, datasets: list, meta=None):
        assert all(isinstance(i, DistributedDataset) for i in datasets)
        self.countOn = next(iter(datasets)).countOn
        assert all(self.countOn == i.countOn for i in datasets)
        for i in datasets: i.setgroup(self)
        ConcatDataset.__init__(self, datasets)
        self.meta = meta

    def getDistribution(self, dataset=None, countOn=0):
        if dataset is None:
            return torch.stack([i.getDistribution(countOn) for i in self.datasets]).sum(dim=0)
        return self.datasets[dataset].getDistribution(countOn)

    def __getattribute__(self, name):
        try: return ConcatDataset.__getattribute__(self, name)
        except AttributeError: return self.meta[name]

    @property
    def distribution(self): return self.getDistribution()

class CachedDatasetGroup(DistributedDatasetList):
    '''
    Dataset for a group of cached datasets.
    '''
    def __init__(self, path):
        d = torch.load(path)
        shapedic: dict = d.pop('data')
        datasets = [DistributedDataset(*i) for i in shapedic.values()]
        DistributedDatasetList.__init__(self, datasets, d)

    @property
    def K(self): return [len(i) for i in self.meta['classname']]

def classSpecSplit(dataset: DistributedDatasetList, t, v):
    '''
    Split tensors in the condition that:
        for each class that is given, the size of items are kept as the given number(cls_vnum)
    NOTE: for each class, items for validation must be less than those for training.

    return: (trainset, validation set)
    '''
    if t < v:
        print('勇气可嘉.')
        return classSpecSplit(dataset, v, t)[::-1]
    vr = v / (v + t)
    countOn = dataset.countOn

    vcs = defaultdict(list)
    tcs = defaultdict(list)
    for i, D in enumerate(dataset.datasets):
        distrib = D.distribution
        s = 0
        for j, num in enumerate(distrib):
            num = int(num)
            if num == 0: 
                print('Class "%s" in dataset %d is missing. Skipped.' % (D.classname[countOn][j], i))
            elif num == 1:
                print('Class "%s" in dataset %d has only 1 sample. Add to trainset.' % (D.classname[countOn][j], i))
                tcs[i].append(Subset(D, [s]))
            else:
                vnum = max(1, round(vr * num))
                trn, val = random_split(
                    Subset(D, list(range(s, s + num))),
                    [num - vnum, vnum]
                )
                vcs[i].append(val)
                tcs[i].append(trn)
            s += num


    N = len(D.tensors)
    meta = dataset.meta
    meta['K'] = dataset.K
    for dic in (tcs, vcs):
        for k, v in dic.items():
            transpose = [[t[i] for t in chain(*v)] for i in range(N)]
            tensors = [torch.stack(i) for i in transpose]
            dic[k] = DistributedDataset(*tensors, countOn=countOn)
    return DistributedDatasetList(tcs.values(), meta), DistributedDatasetList(vcs.values(), meta)
