'''
Dataset for BIRADs images & annotations. Supports multi-scale images.

* author: JamzumSum
* create: 2021-1-13
'''
import os
from abc import ABC, abstractclassmethod
from collections import defaultdict
from itertools import chain
from random import randint

import torch
from torch.utils.data import ConcatDataset, Dataset, Subset, random_split

from utils.indexserial import IndexLoader

first = lambda it: next(iter(it))

class Distributed(Dataset, ABC):
    def __init__(self, statTitle: list):
        Dataset.__init__(self)
        self.statTitle = statTitle
    
    @abstractclassmethod
    def getDistribution(self, title: str):
        raise NotImplementedError

    @abstractclassmethod
    def argwhere(self, title: str, cond):
        raise NotImplementedError

    @property
    def distribution(self): return self.getDistribution('Ym')

class DistributedSubset(Distributed, Subset):
    def __init__(self, dataset: Distributed, indices):
        Subset.__init__(self, dataset, indices)
        Distributed.__init__(self, dataset.statTitle)

    def getDistribution(self, title):
        return self.dataset.getDistribution(title)

    def argwhere(self, cond, title=None):
        if title is None: ge = lambda i: self.__getitem__(i)
        else: ge = lambda i: self.__getitem__(i)[title]

        return [i for i in self.indices if cond(ge(i))]

    @property
    def meta(self): return self.dataset.meta

class DistributedConcatSet(Distributed, ConcatDataset):
    def __init__(self, datasets, tag=None):
        statTitle = first(datasets).statTitle
        self.tag = tag
        ConcatDataset.__init__(self, datasets)
        Distributed.__init__(self, statTitle)

    def getDistribution(self, title):
        stats = [i.getDistribution(title) for i in self.datasets]
        stats = torch.stack(stats)
        return stats.sum(dim=0)

    def argwhere(self, cond, title=None):
        res = []
        cm = [0] + self.cumulative_sizes[:-1]
        for i, D in enumerate(self.datasets):
            r = D.argwhere(cond, title)
            for k, v in enumerate(r): r[k] = cm[i] + v
            res += r
        return res

    @property
    def taged_datasets(self): return zip(self.tag, self.datasets)
    @property
    def meta(self): return first(self.datasets).meta

class CachedDataset(Distributed):
    def __init__(self, loader, content: dict, meta):
        self.dics = content
        self.titles = meta['title']
        self.meta = meta
        self.loader = loader
        Distributed.__init__(self, meta['statistic_title'])

    def __getitem__(self, i):
        item = {title: self.dics[title][i] for title in self.titles}
        item['X'] = self.loader.load(item['X'])
        return item

    def __len__(self): return len(first(self.dics.values()))

    def argwhere(self, cond, title=None):
        if title is None: ge = lambda i: self.__getitem__(i)
        else: ge = lambda i: self.__getitem__(i)[title]

        return [i for i in range(len(self)) if cond(ge(i))]

    def getDistribution(self, title):
        return self.meta['distribution'][title]

    @property
    def K(self): return [len(i) for i in self.meta['classname']]

class AugmentSet(Distributed, ABC):
    def __init__(self, dataset, aim_size=None):
        self.dataset = dataset
        self.length = aim_size - len(dataset)

    def __len__(self): return self.length

    def _sample(self): 
        return self.dataset.__getitem__(randint(0, len(self.dataset)))

    @abstractclassmethod
    def deformation(self, item): pass

    def __getitem__(self, i):
        return self.deformation(self._sample())

class ElasticAugmentSet(AugmentSet):
    def deformation(self, item):
        # item['X'] = 
        raise NotImplementedError
        return item

class CachedDatasetGroup(DistributedConcatSet):
    '''
    Dataset for a group of cached datasets.
    '''
    def __init__(self, path, augment=True):
        d = torch.load(path)
        shapedic: dict = d.pop('data')
        idxs = d.pop('index')
        self.loader = IndexLoader(os.path.splitext(path)[0] + '.imgs', idxs)
        datasets = [CachedDataset(self.loader, shapedic[shape], d) for shape in shapedic]
        DistributedConcatSet.__init__(self, datasets, tag=shapedic.keys())

def classSpecSplit(dataset: Distributed, t, v, distrib_title='Ym', tag=None):
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

    tcs = []; vcs = []
    if isinstance(dataset, DistributedConcatSet):
        for tag, D in dataset.taged_datasets:
            trn, val = classSpecSplit(D, t, v, distrib_title, tag=tag)
            if trn: tcs.append(trn)
            if val: vcs.append(val)
        return DistributedConcatSet(tcs), DistributedConcatSet(vcs)
    
    distrib = dataset.distribution
    classname = dataset.meta['classname'][distrib_title]

    for clsidx, num in enumerate(distrib):
        num = int(num)
        if num == 0: 
            print('Class "%s" in dataset %s is missing. Skipped.' % (classname[clsidx], str(tag)))
            continue
        elif num == 1:
            print('Class "%s" in dataset %s has only 1 sample. Add to trainset.' % (classname[clsidx], str(tag)))
            tcs.append(DistributedSubset(dataset, torch.arg))
            val = None
        else:
            vnum = max(1, round(vr * num))
            indices = dataset.argwhere(lambda d: d == clsidx, distrib_title)
            extract = DistributedSubset(dataset, indices)
            trn, val = random_split(extract, [num - vnum, vnum])
            tcs.append(DistributedSubset(extract, trn.indices))
            vcs.append(DistributedSubset(extract, val.indices))

    return DistributedConcatSet(tcs), DistributedConcatSet(vcs)
