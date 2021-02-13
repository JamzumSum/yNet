'''
Dataset for BIRADs images & annotations. Supports multi-scale images.

* author: JamzumSum
* create: 2021-1-13
'''
import os
from abc import ABC, abstractclassmethod
from itertools import product
from collections import defaultdict

import torch
from torch.utils.data import ConcatDataset, Dataset, Subset, random_split
from utils.indexserial import IndexLoader

first = lambda it: next(iter(it))
unique = lambda it: list(set(it))

class Distributed(Dataset, ABC):
    def __init__(self, statTitle: list):
        Dataset.__init__(self)
        self.statTitle = statTitle
    
    @abstractclassmethod
    def getDistribution(self, title: str):
        raise NotImplementedError

    @abstractclassmethod
    def argwhere(self, cond, title=None, indices=None):
        raise NotImplementedError

    @property
    def distribution(self): return self.getDistribution('Ym')

    def joint(self, title1: str, title2: str):
        K1 = len(self.getDistribution(title1))
        K2 = len(self.getDistribution(title2))
        m = torch.empty((K1, K2), dtype=torch.long)
        for i in range(K1):
            arg1 = self.argwhere(lambda d: d == i, title1)
            for j in range(K2): m[i, j] = len(self.argwhere(lambda d: d == j, title2, arg1))
        return m
        
class DistributedSubset(Distributed, Subset):
    def __init__(self, dataset: Distributed, indices):
        Subset.__init__(self, dataset, indices)
        Distributed.__init__(self, dataset.statTitle)

    def getDistribution(self, title):
        K = len(self.meta['classname'][title])
        return torch.LongTensor([len(self.argwhere(lambda l: l == i, title)) for i in range(K)])

    def argwhere(self, cond, title=None, indices=None):
        if title is None: ge = lambda i: i
        else: ge = lambda i: i[title]

        if indices: gen = ((i, self.__getitem__(i), ) for i in indices)
        else: gen = enumerate(self)
        return [i for i, d in gen if cond(ge(d))]

    @property
    def meta(self): return self.dataset.meta

class DistributedConcatSet(Distributed, ConcatDataset):
    def __init__(self, datasets, tag=None):
        statTitle = unique(sum([i.statTitle for i in datasets], []))
        self.tag = tag
        ConcatDataset.__init__(self, datasets)
        Distributed.__init__(self, statTitle)

    def getDistribution(self, title):
        def safe(D):
            try: return D.getDistribution(title)
            except KeyError: return None
        stats = [safe(i) for i in self.datasets]
        stats = [i for i in stats if i is not None]
        if not stats: return None
        if len(stats) == len(self.datasets):
            return torch.stack(stats).sum(dim=0)
        else:
            stats = torch.stack(stats).sum(dim=0)
            stats = stats / stats.sum()
            return torch.round_(len(self) * stats)
            

    def argwhere(self, cond, title=None, indices=None):
        res = []; idxs = []
        cm = [0] + self.cumulative_sizes[:-1]
        for l, r in zip(cm, self.cumulative_sizes):
            idxs.append(None if indices is None else [i - l for i in indices if l <= i < r])

        for i, D in enumerate(self.datasets):
            r = D.argwhere(cond, title, idxs[i])
            for k, v in enumerate(r): r[k] = cm[i] + v
            res += r
        return res

    @property
    def taged_datasets(self): 
        if self.tag: return zip(self.tag, self.datasets)
        else: return enumerate(self.datasets)
    @property
    def meta(self): 
        if len(self.datasets) == 1: return first(self.datasets).meta
        else: 
            m = first(self.datasets).meta
            if all(m == i.meta for i in self.datasets): return m
            return {tag: D.meta for tag, D in self.taged_datasets}

class CachedDataset(Distributed):
    def __init__(self, loader, content: dict, meta):
        self.dics = content
        self.titles = meta['title']
        # If meta is a dict of single item, it must be uniform for any subset of the set.
        # Otherwise meta should have multiple items. So `distribution` must be popped up here.
        self.distrib = meta.pop('distribution')
        self.meta = meta
        self.loader = loader
        Distributed.__init__(self, meta['statistic_title'])

    def __getitem__(self, i, fetch=True):
        item = {title: self.dics[title][i] for title in self.titles}
        if fetch:
            item['X'] = self.loader.load(item['X'])
        return item

    def __len__(self): return len(first(self.dics.values()))

    def argwhere(self, cond, title=None, indices=None, fetch=None):
        if fetch is None: fetch = title in [None, 'X']
        if title is None: ge = lambda i: self.__getitem__(i, fetch)
        else: ge = lambda i: self.__getitem__(i, fetch)[title]

        gen = range(len(self)) if indices is None else indices
        return [i for i in gen if cond(ge(i))]

    def getDistribution(self, title):
        return self.distrib[title]

    @property
    def K(self): return [len(i) for i in self.meta['classname']]

class CachedDatasetGroup(DistributedConcatSet):
    '''
    Dataset for a group of cached datasets.
    '''
    def __init__(self, path):
        d: dict = torch.load(path)
        shapedic: dict = d.pop('data')
        idxs: list = d.pop('index')
        self.loader = IndexLoader(os.path.splitext(path)[0] + '.imgs', idxs)
        def copy(d, shape):
            d = d.copy()
            d['shape'] = shape
            return d
        datasets = [CachedDataset(self.loader, dic, copy(d, shape)) for shape, dic in shapedic.items()]
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
    ttag = []; vtag = []
    if isinstance(dataset, DistributedConcatSet):
        for tag, D in dataset.taged_datasets:
            trn, val = classSpecSplit(D, t, v, distrib_title, tag=tag)
            if trn: 
                tcs.append(trn)
                ttag.append("{}_t".format(tag))
            if val: 
                vcs.append(val)
                vtag.append("{}_v".format(tag))
        return DistributedConcatSet(tcs, ttag), DistributedConcatSet(vcs, vtag)
    
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
            if tag:
                ttag.append("%s_t/%s" % (tag, classname[clsidx]))
                vtag.append("%s_v/%s" % (tag, classname[clsidx]))

    return DistributedConcatSet(tcs, ttag), DistributedConcatSet(vcs, vtag)
