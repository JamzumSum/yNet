'''
Dataloader for BIRADs images & annotations.

* author: JamzumSum
* create: 2021-1-13
'''
import torch
from torch.utils.data import DataLoader, TensorDataset

def count(longT):
    return [int((longT == i).sum()) for i in range(int(longT.max()))]

class Annotated(TensorDataset):
    '''
    Dataloader for data with BIRADs annotation.

    item: image[N, C, H, W], Y_malignant[N], Y_birad[N]
    '''
    def __init__(self):
        d = torch.load('./data/BIRADs/annotated.pt')
        TensorDataset.__init__(self, d['X'], d['Ym'].long(), d['Ybirad'].long())
        self.cls_name = d['cls_name']

    @property
    def BIRADsDistribution(self): 
        if not hasattr(self, '_bd'): self._bd = tuple(count(self.tensors[2]))
        return self._bd

    @property
    def MalignantDistribution(self): 
        if not hasattr(self, '_md'): self._md = tuple(count(self.tensors[1]))
        return self._md
    
    @property
    def K(self): return len(self.cls_name)
class Unannotated(TensorDataset):
    '''
    Dataloader for data without BIRADs annotation.
    
    item: image[N, C, H, W], Y_malignant[N]
    '''
    def __init__(self):
        d = torch.load('./data/BIRADs/unannotated.pt')
        TensorDataset.__init__(self, d['X'], d['Ym'].long())

    @property
    def MalignantDistribution(self): 
        if not hasattr(self, '_md'): self._md = tuple(count(self.tensors[1]))
        return self._md

def classSpecSplit(tensors, cls_sum, cls_vnum):
    '''
    Split tensors in the condition that:
        for each class that is given, the size of items are kept as the given number(cls_vnum)
    NOTE: for each class, items for validation must be less than those for training.
    
    return: (trainset, validation set)
    '''
    K = len(cls_sum)
    assert K == len(cls_vnum)
    assert all(cls_vnum[i] * 2 <= cls_sum[i] for i in range(K))

    vcs = []; tcs = []
    s = 0
    for i in range(K):
        loader = DataLoader(
            TensorDataset(*(t[s:s + cls_sum[i]] for t in tensors)), 
            cls_sum[i] - cls_vnum[i], 
            shuffle=True, drop_last=False
        )
        s += cls_sum[i]
        for isv, d in enumerate(loader):
            if isv: vcs.append(d)
            else: tcs.append(d)

    N = len(tcs[0])
    tcs = [torch.cat([t[i] for t in tcs], dim=0) for i in range(N)]
    vcs = [torch.cat([t[i] for t in vcs], dim=0) for i in range(N)]
    return TensorDataset(*tcs), TensorDataset(*vcs)

def trainValidSplit(t, v):
    '''
    return: (annotated train set, unannotated train set), (annotated val. set, unannotated val. set)
    '''
    if t < v: 
        print('勇气可嘉.')
        return trainValidSplit(v, t)[::-1]

    anno = Annotated()
    unanno = Unannotated()
    vr = v / (t + v)
    a = round(vr * len(anno))  
    u = round(vr * len(unanno))  

    va_distrib = [max(1, round(i * vr)) for i in anno.BIRADsDistribution]
    va_distrib[0] += a - sum(va_distrib)
    ta, va = classSpecSplit(anno.tensors, anno.BIRADsDistribution, va_distrib)
    
    vu_distrib = [max(1, round(i * vr)) for i in unanno.MalignantDistribution]
    vu_distrib[0] += u - sum(vu_distrib)
    tu, vu = classSpecSplit(unanno.tensors, unanno.MalignantDistribution, vu_distrib)

    return (ta, tu), (va, vu)