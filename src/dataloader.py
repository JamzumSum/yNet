'''
Dataloader for BIRADs images & annotations.

* author: JamzumSum
* create: 2021-1-13
'''
import torch
from torch.utils.data import DataLoader, TensorDataset

class Annotated(TensorDataset):
    '''
    Dataloader for data with BIRADs annotation.

    item: image[N, C, H, W], Y_malignant[N], Y_birad[N]
    '''
    def __init__(self):
        d = torch.load('./data/BIRADs/annotated.pt')
        TensorDataset.__init__(self, d['X'], d['Ym'], d['Ybirad'])
        self.cls_name = d['cls_name']

class Unannotated(TensorDataset):
    '''
    Dataloader for data without BIRADs annotation.
    
    item: image[N, C, H, W], Y_malignant[N]
    '''
    def __init__(self):
        d = torch.load('./data/BIRADs/unannotated.pt')
        TensorDataset.__init__(self, d['X'], d['Ym'])

def trainValidSplit(t, v):
    '''
    return: (annotated train set, unannotated train set), (annotated val. set, unannotated val. set)
    '''
    if t < v: 
        print('勇气可嘉.')
        return trainValidSplit(v, t)[::-1]

    anno = Annotated()
    unanno = Unannotated()
    a = round(t / (t + v) * len(anno))  
    u = round(t / (t + v) * len(unanno))  

    loader = DataLoader(Annotated(), a, shuffle=True, drop_last=False)
    for i, d in enumerate(loader):
        if i: va = TensorDataset(*d)
        else: ta = TensorDataset(*d)

    loader = DataLoader(Unannotated(), u, shuffle=True, drop_last=False)
    for i, d in enumerate(loader):
        if i: vu = TensorDataset(*d)
        else: tu = TensorDataset(*d)

    return (ta, tu), (va, vu)