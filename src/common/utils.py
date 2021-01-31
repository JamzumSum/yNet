import torch
from itertools import product

def freeze(tensor, f=0.):
    return (1 - f) * tensor + f * tensor.detach()
    
def unsqueeze_as(s, t, dim=-1):
    while s.dim() < t.dim():
        s = s.unsqueeze(dim)
    return s

def cond_trans(T, F, cond):
    '''
    Conditional transfer like: 
        [t if c else f for t, f, c in zip(T, F, cond)]
    - T: [N, ...]
    - F: [N, ...], same as T
    - cond: [N] or [N, 1]. 0/1
    return [N, ...], same as T
    '''
    assert T.shape == F.shape
    cond = unsqueeze_as(cond, T, 1) # [N, ...]
    return T * cond + F * (1 - cond)

def class_trans(T, F, cond):
    '''
    T: [N, K, ...]
    F: [N, K, ...]
    cond: [N, K]
    return [N, ...]
    '''
    cond = unsqueeze_as(cond, T, 2) # [N, K, ...]
    return (T * cond + F * (1 - cond)).sum(dim=1)

def gray2JET(x, thresh=.5):
    """
    - x: [..., H, W],       NOTE: float 0~1
    - O: [..., 3, H, W],    NOTE: BGR, float 0~1
    """
    x = 255 * x
    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)
    B = [128 + 4 * x, 255 * ones, 255 * ones, 254 * ones, 638 - 4 * x, ones, zeros, zeros]
    G = [zeros, zeros, 4 * x - 128, 255 * ones, 255 * ones, 255 * ones, 892 - 4 * x, zeros]
    R = [zeros, zeros, zeros, 2 * ones, 4 * x - 382, 254 * ones, 255 * ones, 1148 - 4 * x]
    cond = [
        x < 31, x == 32, (33 <= x) * (x <= 95), x == 96, 
        (97 <= x) * (x <= 158), x == 159, (160 <= x) * (x <= 223), 224 <= x
    ]
    cond = torch.stack(cond)    # [8, :]
    B = torch.sum(torch.stack(B) * cond, dim=0)
    G = torch.sum(torch.stack(G) * cond, dim=0)
    R = torch.sum(torch.stack(R) * cond, dim=0)
    O = torch.stack([R, G, B], dim=-3) / 255
    return unsqueeze_as(x > thresh * 255, O, 1) * O

class ConfusionMatrix:
    def __init__(self, K=None, smooth=1e-8):
        if K: self.m = torch.zeros(K, K, dtype=torch.int)
        self.K = K
        self.eps = smooth
        
    @property
    def initiated(self): return hasattr(self, 'm')
    @property
    def N(self): return self.m.sum() if self.initiated else None

    def add(self, P, Y):
        '''P&Y: [N]'''
        if self.K: K = self.K
        else: self.K = K = int(Y.max())

        if not self.initiated: 
            self.m = torch.zeros(K, K, dtype=torch.int, device=P.device)
        if self.m.device != P.device: 
            self.m = self.m.to(P.device)
        if Y.device != P.device:
            Y = Y.to(P.device)

        for i, j in product(range(K), range(K)):
            self.m[i, j] += ((P == i) * (Y == j)).sum()
    
    def accuracy(self):
        acc = self.m.diag().sum()
        return acc / self.m.sum()

    def err(self, *args, **kwargs): 
        return 1 - self.accuracy(*args, **kwargs)
    
    def precision(self, reduction='none'):
        acc = self.m.diag()
        prc = acc / (self.m.sum(dim=1) + self.eps)
        if reduction == 'mean': return prc.mean()
        else: return prc

    def recall(self, reduction='none'):
        acc = self.m.diag()
        rec = acc / (self.m.sum() - self.m.sum(dim=0) - self.m.sum(dim=1) + self.m.diag() + self.eps)
        if reduction == 'mean': return rec.mean()
        else: return rec

    def fscore(self, beta=1, reduction='mean'):
        P = self.precision()
        R = self.recall()
        f = (1 + beta * beta) * (P * R) / (beta * beta * P + R + self.eps)
        if reduction == 'mean': return f.mean()
        else: return f