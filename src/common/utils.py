import torch
from itertools import product

def freeze(tensor, f=0.):
    return (1 - f) * tensor + f * tensor.detach()
    
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
    return (x > thresh * 255) * torch.stack([R, G, B], dim=-3) / 255

class ConfusionMatrix:
    def __init__(self, K=None):
        '''P&Y: [N]'''
        if K: self.m = torch.zeros(K, K, dtype=torch.int)
        self.K = K
        
    @property
    def initiated(self): return hasattr(self, 'm')
    @property
    def N(self): return self.m.sum() if self.initiated else None

    def add(self, P, Y):
        if self.K: K = self.K
        else: self.K = K = int(Y.max())

        if not self.initiated: 
            self.m = torch.zeros(K, K, dtype=torch.int)

        for i, j in product(range(K), range(K)):
            self.m[i, j] += ((P == i) * (Y == j)).sum()
    
    def accuracy(self, reduction='mean'):
        acc = self.m.diag()
        if reduction == 'mean': return acc.mean()
        else: return acc

    def err(self, *args, **kwargs): 
        return 1 - self.accuracy(*args, **kwargs)
    
    def precision(self, reduction='none'):
        acc = self.m.diag()
        prc = acc / self.m.sum(dim=0)
        if reduction == 'mean': return prc.mean()
        else: return prc

    def recall(self, reduction='none'):
        acc = self.m.diag()
        rec = acc / self.m.sum(dim=1)
        if reduction == 'mean': return rec.mean()
        else: return rec

    def fscore(self, beta=1, reduction='mean'):
        P = self.precision()
        R = self.recall()
        f = (1 + beta * beta) * (P * R) / (beta * beta * P + R)
        if reduction == 'mean': return f.mean()
        else: return f