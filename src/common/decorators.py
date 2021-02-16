from functools import wraps

import torch
from collections import defaultdict
from torch.utils.checkpoint import checkpoint


def checkpointed(func, **kwargs):
    @wraps(func)
    def checkpointWrapper(*args):
        return checkpoint(func, *args, **kwargs)
    return checkpointWrapper

class CheckpointSupport:
    def __init__(self, memory_trade=False):
        self.memory_trade = memory_trade
    
    def __call__(self, instance):
        if not self.memory_trade: return instance
        instance.forward = checkpointed(instance.forward)
        return instance

def NoGrad(func):
    @wraps(func)
    def nogradwrapper(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return nogradwrapper

def Batched(func, bar=None):
    @wraps(func)
    def thatinloader(loader, **kwargs):
        res = [func(**d, **kwargs) for d in loader]
        if not res or res[0] is None: return

        resdic = defaultdict(list)
        for r in res:
            if isinstance(r, dict): gen = r.items()
            elif isinstance(r, (list, tuple)): gen = enumerate(r)
            for j, t in gen:
                resdic[j].append(t)
        for k, v in resdic.items():
            f = torch.cat if v[0].dim() > 0 else torch.stack
            resdic[k] = f(v, dim=0)

        if isinstance(r, dict): return dict(resdic)
        elif isinstance(r, (list, tuple)): 
            return tuple(resdic[i] for i in range(len(resdic)))
            
    return thatinloader
