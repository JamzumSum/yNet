from functools import wraps

import torch
from collections import defaultdict
from torch.utils.checkpoint import checkpoint
from .utils import deep_merge

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
        res = [func(d, **kwargs) for d in loader]
        return deep_merge(res)
            
    return thatinloader
