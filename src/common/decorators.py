from collections import defaultdict
from functools import wraps

import torch
from torch.utils.checkpoint import checkpoint

from . import deep_collate


def checkpointed(func, **kwargs):
    @wraps(func)
    def checkpointWrapper(*args):
        return checkpoint(func, *args, **kwargs)

    return checkpointWrapper


class CheckpointSupport:
    def __init__(self, memory_trade=False):
        self.memory_trade = memory_trade

    def __call__(self, instance):
        if not self.memory_trade:
            return instance
        instance.forward = checkpointed(instance.forward)
        return instance


def NoGrad(func):
    @wraps(func)
    def nogradwrapper(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)

    return nogradwrapper


def Batched(func):
    @wraps(func)
    def thatinloader(loader, **kwargs):
        res = [func(d, **kwargs) for d in loader]
        return deep_collate(res)

    return thatinloader



def d3support(func, asis=False):
    """
    make func(X, *args, **kwargs) that X must be [N, C, H, W] supports [C, H, W]
    by adding an extra dim before calling func.
    
    If asis=False, the output dim will be reduced automatically if inputs' dim=3.
    """

    @wraps(func)
    def d3wrapper(X, *args, **kwargs):
        d3 = X.dim() == 3
        if d3:
            X = X.unsqueeze(0)
        else:
            assert X.dim() == 4

        X = func(X, *args, **kwargs)

        if not asis and d3 and torch.is_tensor(X) and X.dim() == 4:
            return X.squeeze(0)
        else:
            return X

    return d3wrapper

