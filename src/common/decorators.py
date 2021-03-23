from collections import defaultdict
from functools import wraps, partial

import torch
from torch.utils.checkpoint import checkpoint

from . import deep_collate


def checkpointed(func, **kwargs):
    @wraps(func)
    def checkpointWrapper(*args):
        return checkpoint(func, *args, **kwargs)

    return checkpointWrapper


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


class d3support:
    r"""
    make func(X, *args, **kwargs) that X must be [N, C, H, W] supports [C, H, W]
    by adding an extra dim before calling func.
    
    If asis=False, the output dim will be reduced automatically if inputs' dim=3.
    """
    def __init__(self, asis=False):
        self.asis = asis

    def __call__(self, func):
        @wraps(func)
        def d3wrapper(X, *args, **kwargs):
            d3 = X.dim() == 3
            if d3: 
                X = X.unsqueeze(0)
            else: 
                assert X.dim() == 4

            X = func(X, *args, **kwargs)

            if not self.asis and d3 and torch.is_tensor(X) and X.dim() == 4:
                return X.squeeze(0)
            else: return X

        return d3wrapper


def autoProperty(func, strict: dict = None):
    assert func.__name__ == "__init__"
    parg_name = func.__code__.co_varnames[1:]
    arg_name: list = parg_name[:func.__code__.co_argcount - 1]
    if (i := func.__defaults__) is None:
        defaults = {}
    else:
        defaults = {k: v for k, v in zip(arg_name[::-1], i)}
    if (i := func.__kwdefaults__) is not None:
        defaults.update(i)

    @wraps(func)
    def initWrapper(self, *args, **kwargs):
        annotation = None if strict is None else strict.copy()
        d = kwargs.copy()
        d.update(zip(parg_name, args))

        for k, v in d.items():
            if annotation is None or k in annotation:
                setattr(self, k, v)
                if annotation:
                    annotation.pop(k)
        if annotation:
            for i in annotation:
                setattr(self, i, defaults[i])
        func(self, *args, **kwargs)

    return initWrapper


def autoPropertyClass(cls, strict=True):
    cls.__init__ = autoProperty(cls.__init__,
                                cls.__annotations__ if strict else None)
    return cls
