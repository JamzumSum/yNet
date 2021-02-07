from functools import wraps

import torch
from collections import defaultdict

class KeyboardInterruptWrapper:
    def __init__(self, solution):
        self._s = solution

    def __call__(self, func):
        @wraps(func)
        def KBISafeWrapper(*args, **kwargs):
            try: return func(*args, **kwargs)
            except KeyboardInterrupt:
                self._s(*args, **kwargs)
        return KBISafeWrapper

def NoGrad(func):
    @wraps(func)
    def nogradwrapper(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return nogradwrapper

def Batched(func, bar=None):
    @wraps(func)
    def thatinloader(loader, *args, **kwargs):
        
        res = [func(*d, *args, **kwargs) for d in loader]
        if not res or res[0] is None: return

        resdic = defaultdict(list)
        for r in res:
            for j, t in enumerate(r):
                resdic[j].append(t)
        for k, v in resdic.items():
            f = torch.cat if v[0].dim() > 0 else torch.stack
            resdic[k] = f(v, dim=0)
        return tuple(resdic[i] for i in range(len(resdic)))
    return thatinloader
