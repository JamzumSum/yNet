from functools import wraps

import torch

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

def Batched(func):
    @wraps(func)
    def thatinloader(loader, *args, **kwargs):
        res = [func(*d, *args, **kwargs) for d in loader]
        if not res or res[0] is None: return
        N = len(res[0])
        res = [[d[i] for d in res] for i in range(N)]
        return tuple((torch.cat if d[0].dim() > 0 else torch.stack)(d) for d in res)
    return thatinloader
