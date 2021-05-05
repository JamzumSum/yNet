import torch
from collections import defaultdict

merge = lambda l: sum(l, [])


@torch.jit.script
def yes(p: float):
    return torch.rand(()) < p


@torch.jit.script
def freeze(tensor, f=0.0):
    # type: (Tensor, float) -> Tensor
    return (1 - f) * tensor + f * tensor.detach()


@torch.jit.script
def unsqueeze_as(s, t, dim=-1):
    # type: (Tensor, Tensor, int) -> Tensor
    while s.dim() < t.dim():
        s = s.unsqueeze(dim)
    return s


def deep_collate(out_ls: list, force_stack=False, filterout=None):
    if not out_ls or out_ls[0] is None:
        return

    resdic = defaultdict(list)
    if filterout is None:
        filterout = []

    for r in out_ls:
        if isinstance(r, dict):
            gen = r.items()
        elif isinstance(r, (list, tuple)):
            gen = enumerate(r)
        for j, t in gen:
            if j in filterout:
                continue
            resdic[j].append(t)
    for k, v in resdic.items():
        f = (any(not torch.is_tensor(i) for i in v)
             and merge) or ((force_stack or v[0].dim == 0) and torch.stack) or torch.cat
        resdic[k] = f(v)

    if isinstance(r, dict):
        return dict(resdic)
    elif isinstance(r, (list, tuple)):
        return tuple(resdic[i] for i in range(len(resdic)))


def spatial_softmax(x, dtype=None):
    assert x.dim() == 4
    N, C, H, W = x.shape
    return torch.softmax(x.view(N, C, -1), -1, dtype).view(N, C, H, W)


@torch.jit.script
def swish(x):
    return x * x.sigmoid()
