import torch
from collections import defaultdict


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

def deep_collate(out_ls: list, force_stack=False):
    if not out_ls or out_ls[0] is None:
        return

    resdic = defaultdict(list)
    for r in out_ls:
        if isinstance(r, dict):
            gen = r.items()
        elif isinstance(r, (list, tuple)):
            gen = enumerate(r)
        for j, t in gen:
            resdic[j].append(t)
    for k, v in resdic.items():
        f = torch.stack if force_stack or v[0].dim == 0 else torch.cat
        resdic[k] = f(v, dim=0)

    if isinstance(r, dict):
        return dict(resdic)
    elif isinstance(r, (list, tuple)):
        return tuple(resdic[i] for i in range(len(resdic)))
