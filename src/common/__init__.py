import torch


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
