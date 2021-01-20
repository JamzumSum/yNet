import torch

def freeze(tensor, f=0.):
    return (1 - f) * tensor + f * tensor.detach()
    
def gray2JET(x, thresh=.6):
    """
    - x: [H, W], NOTE: float 0~1
    - O: [3, H, W], NOTE: BGR, float 0~1
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
    return (x < thresh * 255) * torch.stack([R, G, B]) / 255