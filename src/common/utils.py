import torch

def freeze(tensor, f=0.):
    return (1 - f) * tensor + f * tensor.detach()
    
def gray2JET(x, thresh=.5):
    """
    - x: [H, W], NOTE: float 0~1
    - O: [3, H, W], NOTE: BGR, float 0~1
    """
    x = 255 * x * (x < thresh)
    B = [128 + 4 * x, 255, 255, 254, 250 - 4 * x, 1, 0, 0]
    G = [0, 0, 4 + 4 * x, 255, 255, 255, 252 - 4 * x, 0]
    R = [0, 0, 0, 2, 6 + 4 * x, 254, 255, 252 - 4 * x]
    cond = [
        0 <= x < 31, x == 32, 33 <= x <= 95, x == 96, 
        97 <= x <= 158, x == 159, 160 <= x <= 223, 224 <= x <= 255
    ]
    cond = torch.stack(cond)    # [8, :]
    B = torch.sum(torch.stack(B) * cond, dim=1)
    G = torch.sum(torch.stack(G) * cond, dim=1)
    R = torch.sum(torch.stack(R) * cond, dim=1)
    return torch.stack([B, G, R]) / 255
