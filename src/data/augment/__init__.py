import torch
import torch.nn.functional as F
from common import unsqueeze_as
from common.decorators import d3support
from cv2 import getGaussianKernel


def getGaussianFilter(sigma):
    """type: int -> Tensor, int"""
    padding = int(4 * sigma + 0.5)
    kernel = 2 * padding + 1
    kernel = getGaussianKernel(kernel, sigma).astype("float32")
    kernel = kernel @ kernel.T
    kernel = torch.from_numpy(kernel)
    kernel = kernel.unsqueeze_(0).unsqueeze_(0)
    return kernel, padding


@d3support
@torch.jit.script
def elastic(X, kernel, padding, alpha=34.0):
    # type: (Tensor, Tensor, int, float) -> Tensor
    """
    X: [(N,) C, H, W]
    """
    H, W = X.shape[-2:]

    dx = torch.rand(X.shape[-2:], device=kernel.device) * 2 - 1
    dy = torch.rand(X.shape[-2:], device=kernel.device) * 2 - 1

    xgrid = torch.arange(W, device=dx.device).repeat(H, 1)
    ygrid = torch.arange(H, device=dy.device).repeat(W, 1).T
    with torch.no_grad():
        dx = alpha * F.conv2d(
            unsqueeze_as(dx, X, 0), kernel, bias=None, padding=padding
        )
        dy = alpha * F.conv2d(
            unsqueeze_as(dy, X, 0), kernel, bias=None, padding=padding
        )
    H /= 2
    W /= 2
    dx = (dx + xgrid - W) / W
    dy = (dy + ygrid - H) / H
    grid = torch.stack((dx.squeeze(1), dy.squeeze(1)), dim=-1)
    return F.grid_sample(X, grid, padding_mode="reflection", align_corners=False)


@d3support
def translate(X, dx=0.0, dy=0.0):
    # type: (Tensor, float, float) -> Tensor
    """
    X: [(N,) C, H, W]
    """
    theta = torch.Tensor([[1, 0, -dx], [0, 1, -dy]], device=X.device)
    grid = F.affine_grid(theta.unsqueeze(0), X.shape, align_corners=False)
    return F.grid_sample(X, grid)


@d3support
def scale(X, scale=1.0):
    # type: (Tensor, float) -> Tensor
    scale = 1 / scale
    theta = torch.Tensor([[scale, 0, 0], [0, scale, 0]], device=X.device)
    grid = F.affine_grid(theta.unsqueeze(0), X.shape, align_corners=False)
    return F.grid_sample(X, grid)
