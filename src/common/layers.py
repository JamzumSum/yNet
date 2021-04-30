import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil
from . import swish


class SEBlock(nn.Sequential):
    def __init__(self, L, hs=128):
        nn.Sequential.__init__(
            self, nn.Linear(L, hs), nn.ReLU(), nn.Linear(hs, L),
            nn.Softmax(dim=-1)
                                                                 # use softmax instead of sigmoid here since the attention-ed channels are sumed,
                                                                 # while the sum might be greater than 1 if sum of the attention vector is not restricted.
        )
        nn.init.constant_(self[2].bias, 1 / L)

    def forward(self, X):
        """
        X: [N, K, H, W, L]
        O: [N, K, H, W]
        """
        X = X.permute(4, 0, 1, 2, 3)                                # [L, N, K, H, W]
        Xp = F.adaptive_avg_pool2d(X, (1, 1))                       # [L, N, K, 1, 1]
        Xp = Xp.permute(1, 2, 3, 4, 0)                              # [N, K, 1, 1, L]
        Xp = nn.Sequential.forward(self, Xp).permute(4, 0, 1, 2, 3) # [L, N, K, 1, 1]
        return (X * Xp).sum(dim=0)


class PyramidPooling(nn.Module):
    """
    Use pyramid pooling instead of max-pooling to make sure more elements in CAM can be backward. 
    Otherwise only the patch with maximum average confidence has grad while patches and small.
    Moreover, the size of patches are fixed so is hard to select. Multi-scaled patches are suitable.
    """
    def __init__(self, patch_sizes, hs=128):
        nn.Module.__init__(self)
        if any(i & 1 for i in patch_sizes):
            print(
                """Warning: At least one value in `patch_sizes` is odd. 
            Channel-wise align may behave incorrectly."""
            )
        self.patch_sizes = sorted(patch_sizes)
        self.atn = SEBlock(self.L, hs)

    @property
    def L(self):
        return len(self.patch_sizes)

    def forward(self, X):
        """
        X: [N, C, H, W]
        O: [N, K, 2 * H//P_0 -1, 2 * W//P_0 - 1]
        """
        # set stride as P/2, so that patches overlaps each other
        # hopes to counterbalance the lack-representating of edge pixels of a patch.
        ls = [
            F.avg_pool2d(X, patch_size, patch_size // 2)
            for patch_size in self.patch_sizes
        ]
        base = ls.pop(0)                       # [N, K, H//P0, W//P0]
        ls = [F.interpolate(i, base.shape[-2:], mode="nearest") for i in ls]
        ls.insert(0, base)
        ls = torch.stack(ls, dim=-1)           # [N, K, H//P0, W//P0, L]
        return self.atn(ls)


class MLP(nn.Sequential):
    def __init__(self, ic, oc, hidden_layers, final_bn=True, final_relu=False):
        layers = []
        cc = ic
        for i in hidden_layers:
            layers.append(nn.Linear(cc, i))
            layers.append(nn.BatchNorm1d(i))
            layers.append(nn.ReLU())
            cc = i
        layers.append(nn.Linear(cc, oc))
        if final_bn:
            layers.append(nn.BatchNorm1d(cc))
        if final_relu:
            layers.append(nn.ReLU())
        super().__init__(*layers)


class Swish(nn.Module):
    def forward(self, x):
        return swish(x)


class BlurPool(nn.Module):
    '''
    https://github.com/adobe/antialiased-cnns/blob/d4bf038a24cb2cdeae721ccaeeb1bd0c81c8dff7/antialiased_cnns/blurpool.py#L13
    '''
    def __init__(self, channels, pad_type='reflect', filt_size=4, stride=2, pad_off=0):
        super(BlurPool, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off

        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        a = {2: [1, 1], 3: [1, 2, 1], 5: [1, 4, 6, 4, 1]}[filt_size]
        a = torch.tensor(a, dtype=torch.float)

        filt = a[:, None] @ a[None, :]
        filt = filt / torch.sum(filt)
        self.register_buffer('kernel', filt[None, None, :, :].repeat(channels, 1, 1, 1))

        pad_sizes = [
            int(1. * (filt_size - 1) / 2),
            int(ceil(1. * (filt_size - 1) / 2)),
        ]
        pad_sizes = [i + pad_off for i in pad_sizes * 2]
        self.pad = self.get_pad_layer(pad_type)(pad_sizes)

    @staticmethod
    def get_pad_layer(pad_type):
        return {
            'reflect': nn.ReflectionPad2d,
            'replicate': nn.ReplicationPad2d,
            'zeros': nn.ZeroPad2d,
        }[pad_type]

    def forward(self, X):
        return F.conv2d(self.pad(X), self.kernel, stride=self.stride, groups=X.size(1))


class MaxBlurPool2d(nn.MaxPool2d):
    def __init__(self, ic, kernel_size, stride, blur_ker=3) -> None:
        super().__init__(kernel_size, stride=1)
        self.blur = BlurPool(ic, filt_size=blur_ker, stride=stride)

    def forward(self, input):
        return self.blur(super().forward(input))
