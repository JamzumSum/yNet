"""
A torch implement for U-Net.

* see: U-Net: Convolutional Networks for Biomedical Image Segmentation

* author: JamzumSum
* create: 2021-1-11
"""
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.layers import BlurPool, MaxBlurPool2d, Swish
from common.support import SelfInitialed
from misc import CheckpointSupport
from misc.decorators import autoPropertyClass


@autoPropertyClass
class ChannelInference(nn.Module):
    ic: int
    oc: int

    def __init__(self, ic: int, oc: int):
        super().__init__()


class ChannelNorm(ChannelInference, nn.GroupNorm):
    def __init__(self, ic, channels=16, *args, **kwargs):
        nn.GroupNorm.__init__(self, max(1, ic // channels), ic, *args, **kwargs)
        ChannelInference.__init__(self, ic, ic)


def norm_layer(norm: str, ndim=2):
    return {
        "batchnorm": [nn.BatchNorm1d, nn.BatchNorm2d][ndim - 1],
        "groupnorm": ChannelNorm,
        "none": nn.Identity,
    }[norm]


class ParallelLayers(nn.Module):
    def __init__(self, layers: list, dim=1):
        super().__init__()
        for i, layer in enumerate(layers):
            self.add_module(f"P{i}", layer)
        self.layers = layers
        self.dim = dim

    def forward(self, X):
        r = [f(X) for f in self.layers]
        if self.dim is None: return r
        return torch.cat(r, dim=self.dim)


@autoPropertyClass
class ConvStack2(ChannelInference):
    """
    [N, ic, H, W] -> [N, oc, H, W]
    """

    res: bool

    def __init__(self, ic, oc, *, res=False, norm="batchnorm", padding_mode='same'):
        super().__init__(ic, oc)

        # nonlinear = Swish if ic < oc else nn.ReLU
        nonlinear = nn.PReLU
        bias = norm == "none"
        self.pad = {'same': 1, 'none': 0}[padding_mode]

        self.CBR = nn.Sequential(
            nn.Conv2d(ic, oc, 3, 1, self.pad, bias=bias),
            norm_layer(norm)(oc),
            nonlinear(),
        )
        self.CB = nn.Sequential(
            nn.Conv2d(oc, oc, 3, 1, self.pad, bias=bias),
            norm_layer(norm)(oc)
        )
        if res:
            self.downsample = (
                nn.Sequential(nn.Conv2d(ic, oc, 1, bias=bias),
                              norm_layer(norm)(oc)) if ic != oc else nn.Identity()
            )

    def forward(self, X):
        r = self.CBR(X)
        if self.res:
            ds = self.downsample(X)
            if self.pad == 0: ds = ds[..., 2:-2, 2:-2]
            r = ds + self.CB(r)
        else:
            r = self.CB(r)
        return torch.relu(r)


class DownConv(ChannelInference):
    """
    [N, C, H, W] -> [N, C, H//2, W//2]
    """
    def __init__(self, ic, mode="maxpool", blur=False):
        """
        Args:
            ic (int): input channel
            mode (str, optional): `maxpool`/`avgpool`. Defaults to "maxpool".
            blur (str, optional): `none`. blur kernel before pooling.
        """
        super().__init__(ic, ic)

        f = {
            ("maxpool", False): nn.MaxPool2d,
            ("avgpool", False): nn.AvgPool2d,
            ('maxpool', True): partial(MaxBlurPool2d, ic=ic),
            ('avgpool', True): partial(BlurPool, channels=ic),
        }[(mode, blur)]
        self.pool = f(kernel_size=2, stride=2)

    def forward(self, X):
        return self.pool(X)


class UpConv(ChannelInference, nn.Sequential):
    """
    [N, C, H, W] -> [N, C//2, H*2, W*2]
    """
    def __init__(self, ic, norm="batchnorm", transConv=False):
        ChannelInference.__init__(self, ic, ic // 2)
        bias = norm == "none"

        if transConv:
            layers = [nn.ConvTranspose2d(ic, self.oc, 2, 2, bias=False)]
        else:
            # NOTE: Since 2x2 conv cannot be aligned when the shape is odd,
            # 0318: conv here is mainly object to reduce channel size. Hence use a conv1x1 instead.
            layers = [
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(ic, ic // 2, 1, bias=bias),
            ]
        layers.append(norm_layer(norm)(self.oc))
        nn.Sequential.__init__(self, *layers)

    def forward(self, X):
        return nn.Sequential.forward(self, X)


@autoPropertyClass
class BareUNet(ChannelInference):
    """
    [N, ic, H, W] -> [N, fc * 2^level, H, W], [N, fc, H, W]
    """

    level: int
    fc: int
    cps: CheckpointSupport
    cat: bool
    backbone_only: bool

    def __init__(
        self,
        ic,
        level=4,
        fc=64,
        *,
        cps=None,
        residual=False,
        norm='batchnorm',
        transConv=False,
        padding_mode='none',
        antialias=True,
        backbone_only=False,
        cat=True,
    ):
        super().__init__(ic, fc * 2 ** level)
        uniarg = dict(res=residual, norm=norm, padding_mode=padding_mode)

        self.L1 = ConvStack2(ic, fc, **uniarg)
        cc = self.L1.oc

        for i in range(level):
            dsample = DownConv(cc, blur=antialias)
            cc = dsample.oc
            conv = ConvStack2(cc, cc * 2, **uniarg)
            cc = conv.oc
            self.add_module(f"D{i + 1}", dsample)
            self.add_module(f"L{i + 2}", conv)

        if backbone_only: return

        for i in range(level):
            usample = UpConv(cc, norm=norm, transConv=transConv)
            cc = usample.oc
            conv = ConvStack2(cc * 2 if self.cat else cc, cc, **uniarg)
            cc = conv.oc
            self.add_module(f"U{i + 1}", usample)
            self.add_module(f"L{i + self.level + 2}", conv)

    def add_module(self, name, model):
        return nn.Module.add_module(self, name, self.cps(model))

    def catoradd(self, X, Y):
        """Crop X. Then cat X & Y or add them.

        Args:
            X (Tensor): [N, C, H, W]
            Y (Tensor): [N, C, H, W]

        Returns:
            Tensor: [N, 2C, H, W] if cat, else [N, C, H, W]
        """
        top = (X.size(-2) - Y.size(-2)) // 2
        left = (X.size(-1) - Y.size(-1)) // 2

        X = X[..., top:top + Y.size(-2), left:left + Y.size(-1)]
        return torch.cat([X, Y], dim=1) if self.cat else X + Y

    def _L(self, i) -> ConvStack2:
        return self._modules[f"L{i}"]

    def _D(self, i) -> DownConv:
        return self._modules[f"D{i}"]

    def _U(self, i) -> UpConv:
        return self._modules[f"U{i}"]

    def forward(self, X, expand=True):
        """
        X: [N, C, H, W]
        O: [N, fc, H, W], [N, oc, H, W]
        """
        xn = [self.L1(X)]
        L = self.level

        for i in range(1, L + 1):
            xn.append(
                self._L(i + 1)(self._D(i)(xn[-1])) # [N, t * fc, H//t, W//t], t = 2^i
            )

        if not expand:
            return xn[L], None

        for i in range(L):
            xn.append(
                self._L(L + i + 2)(
                    self.catoradd(
                        xn[L - i - 1],
                        self._U(i + 1)(xn[L + i]),
                    )                              # [N, t*fc, H//t, W//t], t = 2^(level - i - 1)
                )
            )

        return xn[L], xn[-1]


@autoPropertyClass
class UNet(BareUNet):
    """Add multiple parallel header along with original segment header.

    illustrate:
        finalx ---conv-> seg1 (original header)
                --tanh--conv--> seg2 (additional header 1)
                --tanh--conv--> seg3 (additional header 2)
                ...
    return:
        [bottomx, *header_outputs]
        e.g.    bottomx, seg
                bottomx, seg1, add_seg1, add_seg2, ...
    """
    oc: int

    def __init__(self, ic, oc, level=4, fc=64, *, headeroc=None, **kwargs):
        super().__init__(ic, level, fc, **kwargs)
        headers = [nn.Sequential(nn.Conv2d(fc, oc, 1), nn.Sigmoid())]
        if not self.backbone_only and headeroc:
            headers.extend(
                nn.Sequential(nn.Tanh(), nn.Conv2d(fc, oc, 1), nn.Sigmoid())
                for oc in headeroc
            )
        if not self.backbone_only:
            self.headers = ParallelLayers(headers, None)

    @staticmethod
    def padback(X, shape):
        top = shape[-2] - X.size(-2)
        left = shape[-1] - X.size(-1)

        return F.pad(X, [left // 2, left - left // 2, top // 2, top - top // 2])

    def forward(self, X, expand: bool = True) -> dict:
        assert not (expand and self.backbone_only)
        
        bottomx, finalx = super().forward(X, expand)
        d = {"bottom": bottomx}
        if not expand:
            return d

        d['seg'] = [self.padback(i, X.shape) for i in self.headers(finalx)]
        return d
