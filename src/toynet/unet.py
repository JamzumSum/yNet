"""
A torch implement for U-Net.

* see: U-Net: Convolutional Networks for Biomedical Image Segmentation

* author: JamzumSum
* create: 2021-1-11
"""
import torch
import torch.nn as nn
from common.decorators import autoPropertyClass
from common.support import SelfInitialed
from misc import CheckpointSupport


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


@autoPropertyClass
class ParallelLayers(nn.Module):
    dim: int

    def __init__(self, layers: list, dim=1):
        super().__init__()
        for i, layer in enumerate(layers):
            self.add_module(f"P{i}", layer)
        self.layers = layers

    def forward(self, X):
        return torch.cat([f(X) for f in self.layers], dim=self.dim)


class MultiScale(ChannelInference, nn.Sequential):
    """
    Multi-scale module by using parallel conv with various dilation rate. 
    If atrous_num=0, it degenerates to a usual conv3x3.

    NOTE: There should be a conv before this layer according to oridinal implementation.
    """

    def __init__(self, ic, oc, fc=None, atrous_num=1, bias=True):
        if atrous_num:
            if fc is None:
                fc = self._auto_fc(oc, atrous_num)
            atrous_layer = [nn.Conv2d(ic, fc, 3, 1, 1)]
            for i in range(1, atrous_num + 1):
                atrous_layer.append(nn.Conv2d(ic, fc, 3, 1, i, i))

            conv_ic = fc * (atrous_num + 1)
        else:
            conv_ic = ic

        ChannelInference.__init__(self, ic, oc)
        nn.Sequential.__init__(
            self,
            ParallelLayers(atrous_layer) if atrous_num else nn.Identity(),
            nn.Conv2d(conv_ic, oc, 3, 1, 1, bias=bias),
        )

    def forward(self, X):
        return nn.Sequential.forward(self, X)

    @staticmethod
    def _auto_fc(oc, atrous_num):
        return (
            oc // (atrous_num + 1)
            if oc % (atrous_num + 1) == 0
            else oc // 2
            if oc & 1 == 0
            else oc
        )


@autoPropertyClass
class ConvStack2(ChannelInference):
    """
    [N, ic, H, W] -> [N, oc, H, W]
    """

    res: bool

    def __init__(self, ic, oc, *, res=False, norm="batchnorm", atrous_num=0):
        super().__init__(ic, oc)

        # nonlinear = nn.ELU if ic < oc else nn.ReLU
        nonlinear = nn.ReLU
        bias = norm == "none"

        self.CBR = nn.Sequential(
            nn.Conv2d(ic, oc, 3, 1, 1, bias=bias), norm_layer(norm)(oc), nonlinear(),
        )
        self.CB = nn.Sequential(
            MultiScale(oc, oc, None, atrous_num, bias), norm_layer(norm)(oc)
        )
        if res:
            self.downsample = (
                nn.Sequential(nn.Conv2d(ic, oc, 1, bias=bias), norm_layer(norm)(oc))
                if ic != oc
                else nn.Identity()
            )

    def forward(self, X):
        r = self.CBR(X)
        if self.res:
            r = self.downsample(X) + self.CB(r)
        else:
            r = self.CB(r)
        return torch.relu(r)


class DownConv(ChannelInference):
    """
    [N, C, H, W] -> [N, C, H//2, W//2]
    """

    def __init__(self, ic, mode="maxpool"):
        """
        mode: pooling method.
            maxpool
            avgpool
            conv
            blur    # TODO
        """
        super().__init__(ic, ic)

        class AvgConv(nn.Conv2d, SelfInitialed):
            def __init__(self, kernel_size, stride):
                super().__init__(ic, ic, kernel_size, stride, bias=False)

            def selfInit(self):
                torch.nn.init.constant_(self.weight, 1 / self.kernel_size[0] ** 2)

        f = {"maxpool": nn.MaxPool2d, "avgpool": nn.AvgPool2d, "conv": AvgConv}[mode]
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
            layers = [
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                # NOTE: Since 2x2 conv cannot be aligned when the shape is odd,
                # the kernel size here is changed to 3x3. And padding is 1 to keep the same shape.
                # 0318: conv here is mainly object to reduce channel size. Hence use a conv1x1 instead.
                nn.Conv2d(ic, ic // 2, 1, bias=bias),
            ]
        layers.append(norm_layer(norm)(self.oc))
        nn.Sequential.__init__(self, *layers)

    def forward(self, X):
        return nn.Sequential.forward(self, X)


@autoPropertyClass
class UNetWOHeader(ChannelInference):
    """
    [N, ic, H, W] -> [N, fc * 2^level, H, W], [N, fc, H, W]
    """

    level: int
    fc: int
    cps: CheckpointSupport

    def __init__(
        # fmt: off
        self, ic, level=4, fc=64, *, 
        cps=None, residual=False, norm='batchnorm', 
        multiscale=False, transConv=False
        # fmt: on
    ):
        super().__init__(ic, fc * 2 ** level)
        self.add_module("L1", ConvStack2(ic, fc, res=residual))
        cc = self._L(1).oc

        for i in range(level):
            dsample = DownConv(cc, "conv")
            cc = dsample.oc
            conv = ConvStack2(
                cc,
                cc * 2,
                res=residual,
                norm=norm,
                atrous_num=max(1, level - i) if multiscale else 0,
            )
            cc = conv.oc
            self.add_module("D%d" % (i + 1), dsample)
            self.add_module("L%d" % (i + 2), conv)

        for i in range(level):
            usample = UpConv(cc, norm=norm, transConv=transConv)
            cc = usample.oc
            conv = ConvStack2(cc * 2, cc, res=residual)
            cc = conv.oc
            self.add_module("U%d" % (i + 1), usample)
            self.add_module("L%d" % (i + self.level + 2), conv)

    def add_module(self, name, model):
        return nn.Module.add_module(self, name, self.cps(model))

    @staticmethod
    def _padCat(X, Y):
        """
        Pad Y and concat with X.
        X: [N, C, H_max, W_max]
        Y: [N, C, H_min, W_min]
        -> [N, C, H_max, W_max]
        """
        # hmax, wmax = X.shape[-2:]
        # hmin, wmin = Y.shape[-2:]
        # padl = (wmax - wmin) // 2
        # padr = wmax - wmin - padl
        # padt = (hmax - hmin) // 2
        # padb = hmax - hmin - padt
        # Y = torch.nn.functional.pad(Y, (padl, padr, padt, padb), 'constant')
        return torch.cat([X, Y], dim=1)

    def _L(self, i) -> ConvStack2:
        return self._modules["L%d" % i]

    def _D(self, i) -> DownConv:
        return self._modules["D%d" % i]

    def _U(self, i) -> UpConv:
        return self._modules["U%d" % i]

    def forward(self, X, expand=True):
        """
        X: [N, C, H, W]
        O: [N, fc, H, W], [N, oc, H, W]
        """
        if self.cps.memory_trade:
            X = X.clone().requires_grad_()

        xn = [self.L1(X)]

        for i in range(1, self.level + 1):
            xn.append(
                self._L(i + 1)(self._D(i)(xn[-1]))
            )  # [N, t * fc, H//t, W//t], t = 2^i

        if not expand:
            return xn[self.level], None

        for i in range(self.level):
            xn.append(
                self._L(self.level + i + 2)(
                    # [N, t*fc, H//t, W//t], t = 2^(level - i - 1)
                    self._padCat(
                        xn[self.level - i - 1], self._U(i + 1)(xn[self.level + i])
                    )
                )
            )

        return xn[self.level], xn[-1]


@autoPropertyClass
class UNet(UNetWOHeader):
    r"""
    Add multiple parallel header along with original segment header.
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
        self.headers = [nn.Sequential(nn.Conv2d(self.fc, oc, 1), nn.Sigmoid())]
        if headeroc:
            self.headers.extend(
                nn.Sequential(nn.Tanh(), nn.Conv2d(self.fc, oc, 1), nn.Sigmoid())
                for oc in headeroc
            )
        for i, f in enumerate(self.headers):
            self.add_module("header %d" % (i + 1), f)

    def forward(self, X, expand: bool = True) -> dict:
        bottomx, finalx = UNetWOHeader.forward(self, X, expand)
        r = {"bottom": bottomx}
        if not expand:
            return r

        for i, f in enumerate(self.headers):
            r[f"seg{i}"] = f(finalx)
        return r
