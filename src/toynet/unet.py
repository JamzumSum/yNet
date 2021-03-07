"""
A torch implement for U-Net.

* see: U-Net: Convolutional Networks for Biomedical Image Segmentation

* author: JamzumSum
* create: 2021-1-11
"""
import torch
import torch.nn as nn
from common.decorators import checkpoint, CheckpointSupport
from common.support import SelfInitialed


def norm_layer(norm: str, ic):
    if norm == "batchnorm":
        return nn.BatchNorm2d(ic)
    elif norm == "groupnorm":
        return nn.GroupNorm(max(1, ic // 16), ic)
    else:
        raise ValueError(norm)


class ChannelInference(nn.Module):
    def __init__(self, ic: int, oc: int):
        super().__init__()
        self.ic = ic
        self.oc = oc


class ConvStack2(ChannelInference):
    """
    [N, ic, H, W] -> [N, oc, H, W]
    """

    def __init__(self, ic, oc, res=False, norm="batchnorm"):
        super().__init__(ic, oc)
        self.res = res
        self.CBR = nn.Sequential(
            nn.Conv2d(ic, oc, 3, 1, 1, bias=False), norm_layer(norm, oc), nn.ReLU()
        )
        self.CB = nn.Sequential(
            nn.Conv2d(oc, oc, 3, 1, 1, bias=False), norm_layer(norm, oc)
        )

    def forward(self, X):
        X = self.CBR(X)
        if self.res:
            X = X + self.CB(X)
        else:
            X = self.CB(X)
        return torch.relu(X)


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


class UpConv(ChannelInference):
    """
    [N, C, H, W] -> [N, C//2, H*2, W*2]
    """

    def __init__(self, ic, norm="batchnorm"):
        super().__init__(ic, ic // 2)

        self.seq = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            # NOTE: Since 2x2 conv cannot be aligned when the shape is odd,
            # the kernel size here is changed to 3x3. And padding is 1 to keep the same shape.
            nn.Conv2d(ic, ic // 2, 3, 1, 1, bias=False),
            norm_layer(norm, ic // 2),
        )

    def forward(self, X):
        return self.seq(X)


class UNetWOHeader(ChannelInference):
    """
    [N, ic, H, W] -> [N, fc * 2^level, H, W], [N, fc, H, W]
    """

    def __init__(
        # fmt: off
        self, ic, level=4, fc=64, *, 
        memory_trade=False, inner_res=False, norm='batchnorm'
        # fmt: on
    ):
        super().__init__(ic, fc * 2 ** level)
        self.level = level
        self.cps = CheckpointSupport(memory_trade)
        self.fc = fc
        self.add_module('L1', ConvStack2(ic, fc, res=inner_res))
        cc = self._L(1).oc

        for i in range(level):
            dsample = DownConv(cc, "conv")
            cc = dsample.oc
            conv = ConvStack2(cc, oc=cc * 2, res=inner_res, norm=norm)
            cc = conv.oc
            self.add_module("D%d" % (i + 1), dsample)
            self.add_module("L%d" % (i + 2), conv)

        for i in range(level):
            usample = UpConv(cc, norm=norm)
            cc = usample.oc
            conv = ConvStack2(cc * 2, cc, res=inner_res)
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


class UNet(UNetWOHeader):
    """
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

    def __init__(self, ic, oc, level=4, fc=64, headeroc=None, **kwargs):
        super().__init__(ic, level, fc, **kwargs)
        self.headers = [nn.Conv2d(self.fc, oc, 1)]
        if headeroc:
            self.headers.extend(
                nn.Sequential(nn.Tanh(), nn.Conv2d(self.fc, oc, 1), nn.Sigmoid())
                for oc in headeroc
            )
        for i, f in enumerate(self.headers):
            self.add_module("header %d" % (i + 1), f)

    def forward(self, X, expand: bool = True):
        bottomx, finalx = UNetWOHeader.forward(self, X, expand)
        if not expand:
            return bottomx

        r = [bottomx]
        r.extend(f(finalx) for f in self.headers)
        return r
