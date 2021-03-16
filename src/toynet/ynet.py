"""
YNet has an additional branch at the bottom of UNet.
The additional path is used of embedding learning.

* author: JamzumSum
* create: 2021-3-15
"""
import torch.nn as nn
from common.support import SegmentSupported, SelfInitialed
from .unet import ChannelNorm, ConvStack2, DownConv, UNet


class YNet(nn.Module, SegmentSupported, SelfInitialed):
    """
    Generate embedding and segment of an image.
    YNet: image[N, 1, H, W] ->  segment[N, 1, H, W], embedding[N, D], D = fc * 2^(ul+yl)
    """

    def __init__(
        self,
        in_channel,
        width=64,
        ulevel=4,
        ylevels: list = None,
        memory_trade=False,
        residual=True,
        zero_init_residual=True,
        norm="batchnorm",
    ):
        nn.Module.__init__(self)

        self.memory_trade = memory_trade
        self.norm = norm
        self.residual = residual
        if ylevels is None:
            ylevels = []
        self.ylevel = len(ylevels)

        self.unet = UNet(
            ic=in_channel,
            oc=1,
            fc=width,
            level=ulevel,
            residual=residual,
            memory_trade=memory_trade,
            norm=norm,
        )
        cc = self.unet.oc
        ylayers = []
        for ylevel in ylevels:
            for i in range(ylevel):
                if i % ylevel:
                    ylayers.append(ConvStack2(cc, cc, self.residual, self.norm))
                else:
                    ylayers.append(ConvStack2(cc, 2 * cc, self.residual, self.norm))
                    ylayers.append(DownConv(2 * cc))
                cc = ylayers[-1].oc
        self.yoc = cc
        self.ypath = nn.Sequential(*ylayers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.selfInit()

    @property
    def max_depth(self):
        return self.unet.level + self.ylevel

    def selfInit(self, zero_init_residual=False):
        for m in self.modules():
            if isinstance(m, SelfInitialed):
                if m is self:
                    continue
                m.selfInit()
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, ConvStack2):
                    nn.init.constant_(m.CB[1].weight, 0)

    def forward(self, X, segment=True, classify=True):
        """
        args: 
            X: [N, ic, H, W]
            mask: optional [N, 1, H, W]
        flag:
            segment: if true, the whole unet will be inferenced to generate a segment map.
            classify: if true, the ypath is inferenced to get classification result.
            logit: skip softmax for some losses that needn't that.
        return: 
            segment map  [N, 2, H, W]. return this alone if `segment but not classify`
            x: bottom of unet feature. [N, fc * 2^ul]
            Pm        [N, 2]
            Pb        [N, K]
        """
        assert segment or classify, "hello?"
        if segment:
            c, seg = self.unet(X)
        else:
            c, seg = self.unet(X, False), None

        if not classify:
            return seg

        if self.ylevel:
            c = self.ypath(c)
        ft = self.pool(c)[..., 0, 0]  # [N, D], D = fc * 2^(ul + yl)
        return seg, ft
