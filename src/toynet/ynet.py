"""
YNet has an additional branch at the bottom of UNet.
The additional path is used of embedding learning.

* author: JamzumSum
* create: 2021-3-15
"""
import torch.nn as nn
from common.support import SegmentSupported, SelfInitialed
from misc import CheckpointSupport

from .unet import ConvStack2, DownConv, UNet


class YNet(nn.Module, SegmentSupported, SelfInitialed):
    """Generate embedding and segment of an image.
    YNet: image[N, 1, H, W] ->  segment[N, 1, H, W], embedding[N, D], 
    D = fc * 2^(ul+yl)

    Args:
        cps (CheckpointSupport)
        in_channel (int): input channel
        width (int, optional): [description]. Defaults to 64.
        ulevel (int, optional): unet level. Defaults to 4.
        ylevels (list, optional): [description]. Defaults to None.
        residual (bool, optional): use res-block as base unit. Defaults to True.
        zero_init_residual (bool, optional): init residual path as 0. Defaults to True.
        norm (str, optional): [description]. Defaults to "batchnorm".
        multiscale (bool, int, optional): atrous layer num. Defaults to False(0).
    """
    def __init__(
        self,
        cps: CheckpointSupport,
        in_channel,
        width=64,
        ulevel=4,
        *,
        ylevels: list = None,
        residual=True,
        zero_init_residual=True,
        norm="batchnorm",
        multiscale=False,
    ):
        nn.Module.__init__(self)

        if ylevels is None:
            ylevels = []
        self.ylevel = len(ylevels)

        self.unet = UNet(
            in_channel,
            1,
            ulevel,
            width,
            cps=cps,
            residual=residual,
            norm=norm,
            multiscale=multiscale,
        )
        cc = self.unet.oc

        uniarg = dict(res=residual, norm=norm, atrous_num=int(multiscale))
        gen = ((ylevel, i) for ylevel in ylevels for i in range(ylevel))
        ylayers = []

        for ylevel, i in gen:
            if i % ylevel:
                ylayers.append(cps(ConvStack2(cc, cc, **uniarg)))
            else:
                ylayers.append(cps(ConvStack2(cc, 2 * cc, **uniarg)))
                ylayers.append(DownConv(2 * cc))
            cc = ylayers[-1].oc
        self.yoc = cc
        self.ypath = nn.Sequential(*ylayers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.selfInit(zero_init_residual)

    def selfInit(self, zero_init_residual=False):
        for m in self.modules():
            if isinstance(m, SelfInitialed):
                if m is self: continue
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

    def forward(self, X, segment=True, classify=True) -> dict:
        """[summary]

        Args:
            X (Tensor): [N, ic, H, W]
            segment (bool, optional): if true, the whole unet will be inferenced to generate a segment map.
            classify (bool, optional): if true, the ypath is inferenced to get classification result. 

        Returns:
            dict: [description]
        """
        assert segment or classify, "hello?"
        d = self.unet(X, segment)

        r = {}
        if segment: r['seg'] = d['seg0']
        if not classify: return r

        c = d["bottom"]
        if self.ylevel:
            c = self.ypath(c)
        r["ft"] = self.pool(c)[..., 0, 0]      # [N, D], D = fc * 2^(ul + yl)
        return r
