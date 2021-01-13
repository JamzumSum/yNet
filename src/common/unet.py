'''
A torch implement for U-Net.

* see: U-Net: Convolutional Networks for Biomedical Image Segmentation

* author: JamzumSum
* create: 2021-1-11
'''
import torch
import torch.nn as nn

class NeedShape:
    def __init__(self, *args):
        self.isp = args
    @property
    def ishape(self): return self.isp
    @property
    def oshape(self): return self.isp

class ConvStack2(nn.Sequential, NeedShape):
    '''
    [N, ic, H, W] -> [N, oc, H, W]
    '''
    def __init__(self, ic, ih, iw, oc):
        nn.Sequential.__init__(
            self, 
            nn.Conv2d(ic, oc, 3, 1, 1), 
            nn.ReLU(),
            nn.Conv2d(oc, oc, 3, 1, 1), 
            nn.BatchNorm2d(oc),
            nn.ReLU()
        )
        NeedShape.__init__(self, ic, ih, iw)
        self._osp = (oc, ih, iw)

    @property
    def oshape(self): return self._osp

class DownConv(nn.Conv2d, NeedShape):
    '''
    [N, C, H, W] -> [N, C, H//2, W//2]
    '''
    def __init__(self, ic, ih, iw):
        nn.Conv2d.__init__(self, ic, ic, 1, 2)
        self._osp = (ic, ih // 2, iw //2)
        NeedShape.__init__(self, ic, ih, iw)

    @property
    def oshape(self): return self._osp

class UpConv(nn.Sequential, NeedShape):
    '''
    [N, C, H, W] -> [N, C//2, H*2, W*2]
    '''
    def __init__(self, ic, ih, iw):
        nn.Sequential.__init__(
            self, 
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), 
            nn.Conv2d(ic, ic//2, 2),      # NOTE: 2x2 cannot be aligned
            nn.BatchNorm2d(ic // 2)
        )
        self.BN = nn.BatchNorm2d(ic // 2)
        NeedShape.__init__(self, ic, ih, iw)
        self._osp = (ic //2, ih * 2, iw * 2)

    @property
    def oshape(self): return self._osp


class UNet(nn.Module, NeedShape):
    '''
    [N, ic, H, W] -> [N, 64, H, W], [N, oc, H, W]
    '''
    def __init__(self, ic, ih, iw, oc):
        NeedShape.__init__(self, ic, ih, iw)
        nn.Module.__init__(self)

        self.L1 = ConvStack2(ic, ih, iw, 64)
        cshape = self.L1.oshape

        for i in range(4):
            dsample = DownConv(*cshape)
            cshape = dsample.oshape
            conv = ConvStack2(*cshape, oc=cshape[0] * 2)
            cshape = conv.oshape
            self.add_module('D%d' % (i + 1), dsample)
            self.add_module('L%d' % (i + 2), conv)

        for i in range(4):
            usample = UpConv(*cshape)
            cshape = usample.oshape
            conv = ConvStack2(cshape[0] * 2, *cshape[1:], cshape[0])
            cshape = conv.oshape
            self.add_module('U%d' % (i + 1), usample)
            self.add_module('L%d' % (i + 6), conv)
        
        self._osp = (oc, *cshape[-2:])
        self.DW = nn.Conv2d(64, oc, 1)

    @property
    def oshape(self): return self._osp

    def cropCat(self, X, Y):
        '''
        Crop X and concat to Y.
        X: [N, C, H_max, W_max]
        Y: [N, C, H_min, W_min]
        -> [N, C, H_min, W_min]
        '''
        t1 = (X.shape[2] - Y.shape[2]) // 2
        t2 = (X.shape[3] - Y.shape[3]) // 2
        cropX = X[:, :, t1: t1 + Y.shape[2], t2: t2 + Y.shape[3]]
        return torch.cat([cropX, Y], dim=1)
        
    def forward(self, X):
        '''
        X: [N, C, H, W]
        O: [N, 64, H, W], [N, oc, H, W]
        '''
        x1 = self.L1(X)             # [N, 64, H, W]
        x2 = self.L2(self.D1(x1))   # [N, 128, H//2, W//2]
        x3 = self.L3(self.D2(x2))   # [N, 256, H//4, W//4]
        x4 = self.L4(self.D3(x3))   # [N, 512, H//8, W//8]
        x5 = self.L5(self.D4(x4))   # [N, 1024, H//16, W//16]

        x6 = self.L6(self.cropCat(x4, self.U1(x5)))     # [N, 512, H//8, W//8]
        x7 = self.L7(self.cropCat(x3, self.U2(x6)))     # [N, 256, H//4, W//4]
        x8 = self.L8(self.cropCat(x2, self.U3(x7)))     # [N, 128, H//2, W//2]
        x9 = self.L9(self.cropCat(x1, self.U4(x8)))     # [N, 64, H, W]

        oh, ow = self.oshape[-2:]
        r = torch.sigmoid(self.DW(x9)[:, :, :oh, :ow])  # [N, oc, H, W]
        return x9, r

if __name__ == "__main__":
    unet = UNet(1, 572, 572, 2)
    for name, model in unet.named_modules():
        if isinstance(model, NeedShape):
            print(name, model.oshape)
    x = torch.randn(2, 1, 572, 572)
    y = unet(x)
    print(y.shape)