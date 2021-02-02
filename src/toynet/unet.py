'''
A torch implement for U-Net.

* see: U-Net: Convolutional Networks for Biomedical Image Segmentation

* author: JamzumSum
* create: 2021-1-11
'''
import torch
import torch.nn as nn

class ChannelInference:
    def __init__(self, ic, oc):
        self.ic = ic
        self.oc = oc

class ConvStack2(nn.Sequential, ChannelInference):
    '''
    [N, ic, H, W] -> [N, oc, H, W]
    '''
    def __init__(self, ic, oc):
        nn.Sequential.__init__(
            self, 
            nn.Conv2d(ic, oc, 3, 1, 1), 
            nn.ReLU(),
            nn.Conv2d(oc, oc, 3, 1, 1), 
            nn.BatchNorm2d(oc),
            nn.ReLU()
        )
        ChannelInference.__init__(self, ic, oc)
class DownConv(nn.Conv2d, ChannelInference):
    '''
    [N, C, H, W] -> [N, C, H//2, W//2]
    '''
    def __init__(self, ic):
        nn.Conv2d.__init__(self, ic, ic, 1, 2)
        ChannelInference.__init__(self, ic, ic)

class UpConv(nn.Sequential, ChannelInference):
    '''
    [N, C, H, W] -> [N, C//2, H*2, W*2]
    '''
    def __init__(self, ic):
        nn.Sequential.__init__(
            self, 
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), 
            # NOTE: Since 2x2 conv cannot be aligned when the shape is odd, 
            # the kernel size here is changed to 3x3. And padding is 1 to keep the same shape.
            nn.Conv2d(ic, ic//2, 3, 1, 1),      
            nn.BatchNorm2d(ic // 2)
        )
        self.BN = nn.BatchNorm2d(ic // 2)
        ChannelInference.__init__(self, ic, ic // 2)

class UNet(nn.Module):
    '''
    [N, ic, H, W] -> [N, fc, H, W], [N, oc, H, W]
    '''
    def __init__(self, ic, oc, fc=64, softmax=False):
        nn.Module.__init__(self)
        self.softmax = softmax

        self.L1 = ConvStack2(ic, fc)
        cc = self.L1.oc

        for i in range(4):
            dsample = DownConv(cc)
            cc = dsample.oc
            conv = ConvStack2(cc, oc=cc * 2)
            cc = conv.oc
            self.add_module('D%d' % (i + 1), dsample)
            self.add_module('L%d' % (i + 2), conv)

        for i in range(4):
            usample = UpConv(cc)
            cc = usample.oc
            conv = ConvStack2(cc * 2, cc)
            cc = conv.oc
            self.add_module('U%d' % (i + 1), usample)
            self.add_module('L%d' % (i + 6), conv)
        
        self.DW = nn.Conv2d(fc, oc, 1)

    def _padCat(self, X, Y):
        '''
        Pad Y and concat with X.
        X: [N, C, H_max, W_max]
        Y: [N, C, H_min, W_min]
        -> [N, C, H_max, W_max]
        '''
        hmax, wmax = X.shape[-2:]
        hmin, wmin = Y.shape[-2:]
        padl = (wmax - wmin) // 2
        padr = wmax - wmin - padl
        padt = (hmax - hmin) // 2
        padb = hmax - hmin - padt
        Y = torch.nn.functional.pad(Y, (padl, padr, padt, padb), 'constant')
        return torch.cat([X, Y], dim=1)
        
    def forward(self, X):
        '''
        X: [N, C, H, W]
        O: [N, fc, H, W], [N, oc, H, W]
        '''
        x1 = self.L1(X)             # [N, fc, H, W]
        x2 = self.L2(self.D1(x1))   # [N, 2*fc, H//2, W//2]
        x3 = self.L3(self.D2(x2))   # [N, 4*fc, H//4, W//4]
        x4 = self.L4(self.D3(x3))   # [N, 8*fc, H//8, W//8]
        x5 = self.L5(self.D4(x4))   # [N, 16*fc, H//16, W//16]

        x6 = self.L6(self._padCat(x4, self.U1(x5)))     # [N, 8*fc, H//8, W//8]
        x7 = self.L7(self._padCat(x3, self.U2(x6)))     # [N, 4*fc, H//4, W//4]
        x8 = self.L8(self._padCat(x2, self.U3(x7)))     # [N, 2*fc, H//2, W//2]
        x9 = self.L9(self._padCat(x1, self.U4(x8)))     # [N, fc, H, W]
        logit = self.DW(x9)         # [N, oc, H, W]

        if self.softmax: 
            return x9, torch.softmax(logit, 1)
        else: 
            return x9, torch.sigmoid(logit)

if __name__ == "__main__":
    unet = UNet(3, 1, 16)
    x = torch.randn(2, 3, 512, 512)
    y = unet(x)
    print(i.shape for i in y)