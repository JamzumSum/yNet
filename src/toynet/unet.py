'''
A torch implement for U-Net.

* see: U-Net: Convolutional Networks for Biomedical Image Segmentation

* author: JamzumSum
* create: 2021-1-11
'''
import torch
import torch.nn as nn
from common.decorators import checkpoint, CheckpointSupport

class ChannelInference:
    def __init__(self, ic, oc):
        self.ic = ic
        self.oc = oc

class ConvStack2(nn.Module, ChannelInference):
    '''
    [N, ic, H, W] -> [N, oc, H, W]
    '''
    def __init__(self, ic, oc, res=False):
        nn.Module.__init__(self)
        self.res = res
        self.CBR = nn.Sequential(
            nn.Conv2d(ic, oc, 3, 1, 1, bias=False), 
            nn.BatchNorm2d(oc),
            nn.ReLU()
        )
        self.CB = nn.Sequential(
            nn.Conv2d(oc, oc, 3, 1, 1, bias=False), 
            nn.BatchNorm2d(oc)
        )
        ChannelInference.__init__(self, ic, oc)

    def forward(self, X):
        X = self.CBR(X)
        if self.res: X = X + self.CB(X)
        else: X = self.CB(X)
        return torch.relu(X)

class DownConv(nn.Conv2d, ChannelInference):
    '''
    [N, C, H, W] -> [N, C, H//2, W//2]
    '''
    def __init__(self, ic):
        nn.Conv2d.__init__(self, ic, ic, 2, 2, bias=False)
        ChannelInference.__init__(self, ic, ic)
        torch.nn.init.constant_(self.weight, 0.25)

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
            nn.Conv2d(ic, ic//2, 3, 1, 1, bias=False),      
            nn.BatchNorm2d(ic // 2)
        )
        ChannelInference.__init__(self, ic, ic // 2)

class UNetWOHeader(nn.Module):
    '''
    [N, ic, H, W] -> [N, fc * 2^level, H, W], [N, fc, H, W]
    '''
    def __init__(self, ic, level=4, fc=64, inner_res=False, memory_trade=False):
        nn.Module.__init__(self)
        self.level = level
        self.cps = CheckpointSupport(memory_trade)
        self.fc = fc
        self.L1 = ConvStack2(ic, fc, res=inner_res)
        cc = self.L1.oc

        for i in range(level):
            dsample = DownConv(cc)
            cc = dsample.oc
            conv = ConvStack2(cc, oc=cc * 2, res=inner_res)
            cc = conv.oc
            self.add_module('D%d' % (i + 1), dsample)
            self.add_module('L%d' % (i + 2), conv)

        for i in range(level):
            usample = UpConv(cc)
            cc = usample.oc
            conv = ConvStack2(cc * 2, cc, res=inner_res)
            cc = conv.oc
            self.add_module('U%d' % (i + 1), usample)
            self.add_module('L%d' % (i + self.level + 2), conv)
        

    def add_module(self, name, model):
        return nn.Module.add_module(self, name, self.cps(model))

    def _padCat(self, X, Y):
        '''
        Pad Y and concat with X.
        X: [N, C, H_max, W_max]
        Y: [N, C, H_min, W_min]
        -> [N, C, H_max, W_max]
        '''
        # hmax, wmax = X.shape[-2:]
        # hmin, wmin = Y.shape[-2:]
        # padl = (wmax - wmin) // 2
        # padr = wmax - wmin - padl
        # padt = (hmax - hmin) // 2
        # padb = hmax - hmin - padt
        # Y = torch.nn.functional.pad(Y, (padl, padr, padt, padb), 'constant')
        return torch.cat([X, Y], dim=1)
        
    def forward(self, X, expand=True):
        '''
        X: [N, C, H, W]
        O: [N, fc, H, W], [N, oc, H, W]
        '''        
        if self.cps.memory_trade: X = X.clone().requires_grad_()
        xn = [self.L1(X)]
        L_ = lambda i: self._modules['L%d' % i]
        D_ = lambda i: self._modules['D%d' % i]
        U_ = lambda i: self._modules['U%d' % i]
        X_ = lambda i: xn[i - 1]

        for i in range(1, self.level + 1):
            xn.append(L_(i + 1)(D_(i)(xn[-1])))   # [N, t * fc, H//t, W//t], t = 2^i
        bottomx = xn[-1]
        if not expand: return bottomx, None

        for i in range(self.level):
            xn.append(L_(self.level + i + 2)(
                # [N, t*fc, H//t, W//t], t = 2^(level - i - 1)
                self._padCat(X_(self.level - i), U_(i + 1)(X_(self.level + i + 1)))
            ))

        return bottomx, xn[-1]
        

class UNet(UNetWOHeader):
    def __init__(self, oc, headers=[], *args, **kwargs):
        UNetWOHeader.__init__(self, *args, **kwargs)
        self.headers = [nn.Conv2d(self.fc, oc, 1)]
        self.headers.extend(
            nn.Sequential(nn.Tanh(), nn.Conv2d(self.fc, oc, 1)) for oc in headers
        )
        for i, f in enumerate(self.headers): self.add_module("header %d" % (i + 1), f)
        self.sigma = nn.Sigmoid()

    def forward(self, X, expand=True):
        bottomx, finalx = UNetWOHeader.forward(self, X, expand)
        if not expand: return (bottomx, *(None for f in self.headers))

        act = (self.sigma(f(finalx)) for f in self.headers)
        return (bottomx, *act)

if __name__ == "__main__":
    unet = UNet(3, 1, 16)
    x = torch.randn(2, 3, 512, 512)
    y = unet(x)
    print(i.shape for i in y)
