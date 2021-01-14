'''
A toy implement for classifying benign/malignant and BIRADs

* author: JamzumSum
* create: 2021-1-11
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.unet import UNet


class BIRADsUNet(UNet):
    '''
    [N, 1, H, W] -> [N, 1, H, W], [N, K, H, W]
    '''
    def __init__(self, ic, ih, iw, K):
        UNet.__init__(self, ic, ih, iw, 1)
        self.BDW = nn.Conv2d(64, K, 1)

    def forward(self, X):
        '''
        X: [N, 1, H, W]
        return: 
        - benign/malignant Class Activation Mapping     [N, 1, H, W]
        - BIRADs CAM                    [N, H, W, K]
        '''
        x9, Mhead = UNet.forward(self, X)
        Bhead = torch.sigmoid(self.BDW(torch.tanh(x9)))
        return Mhead, Bhead

class ToyNetV1(nn.Module):
    def __init__(self, ishape, K, patch_size, a=1.):
        nn.Module.__init__(self)
        self.backbone = BIRADsUNet(*ishape, K)
        self.pooling = nn.AvgPool2d(patch_size)
        self.a = a

    def forward(self, X):
        '''
        X: [N, 1, H, W]
        return: 
        - benign/malignant Class Activation Mapping     [N, 1, H, W]
        - BIRADs CAM                    [N, H, W, K]
        - malignant confidence          [N, 1]
        - BIRADs prediction vector      [N, K]
        '''
        Mhead, Bhead = self.backbone(X)
        Mpatches = self.pooling(Mhead)      # [N, 1, H//P, W//P]
        Bpatches = self.pooling(Bhead)      # [N, K, H//P, W//P]

        Mp = torch.amax(Mpatches, dim=(2, 3))        # [N, 1]
        Bp = torch.amax(Bpatches, dim=(2, 3))        # [N, K]
        # Bp = torch.softmax(Bp, -1)        # NOTE: CANNOT use softmax here for crossentropy in torch applies softmax itself
        return Mhead, Bhead, Mp, Bp

    def loss(self, X, Ym, Ybirad):
        '''
        X: [N, 1, H, W]
        Ym: [N], long
        Ybirad: [N], long
        '''
        M, B, Pm, Pb = self.forward(X)      # ToyNetV1 discards two CAMs
        Mloss = F.cross_entropy(torch.cat([1 - Pm, Pm], dim=-1), Ym)    # use [N, 2] to cal. CE
        Bloss = F.cross_entropy(Pb, Ybirad)
        return Mloss + self.a * Bloss, {
            'malignant entropy': Mloss.detach(), 
            'BIRADs entropy': Bloss.detach(), 
            'malignant CAM': M.detach(), 
            'BIRADs CAM': B.detach()
        }

if __name__ == "__main__":
    x = torch.randn(2, 1, 572, 572)
    toy = ToyNetV1(
        (1, 572, 572), 
        5, 12
    )
    loss, _ = toy.loss(x, torch.zeros(2, dtype=torch.long), torch.ones(2, dtype=torch.long))
    loss.backward()
