import torch
import torch.nn.functional as F
import torchvision.models.resnet as resnet
from common.loss import focalBCE

class Resx2(torch.nn.Module):
    '''
    A naive holder of two ResNet. 
    Support: resnet18, resnet34, resnet50, resnet101, resnet152
    '''
    def __init__(self, K, model='resnet50', a=1.):
        torch.nn.Module.__init__(self)
        if isinstance(model, str): model = (model, model)
        self.mbranch = getattr(resnet, model[0])(num_classes=2)
        self.bbranch = getattr(resnet, model[1])(num_classes=K)
        self.a = a

    def forward(self, X):
        return self.mbranch(X), self.bbranch(X)

    def seperatedParameters(self):
        return self.mbranch.parameters(), self.bbranch.parameters()

    def loss(self, X, Ym, Yb=None, piter=0., mweight=None, bweight=None, *args, **argv):
        Pm, Pb = self.forward(X)
        Mloss = F.cross_entropy(Pm, Ym, weight=mweight)    # use [N, 2] to cal. CE
        summary = {
            'loss/malignant CE': Mloss.detach()
        }
        if Yb is None: Bloss = 0
        else:
            Bloss = F.cross_entropy(Pb, Yb, weight=bweight)
            summary['loss/BIRADs CE'] = Bloss.detach()
        return Mloss + self.a * Bloss, summary
