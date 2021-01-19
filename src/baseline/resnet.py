import torch
import torch.nn.functional as F
import torchvision.models.resnet as resnet

class ResForB(torch.nn.Module):
    """
    Hold only one ResNet for classifying BIRADs.
    Malignant loss will be 0.
    Support: resnet18, resnet34, resnet50, resnet101, resnet152
    """
    hotmap = False

    def __init__(self, K, model='resnet50'):
        torch.nn.Module.__init__(self)
        self.bbranch = getattr(resnet, model)(num_classes=K)
    
    def forward(self, X):
        return self.bbranch(X)

    def seperatedParameters(self):
        return (i for i in ()), self.bbranch.parameters()

    def loss(self, X, Ym, Yb=None, piter=0.):
        Pb = self.forward(X)
        Mloss = torch.zeros(1).sum()
        summary = {
            'loss/malignant CE': Mloss
        }
        if Yb is None: Bloss = 0
        else:
            Bloss = F.cross_entropy(Pb, Yb, weight=self.bbalance)
            summary['loss/BIRADs CE'] = Bloss.detach()
        return Mloss + self.a * Bloss, summary


class Resx2(ResForB):
    '''
    A naive holder of two ResNet. 
    Support: resnet18, resnet34, resnet50, resnet101, resnet152
    '''
    def __init__(self, K, model='resnet34', a=1.):
        ResForB.__init__(self, K, model)
        self.mbranch = getattr(resnet, model)(num_classes=2)
        self.a = a

    def forward(self, X):
        return self.mbranch(X), self.bbranch(X)

    def seperatedParameters(self):
        return self.mbranch.parameters(), self.bbranch.parameters()

    def loss(self, X, Ym, Yb=None, piter=0.):
        Pm, Pb = self.forward(X)
        Mloss = F.cross_entropy(Pm, Ym)    # use [N, 2] to cal. CE
        summary = {
            'loss/malignant CE': Mloss.detach()
        }
        if Yb is None: Bloss = 0
        else:
            Bloss = F.cross_entropy(Pb, Yb, weight=self.bbalance)
            summary['loss/BIRADs CE'] = Bloss.detach()
        return Mloss + self.a * Bloss, summary

