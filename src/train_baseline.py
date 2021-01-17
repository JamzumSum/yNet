import torch
import torch.nn.functional as F
import yaml
import torchvision.models.resnet as resnet


from dataloader import trainValidSplit
from spectrainer import ToyNetTrainer

class Resx2(torch.nn.Module):
    '''
    A naive holder of two ResNet. 
    Support: resnet18, resnet34, resnet50, resnet101, resnet152
    '''
    hotmap = False

    def __init__(self, K, model='resnet50', a=1.):
        torch.nn.Module.__init__(self)
        self.mbranch = getattr(resnet, model)(num_classes=2)
        self.bbranch = getattr(resnet, model)(num_classes=K)
        self.a = a

    def forward(self, X):
        return self.mbranch(X), self.bbranch(X)

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

if __name__ == "__main__":
    (ta, tu), (va, vu) = trainValidSplit(8, 2)
    print('annotated train set:', len(ta))
    print('unannotated train set:', len(tu))
    print('annotated validation set:', len(va))
    print('unannotated validation set:', len(vu))

    conf = {}
    with open('./config/resnet50.yml') as f: conf = yaml.safe_load(f)
    trainer = ToyNetTrainer(Resx2, conf)
    trainer.train(ta, tu, va, vu)
