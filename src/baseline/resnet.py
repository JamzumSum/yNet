from . import ParalBack, SimBack


class SimRes(SimBack):
    '''
    A simple ResNet. 
    Support: resnet18, resnet34, resnet50, resnet101, resnet152
    '''
    def __init__(self, cmgr, cps, K, model='resnet34', **kwargs):
        super().__init__(cmgr, cps, K, model, **kwargs)
        self.mbranch.layer1 = self.cps(self.mbranch.layer1)
        self.mbranch.layer2 = self.cps(self.mbranch.layer2)
        self.mbranch.layer3 = self.cps(self.mbranch.layer3)
        self.mbranch.layer4 = self.cps(self.mbranch.layer4)


class Resx2(ParalBack):
    '''
    A naive holder of two ResNet. 
    Support: resnet18, resnet34, resnet50, resnet101, resnet152
    '''
    def __init__(self, cmgr, cps, K, model='resnet34', **kwargs):
        super().__init__(cmgr, cps, K, model, **kwargs)
        self.mbranch.layer1 = self.cps(self.mbranch.layer1)
        self.mbranch.layer2 = self.cps(self.mbranch.layer2)
        self.mbranch.layer3 = self.cps(self.mbranch.layer3)
        self.mbranch.layer4 = self.cps(self.mbranch.layer4)

        self.bbranch.layer1 = self.cps(self.bbranch.layer1)
        self.bbranch.layer2 = self.cps(self.bbranch.layer2)
        self.bbranch.layer3 = self.cps(self.bbranch.layer3)
        self.bbranch.layer4 = self.cps(self.bbranch.layer4)
