from . import ParalBack, SimBack


class SimRes(SimBack):
    '''
    A simple ResNet. 
    Support: resnet18, resnet34, resnet50, resnet101, resnet152
    '''
    def __init__(self, cmgr, cps, model='resnet34', **kwargs):
        super().__init__(cmgr, cps, model, **kwargs)


class Resx2(ParalBack):
    '''
    A naive holder of two ResNet. 
    Support: resnet18, resnet34, resnet50, resnet101, resnet152
    '''
    def __init__(self, cmgr, cps, K, model='resnet34', **kwargs):
        super().__init__(cmgr, cps, K, model, **kwargs)
