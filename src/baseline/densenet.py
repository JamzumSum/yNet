from . import ParalBack, SimBack
from torchvision.models import densenet


class SimDense(SimBack):
    '''
    A simple DenseNet. 
    Support: densenet121, densenet161, densenet169, densenet201
    '''
    package = densenet

    def __init__(self, cmgr, cps, K, model='densenet121', **kwargs):
        super().__init__(cmgr, cps, K, model, memory_efficient=cps.memory_trade, **kwargs)


class Densex2(ParalBack):
    '''
    A naive holder of two DenseNet. 
    Support: densenet121, densenet161, densenet169, densenet201
    '''
    package = densenet

    def __init__(self, cmgr, cps, K, model='densenet121', **kwargs):
        super().__init__(
            cmgr, cps, K, model, memory_efficient=cps.memory_trade, **kwargs
        )
