from abc import ABC, abstractclassmethod, abstractproperty
import torch


class SelfInitialed(ABC):
    @abstractclassmethod
    def selfInit(self):
        pass


class HeatmapSupported(ABC):
    pass


class SegmentSupported(ABC):
    pass


class HasDiscriminator(ABC):
    @abstractclassmethod
    def discrim_weight(self, weight_decay):
        pass


class MultiBranch(ABC):
    @abstractclassmethod
    def branch_weight(self, weight_decay: dict):
        pass

    @abstractproperty
    def branches(self):
        pass


class DeviceAwareness:
    def __init__(self, device: str):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
