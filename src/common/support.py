from abc import ABC, abstractmethod, abstractproperty
import torch


class SelfInitialed(ABC):
    @abstractmethod
    def selfInit(self):
        pass


class HeatmapSupported(ABC):
    pass


class SegmentSupported(ABC):
    pass


class HasDiscriminator(ABC):
    @abstractmethod
    def discrim_weight(self, weight_decay):
        pass


class MultiBranch(ABC):
    @abstractmethod
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
