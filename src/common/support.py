from abc import ABC, abstractclassmethod
import torch

class HeatmapSupported(ABC):
    pass


class SegmentSupported(ABC):
    pass


class HasDiscriminator(ABC):
    pass

class DeviceAwareness:
    def __init__(self, device: str):
        if device is None: 
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)


__all__ = ["HeatmapSupported", "SegmentSupported", "HasDiscriminator"]

