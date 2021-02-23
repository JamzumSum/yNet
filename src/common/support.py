from abc import ABC, abstractclassmethod


class HeatmapSupported(ABC):
    pass


class SegmentSupported(ABC):
    pass


class HasDiscriminator(ABC):
    pass


__all__ = ["HeatmapSupported", "SegmentSupported", "HasDiscriminator"]

