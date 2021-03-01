import pytorch_lightning as pl
import torch

from .augment import ElasticAugmentSet, augmentWith
from .dataloader import FixLoader
from .dataset import CachedDatasetGroup, DistributedConcatSet, classSpecSplit
from common.support import DeviceAwareness


class DPLSet(pl.LightningDataModule, DeviceAwareness):
    train_dataloader_num = 1
    val_dataloader_num = 1

    def __init__(self, conf, sets: list, tv=(8, 2), aug_aimsize=None, device=None):
        DeviceAwareness.__init__(self, device)
        self.conf = conf
        self.sets = tuple(sets)
        self.tv = tv
        self.aimsize = aug_aimsize

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        datasets = [
            CachedDatasetGroup(
                "./data/{setname}/{setname}.pt".format(setname=i), self.device
            )
            for i in self.sets
        ]
        self._ad = DistributedConcatSet(datasets, self.sets,)
        self._td, self._vd = classSpecSplit(self._ad, *self.tv,)
        print("trainset distribution:", self._td.distribution)
        print("validation distribution:", self._vd.distribution)
        self.score_caption = ("validation", "testset")
        
        if self.aimsize is None:
            self._tda = None
        else:
            self._tda = augmentWith(self._td, ElasticAugmentSet, "Ym", self.aimsize)
            print("augmented trainset distribution:", self._tda.distribution)

    def train_dataloader(self):
        return FixLoader(
            self._td if self._tda is None else self._tda,
            **self.conf.get("training", {}),
            device=self.device,
        )

    def val_dataloader(self):
        kwargs = self.conf.get("validating", {})
        return (
            FixLoader(self._vd, **kwargs, device=self.device),
            FixLoader(self._td, **kwargs, device=self.device),
        )

    def test_dataloader(self):
        return FixLoader(
            self._ad, **self.conf.get("validating", {}), device=self.device
        )

