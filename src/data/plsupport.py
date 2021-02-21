import pytorch_lightning as pl
import torch

from .augment import ElasticAugmentSet, augmentWith
from .dataloader import FixLoader
from .dataset import CachedDatasetGroup, DistributedConcatSet, classSpecSplit


class DPLSet(pl.LightningDataModule):
    def __init__(self, conf, sets: list, tv=(8, 2), aug_aimsize=None, seed=None):
        self.seed = (
            int(torch.empty((), dtype=torch.int64).random_().item())
            if seed is None
            else seed
        )
        self.conf = conf
        self.sets = tuple(sets)
        self.tv = tv
        self.aimsize = aug_aimsize

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        datasets = [
            CachedDatasetGroup("./data/{setname}/{setname}.pt".format(setname=i))
            for i in self.sets
        ]
        self._td, self._vd = classSpecSplit(
            DistributedConcatSet(datasets, self.sets,),
            *self.tv,
            generator=torch.Generator().manual_seed(self.seed),
        )
        print("trainset distribution:", self._td.distribution)
        print("validation distribution:", self._vd.distribution)
        if self.aimsize is None:
            self._tda = None
        else:
            self._tda = augmentWith(self._td, ElasticAugmentSet, "Ym", self.aimsize)
            print("augmented trainset distribution:", self._tda.distribution)

    def train_dataloader(self):
        return FixLoader(
            self._td if self._tda is None else self._tda,
            **self.conf.get("training", {}),
        )

    def val_dataloader(self):
        kwargs = self.conf.get("validating", {})
        return FixLoader(self._vd, **kwargs)

