from typing import Optional, Union
import pytorch_lightning as pl
import torch
from misc.decorators import autoPropertyClass
from omegaconf import DictConfig

from .augment.offline import ElasticAugmentSet, augmentWith
from .dataloader import FixBatchLoader, FixLoader
from .dataset import DistributedConcatSet, classSpecSplit
from .dataset.cacheset import CachedDatasetGroup


@autoPropertyClass
class DPLSet(pl.LightningDataModule):
    tv: tuple[int, int]
    mask_prob: float
    device: Optional[Union[str, torch.device]]

    def __init__(
        self,
        dataloader_conf: dict,
        sets: Union[list, dict],
        *,
        tv: tuple = (8, 2),
        mask_prob: float = 1.,
        aug_aimsize: int = None,
        device=None,
    ):
        """[summary]

        Args:
            dataloader_conf (dict): config of dataloaders
            sets (Union[list, dict]): datasets and corresponding aug_aimsize
            tv (tuple, optional): train/validation split rate. Defaults to (8, 2).
            mask_prob (float, optional): usage of mask. Defaults to 1..
            aug_aimsize (int, optional): total aim size after augment. Defaults to None.
            device ([type], optional): augment on this device. Defaults to None.
        """
        super().__init__()

        self.conf = dataloader_conf
        if isinstance(sets, (dict, DictConfig)):
            self.sets = tuple(sets.keys())
            self.aimsize = tuple(sets.values())
        else:
            self.sets = tuple(sets)
            self.aimsize = aug_aimsize

    def prepare_data(self):
        datasets = [
            CachedDatasetGroup(f"./data/{i}", self.device, self.mask_prob)
            for i in self.sets
        ]
        self._ad = DistributedConcatSet(
            datasets,
            self.sets,
        )

    def setup(self, stage=None):
        self._td, self._vd = classSpecSplit(self._ad, *self.tv, 'Ym')
        print("trainset distribution:", self._td.distribution)
        print("validation distribution:", self._vd.distribution)
        self.score_caption = ("validation", "trainset")

        if self.aimsize is None:
            self._tda = None
        else:
            self._tda = augmentWith(
                ElasticAugmentSet, self._td, self.aimsize, device=self.device
            )
            print("augmented trainset distribution:", self._tda.distribution)

    def train_dataloader(self):
        return FixBatchLoader(
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
            self._vd,
            pass_pid=True,
            **self.conf.get("validating", {}),
            device=self.device
        )
