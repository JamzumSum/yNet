from abc import ABC, abstractmethod

import torch
from misc import CoefficientScheduler as CSG

from .loss import *

first = lambda it: next(iter(it))


class HasLoss(ABC):
    @abstractmethod
    def __loss__(self, *args, **kwargs) -> tuple:
        """[summary]

        Returns:
            result, lossdict
        """
        pass


class MultiTask(HasLoss):
    def __init__(self, cmgr: CSG, aug_weight=0.3333) -> None:
        self.cmgr = cmgr
        self.aug_weight = aug_weight

    def multiTaskLoss(self, loss: dict) -> torch.Tensor:
        """cal multi-task loss with coefficients specified by cmgr.

        Returns:
            torch.Tensor: cumulated multi-task loss
        """
        return sum(v * self.cmgr.get("task." + k, 1) for k, v in loss.items())

    def loss(self, *args, **kwargs):
        _, loss = self.__loss__(*args, **kwargs)
        return self.multiTaskLoss(loss)

    @staticmethod
    def lossSummary(loss: dict) -> dict:
        """return loss summary with detached tensors.

        Args:
            loss (dict): [description]

        Returns:
            dict: [description]
        """
        itemdic = {
            "pm": "m/CE",
            "tm": "m/triplet",
            "sim": "siamise/neg_cos_similarity",
            "seg": "segment/mse",
            "pb": "b/CE",
            "tb": "b/triplet",
            "seg_aug": 'segment/mse_aug'
        }
        return {"loss/" + v: loss[k].detach() for k, v in itemdic.items() if k in loss}

    def reduceLoss(self, loss: dict, aug_indices: list) -> dict:
        """reduce batch-wise loss to a loss item according to data-wise weight of a batch.

        Args:
            loss (dict[str, Tensor[float]]): [description]
            aug_indices (list): [description]

        Returns:
            dict: [description]
        """
        device = first(loss.values()).device
        aug_mask = torch.tensor(aug_indices, dtype=torch.float, device=device)
        batch_weight = (1 - (1 - self.aug_weight) * aug_mask) / len(aug_indices)
        return {k: (batch_weight * v).sum() for k, v in loss.items()}
