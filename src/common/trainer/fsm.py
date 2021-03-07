import inspect
import os
from abc import ABC, abstractclassmethod
from collections import defaultdict
from datetime import date

import pytorch_lightning as pl
import torch
from misc import CoefficientScheduler
from omegaconf import DictConfig, ListConfig, OmegaConf


class FSMBase(pl.LightningModule, ABC):
    def __init__(
        self,
        Net,
        model_conf: DictConfig,
        coeff_conf: DictConfig,
        misc: DictConfig,
        op_conf: OmegaConf,
        sg_conf: DictConfig,
    ):
        # UPSTREAM BUG: pl.LightningModule.__init__(self) failed with hparam saving...
        super().__init__()

        self.cls_name = Net.__name__

        self.misc = misc
        self.sg_conf = sg_conf
        self.model_conf = model_conf
        self.cosg = CoefficientScheduler(coeff_conf, {"piter": "x", "max_epochs": "M"})

        if OmegaConf.is_dict(op_conf):
            self.op_cls = getattr(torch.optim, op_conf.pop("name", "SGD"))
            self.op_conf = op_conf
        elif OmegaConf.is_list(op_conf):
            self.op_cls = getattr(torch.optim, op_conf[0])
            self.op_conf = {} if len(op_conf) == 1 else op_conf[1]

        self.net = Net(**model_conf)

    def save_hyperparameters(self, **otherconf):
        conf = {
            "model": self.model_conf,
            "misc": self.misc,
            "optimizer": [self.op_cls.__name__, self.op_conf],
            "scheduler": self.sg_conf,
        }
        conf.update(otherconf)
        # BUG: subclass must inherit save_hyperparameters
        frame = inspect.currentframe().f_back.f_back.f_back
        super().save_hyperparameters(conf, frame=frame)

    def traceNetwork(self):
        param = self.net.named_parameters()
        for name, p in param:
            self.logger.experiment.add_histogram(
                "network/" + name, p, self.current_epoch
            )

    @property
    def piter(self):
        """
        current_epoch / max_epochs.
        Call this only when training/testing.
        """
        return self.current_epoch / self.trainer.max_epochs

    ###########################################################################
    ######################## hooks defined below ##############################
    ###########################################################################

    def on_train_epoch_start(self):
        # self.cosg.update(piter=self.piter, max_epochs=self.trainer.max_epochs)
        self.cosg.update(piter=self.piter)

    def on_save_checkpoint(self, checkpoint: dict):
        checkpoint["seed"] = self.seed

    def on_load_checkpoint(self, checkpoint: dict):
        self.seed = checkpoint.pop("seed")

    def on_fit_start(self):
        torch.autograd.set_detect_anomaly(
            self.misc.get("detect_forward_anomaly", False)
        )
        self._score_buf = defaultdict(list)

    def score_step(self, batch, batch_idx, dataloader_idx=0):
        pass

    def score_epoch_end(self, score_outs):
        pass

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        self._score_buf[dataloader_idx].append(
            self.score_step(
                self.transfer_batch_to_device(batch), batch_idx, dataloader_idx
            )
        )

    def on_validation_epoch_end(self):
        score_outs = [self._score_buf[i] for i in range(len(self._score_buf))]
        self._score_buf.clear()
        self.score_epoch_end(score_outs)

    def on_train_epoch_end(self, outputs):
        self.traceNetwork()
