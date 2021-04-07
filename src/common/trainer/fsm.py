import inspect
import os
from abc import ABC, abstractclassmethod
from collections import defaultdict
from datetime import date
import omegaconf

import pytorch_lightning as pl
import torch
from misc import CoefficientScheduler, CheckpointSupport
from omegaconf import DictConfig, ListConfig, OmegaConf


def splitNameConf(conf, search, default_name: str = None):
    if OmegaConf.is_dict(conf):
        return getattr(search, conf.pop("name", default_name)), conf
    elif OmegaConf.is_list(conf):
        return getattr(search, conf[0]), {} if len(conf) == 1 else conf[1]


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
        # TODO: save hparam
        super().__init__()

        self.cls_name = Net.__name__

        self.misc = misc
        self.model_conf = model_conf
        self.op_cls, self.op_conf = splitNameConf(op_conf, torch.optim, 'SGD')
        self.sg_cls, self.sg_conf = splitNameConf(sg_conf, torch.optim.lr_scheduler)

        # init cmgr
        self.cosg = CoefficientScheduler(coeff_conf, {"piter": "x", "max_epochs": "M"})
        self.cosg.update(piter=0)

        self.net = Net(
            cmgr=self.cosg,
            cps=CheckpointSupport(misc.get('memory_trade', False)),
            **model_conf
        )

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

    @property
    def steps_per_epoch(self):
        td = self.train_dataloader()
        r = sum(len(i) for i in td) if isinstance(td, (list, tuple)) else len(td)
        limit = self.trainer.limit_train_batches
        if isinstance(limit, float): limit = int(limit * r)
        return min(r, limit)

    ###########################################################################
    ######################## hooks defined below ##############################
    ###########################################################################

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx=0):
        op = self.trainer.optimizers[0]
        lr = op.param_groups[0]["lr"]
        if lr < 1e-8:
            return -1
        self.log("lr", lr, True, True, True, False)

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
