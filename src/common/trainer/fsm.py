import inspect
from abc import ABC
from collections import defaultdict

import pytorch_lightning as pl
import torch
from misc import CheckpointSupport, CoefficientScheduler
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch import nn


def splitNameConf(conf, search, default_name: str = None):
    if OmegaConf.is_dict(conf):
        return getattr(search, conf.pop("name", default_name)), conf
    elif OmegaConf.is_list(conf):
        return getattr(search, conf[0]), {} if len(conf) == 1 else conf[1]


class FSMBase(pl.LightningModule, ABC):
    def __init__(
        self,
        net: nn.Module,
        cmgr: CoefficientScheduler,
        conf: DictConfig,
    ):
        # UPSTREAM BUG: pl.LightningModule.__init__(self) failed with hparam saving...
        super().__init__()

        self.misc = conf.misc
        self.op_cls, self.op_conf = splitNameConf(conf.optimizer, torch.optim, 'SGD')
        self.sg_cls, self.sg_conf = splitNameConf(conf.scheduler, torch.optim.lr_scheduler)

        self.cosg = cmgr

        self.net = net
        self.save_hyperparameters(OmegaConf.to_container(conf))

    def traceNetwork(self):
        param = self.net.named_parameters()
        for name, p in param:
            self.logger.experiment.add_histogram(
                "network/" + name, p, self.current_epoch
            )

    @property
    def cls_name(self):
        return self.Net.__name__

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
        self.log("lr", lr, on_step=False, on_epoch=True)

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

    def score_step(self, batch, batch_idx, dataloader_idx=0) -> dict:
        raise NotImplementedError

    def score_epoch_end(self, score_outs):
        raise NotImplementedError

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        r = self.score_step(
            self.transfer_batch_to_device(batch), batch_idx, dataloader_idx
        )
        if r is None: return
        self._score_buf[dataloader_idx].append(r)

    def on_validation_epoch_end(self):
        score_outs = [self._score_buf[i] for i in range(len(self._score_buf))]
        self._score_buf.clear()
        self.score_epoch_end(score_outs)

    def on_train_epoch_end(self, outputs):
        self.traceNetwork()
