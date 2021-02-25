import os
from abc import ABC
from collections import defaultdict
from datetime import date

import pytorch_lightning as pl
import torch


class FSMBase(pl.LightningModule, ABC):
    def __init__(self, Net, conf: dict):
        # UPSTREAM BUG: pl.LightningModule.__init__(self) failed with hparam saving...
        super().__init__()

        self.cls_name = Net.__name__
        self.save_hyperparameters(dict(conf))

        self.paths = conf["paths"]
        self.misc = conf["misc"]

        op_conf = conf.get("optimizer", ("SGD", {}))
        self.op_name = getattr(torch.optim, op_conf[0])
        self.op_conf = {} if len(op_conf) == 1 else op_conf[1]
        self.sg_conf = conf["scheduler"]
        self.model_conf = conf["model"]

        self.max_epochs = conf["flag"].get("max_epochs", 1)

        if self.misc.get("continue", True):
            self._load(Net)
        else:
            self.net = Net(**self.model_conf)
            self.seed = int(
                torch.empty((), dtype=torch.int64).random_(4294967295).item()
            )
        pl.utilities.seed.seed_everything(self.seed)

    def traceNetwork(self):
        param = self.net.named_parameters()
        for name, p in param:
            self.logger.experiment.add_histogram(
                "network/" + name, p, self.current_epoch
            )

    @property
    def model_dir(self):
        return self.paths.get("model_dir", os.path.join("model", self.cls_name)).format(
            date=date.today().strftime("%m%d")
        )

    @property
    def piter(self):
        return self.current_epoch / self.max_epochs

    def _load(self, Net):
        name = self.misc.get("load_from", "latest")
        path = os.path.join(self.model_dir, name + ".pt")
        self.load_from_checkpoint(path, Net=Net)

    def on_save_checkpoint(self, checkpoint: dict):
        checkpoint["seed"] = self.seed

    def on_load_checkpoint(self, checkpoint: dict):
        self.seed = checkpoint.pop("seed")

    def on_fit_start(self):
        torch.autograd.set_detect_anomaly(True)
        self._score_buf = defaultdict(list)

    def score_step(self, batch, batch_idx, dataloader_idx=0):
        pass

    def score_epoch_end(self, score_outs):
        pass

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        self._score_buf[dataloader_idx].append(
            self.score_step(batch, batch_idx, dataloader_idx)
        )

    def on_validation_epoch_end(self):
        score_outs = [self._score_buf[i] for i in range(len(self._score_buf))]
        self._score_buf.clear()
        self.score_epoch_end(score_outs)

    def on_train_epoch_end(self, outputs):
        self.traceNetwork()
