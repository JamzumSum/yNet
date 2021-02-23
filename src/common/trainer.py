"""
A universial trainer ready to be inherited.

* author: JamzumSum
* create: 2021-1-12
"""
import os
from abc import ABC, abstractclassmethod
from collections import defaultdict
from datetime import date

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from data.plsupport import DPLSet


class LightningBase(pl.LightningModule):
    def __init__(self, Net, conf: dict):
        # UPSTREAM BUG: pl.LightningModule.__init__(self) failed with hparam saving...
        super().__init__()

        self.cls_name = Net.__name__
        self.save_hyperparameters(conf)

        self.paths = conf.get("paths", {})
        self.misc = conf.get("misc", {})

        op_conf = conf.get("optimizer", ("SGD", {}))
        self.op_name = getattr(torch.optim, op_conf[0])
        self.op_conf = {} if len(op_conf) == 1 else op_conf[1]
        self.model_conf = conf.get("model", {})

        self.max_epochs = conf.get("flag", {}).get("max_epochs", 1)

        if self.misc.get("continue", True):
            self._load()
        else:
            self.net = Net(**self.model_conf)
            self.seed = None

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

    def _load(self):
        name = self.misc.get("load_from", "latest")
        path = os.path.join(self.model_dir, name + ".pt")
        self.load_from_checkpoint(path)

    def on_save_checkpoint(self, checkpoint: dict):
        checkpoint["seed"] = self.seed

    def on_load_checkpoint(self, checkpoint: dict):
        self.seed = checkpoint.pop("seed")

    def on_fit_start(self):
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


class Trainer(pl.Trainer):
    def __init__(self, PLNet: LightningBase, Net: torch.nn.Module, conf: dict):
        self.cls_name = Net.__name__
        self.misc = conf.get("misc", {})
        self.paths = conf.get("paths", {})

        checkpoint_callback = ModelCheckpoint(
            monitor="err/B-M/validation",
            dirpath=self.model_dir,
            filename="best",
            save_last=True,
            mode="min",
        )
        checkpoint_callback.FILE_EXTENSION = ".pt"
        checkpoint_callback.best_model_path = self.model_dir
        checkpoint_callback.CHECKPOINT_NAME_LAST = "latest"

        board = TensorBoardLogger(
            self.log_dir, self.name, log_graph=True, default_hp_metric=False
        )

        pl.Trainer.__init__(
            self,
            callbacks=[checkpoint_callback],
            default_root_dir=self.model_dir,
            logger=board,
            **conf.get("flag", {})
        )

        self.net = PLNet(Net, conf)
        self.ds = DPLSet(
            conf.get("dataloader", {}),
            conf["datasets"],
            (8, 2),
            self.misc.get("augment", None),
            {"GPU": "cuda", "CPU": "cpu"}[self._device_type],
            self.net.seed,
        )
        self.net.seed = self.ds.seed
        self.net.train_dataloader_num = self.ds.train_dataloader_num
        self.net.test_caption = ["validation", "testset"]

        self.ds.prepare_data()
        self.ds.setup()

    def fit(self, model=None):
        return pl.Trainer.fit(self, model or self.net, datamodule=self.ds)

    def tune(self, model=None):
        return pl.Trainer.tune(self, model or self.net, datamodule=self.ds)

    @property
    def log_dir(self) -> str:
        return self.paths.get("log_dir", os.path.join("log", self.cls_name)).format(
            date=date.today().strftime("%m%d")
        )

    @property
    def model_dir(self):
        return self.paths.get("model_dir", os.path.join("model", self.cls_name)).format(
            date=date.today().strftime("%m%d")
        )

    @property
    def name(self):
        return self.paths.get("name", "default")
