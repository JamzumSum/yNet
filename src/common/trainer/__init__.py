"""
A universial trainer ready to be inherited.

* author: JamzumSum
* create: 2021-1-12
"""
import os
from collections import defaultdict
from datetime import date

import pytorch_lightning as pl
import torch
from data.plsupport import DPLSet
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from .richbar import RichProgressBar


class Trainer(pl.Trainer):
    def __init__(self, FSM, Net: torch.nn.Module, conf: dict):
        conf = defaultdict(dict, conf)

        self.cls_name = Net.__name__
        self.misc = conf["misc"]
        self.paths = conf["paths"]

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
            callbacks=[checkpoint_callback, RichProgressBar()],
            default_root_dir=self.model_dir,
            logger=board,
            **conf["flag"]
        )

        self.net = FSM(Net, conf)
        self.ds = DPLSet(
            conf["dataloader"],
            conf["datasets"],
            (8, 2),
            self.misc.get("augment", None),
            {"GPU": "cuda", "CPU": "cpu"}[self._device_type],
        )
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
