"""
A universial trainer ready to be inherited.

* author: JamzumSum
* create: 2021-1-12
"""
import os
from datetime import date

import pytorch_lightning as pl
import torch
from data.plsupport import DPLSet
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from .richbar import RichProgressBar


class Trainer(pl.Trainer):
    def __init__(self, misc, paths, flag):
        self.misc = misc
        self.paths = paths
        self.name = paths.get("name", "default")

        checkpoint_callback = self._getCheckpointCallback()
        pl.Trainer.__init__(
            self,
            callbacks=[checkpoint_callback, RichProgressBar()],
            default_root_dir=checkpoint_callback.dirpath,
            logger=self._getTensorBoardLogger(),
            **flag
        )

    def _getTensorBoardLogger(self):
        log_dir = self.paths.get("log_dir", os.path.join("log", self.name)).format(
            date=date.today().strftime("%m%d")
        )
        return TensorBoardLogger(
            log_dir, self.name, log_graph=True, default_hp_metric=False
        )

    def _getCheckpointCallback(self):
        model_dir = self.paths.get(
            "model_dir", os.path.join("model", self.name)
        ).format(date=date.today().strftime("%m%d"))
        checkpoint_callback = ModelCheckpoint(
            monitor="err/B-M/validation",
            dirpath=model_dir,
            filename="best",
            save_last=True,
            mode="min",
        )
        checkpoint_callback.FILE_EXTENSION = ".pt"
        checkpoint_callback.best_model_path = model_dir
        checkpoint_callback.CHECKPOINT_NAME_LAST = "latest"
        return checkpoint_callback


def getConfig(path) -> DictConfig:
    d = OmegaConf.load(path)
    if "import" in d:
        path = os.path.join(os.path.dirname(path), d.pop("import"))
        imd = getConfig(path)
        return OmegaConf.merge(imd, d)
    else:
        return d


def getTrainComponents(FSM, Net, conf_path):
    """
    Given the config in all, construct 3 key components for training, 
    which are all initialed with minimal config sections.

    return:
        Trainer
        LightningModule
        LightningDataModule
    """
    conf = getConfig(conf_path)
    trainer = Trainer(conf.misc, conf.paths, conf.flag)

    datamodule = DPLSet(
        conf.dataloader,
        conf.datasets,
        'Ym',
        (8, 2),
        conf.misc.get("augment", None),
        "cpu" if not trainer.gpus else "cuda",
    )
    datamodule.prepare_data()
    datamodule.setup()

    kwargs = dict(
        Net=Net,
        model_conf=conf.model,
        coeff_conf=conf.coefficients,
        misc=conf.misc,
        op_conf=conf.optimizer,
        sg_conf=conf.scheduler,
        branch_conf=conf.branch,
    )
    net = FSM(**kwargs)

    if conf.misc.get("continue", True):
        model_dir = trainer.default_root_dir
        name = conf.misc.get("load_from", "latest") + ".pt"
        path = os.path.join(model_dir, name)
        net = net.load_from_checkpoint(path, **kwargs)
    else:
        net.seed = int(torch.empty((), dtype=torch.int64).random_(4294967295).item())

    net.score_caption = datamodule.score_caption
    pl.utilities.seed.seed_everything(net.seed)

    return trainer, net, datamodule
