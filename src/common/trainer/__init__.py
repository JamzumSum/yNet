"""
A universial trainer ready to be inherited.

* author: JamzumSum
* create: 2021-1-12
"""
import os
from datetime import date
from warnings import warn

import pytorch_lightning as pl
import torch
from data.plsupport import DPLSet
from misc import CheckpointSupport, CoefficientScheduler
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn

from .fsm import FSMBase
from .richbar import RichProgressBar
from .testlogger import TestLogger

DATE = date.today().strftime("%m%d")


class Trainer(pl.Trainer):
    def __init__(
        self, misc: DictConfig, paths: dict, flag: DictConfig, *, logger_stage=None
    ):
        self.misc = misc
        self.paths = paths

        tb = self._getLogger(logger_stage)
        checkpoint_callback = self._getCheckpointCallback(
            tb.version or self.version, logger_stage
        )
        pl.Trainer.__init__(
            self,
            callbacks=[checkpoint_callback, RichProgressBar()],
            default_root_dir=checkpoint_callback.dirpath,
            logger=tb,
            num_sanity_val_steps=0,
            terminate_on_nan=True,
            log_every_n_steps=10,
            auto_select_gpus=bool(flag.gpus),
            **flag
        )

    @property
    def name(self):
        return self.paths.name

    @property
    def version(self) -> str:
        return self.paths.version

    def _getLogger(self, stage=None):
        log_dir = self.paths.get("log_dir", f'log/{DATE}')
        if stage == 'test':
            return TestLogger(log_dir, self.name, self.version)

        return TensorBoardLogger(
            log_dir,
            self.name,
            self.version,
            log_graph=self.misc.get('log_graph', True),
            default_hp_metric=False
        )

    def _getCheckpointCallback(self, version: int = None, stage=None):
        model_dir = self.paths.get("model_dir", f'model/{DATE}')
        model_dir = os.path.join(model_dir, self.name)
        if version or version == 0:
            version = f"version_{version}" if isinstance(version, int) else version
            model_dir = os.path.join(model_dir, version)

        exist = os.path.exists(model_dir) and os.listdir(model_dir)
        if stage == 'test' and not exist:
            raise FileNotFoundError(model_dir)
        elif stage is None and exist:
            raise FileExistsError(model_dir)

        self.model_dir = model_dir

        checkpoint_callback = ModelCheckpoint(
            monitor="err/B-M/validation",
            dirpath=model_dir,
            filename="best",
            save_last=True,
            mode="min",
        )
        checkpoint_callback.FILE_EXTENSION = ".pth"
        checkpoint_callback.best_model_path = model_dir
        checkpoint_callback.CHECKPOINT_NAME_LAST = "latest"
        return checkpoint_callback


def getConfig(path: str) -> DictConfig:
    d = OmegaConf.load(path)
    if "import" not in d: return d

    imp = d.pop("import")
    if isinstance(imp, str):
        imp = [imp]
    elif not isinstance(imp, (list, ListConfig)):
        raise TypeError(imp)
    path = [os.path.join(os.path.dirname(path), i) for i in imp]
    imd = [getConfig(p) for p in path]
    return OmegaConf.merge(*imd, d)


def getConfigWithCLI(path: str) -> DictConfig:
    return OmegaConf.merge(getConfig(path), OmegaConf.from_cli())


def formatConf(conf: DictConfig):
    if conf.paths.model_dir:
        conf.paths.model_dir = conf.paths.model_dir.format(date=DATE)
    if conf.paths.log_dir:
        conf.paths.log_dir = conf.paths.log_dir.format(date=DATE)
    OmegaConf.set_readonly(conf, True)


def gpus2device(gpus):
    if gpus == 0:
        return torch.device('cpu')
    elif isinstance(gpus, int):
        return torch.device(f'cuda:{gpus - 1}')
    elif isinstance(gpus, (list, ListConfig)):
        return torch.device(f'cuda:{gpus[-1]}')
    else:
        raise TypeError(gpus)


def getTrainComponents(FSM: type[FSMBase], Net: type[nn.Module], conf_path: str):
    """Given the config in all, construct 3 key components for training, 
    which are all initialed with minimal config sections.

    Args:
        FSM (type[FSMBase]): [description]
        Net (type[Module]): [description]
        conf_path (str): [description]

    Returns:
        Trainer,
        LightningModule,
        LightningDataModule
    """
    conf = getConfigWithCLI(conf_path)
    formatConf(conf)

    cosg = CoefficientScheduler(conf.coefficients, {"piter": "x", "max_epochs": "M"})
    cosg.update(piter=0)
    net = Net(cmgr=cosg, cps=CheckpointSupport(conf.misc.memory_trade), **conf.model)
    fsm = FSM(net, cosg, conf)

    trainer = Trainer(conf.misc, conf.paths, conf.flag)

    if conf.misc.get("continue", True):
        model_dir = trainer.default_root_dir
        name = conf.misc.get("load_from", "latest") + ".pth"
        path = os.path.join(model_dir, name)
        fsm = fsm.load_from_checkpoint(path, net=net, cmgr=cosg, conf=conf)
    else:
        fsm.seed = conf.misc.seed or int(
            torch.empty((), dtype=torch.int64).random_(4294967295).item()
        )

    pl.utilities.seed.seed_everything(fsm.seed)

    device = gpus2device(trainer.gpus)
    datamodule = DPLSet(
        conf.dataloader,
        conf.data.datasets,
        tv=conf.data.get('split', (8, 2)),
        mask_prob=conf.data.get('mask_usage', 1.),
        aug_aimsize=conf.misc.augment,
        device=device,
    )
    datamodule.prepare_data()
    datamodule.setup()
    fsm.score_caption = datamodule.score_caption

    return trainer, fsm, datamodule


def getTestComponents(FSM: type[FSMBase], Net: type[nn.Module], conf_path: str):
    conf = getConfigWithCLI(conf_path)
    formatConf(conf)

    cosg = CoefficientScheduler(conf.coefficients, {"piter": "x", "max_epochs": "M"})
    cosg.update(piter=0)
    net = Net(cmgr=cosg, cps=CheckpointSupport(conf.misc.memory_trade), **conf.model)
    fsm = FSM(net, cosg, conf)

    trainer = Trainer(conf.misc, conf.paths, conf.flag, logger_stage='test')
    trainer.logger.add.append(OmegaConf.to_container(conf))

    model_dir = trainer.model_dir
    path = os.path.join(model_dir, "best.pth")
    if not os.path.exists(path):
        warn(f'{path} does not exist. Will use latest.pth instead.')
        path = os.path.join(model_dir, "latest.pth")
        assert os.path.exists(path), f'no checkpoint saved: {model_dir}'

    fsm = fsm.load_from_checkpoint(path, net=net, cmgr=cosg, conf=conf)
    pl.utilities.seed.seed_everything(fsm.seed)

    device = gpus2device(trainer.gpus)

    datamodule = DPLSet(conf.dataloader, conf.data.datasets, device=device)
    datamodule.prepare_data()
    datamodule.setup()

    return trainer, fsm, datamodule
