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
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.cli import LightningCLI
from torch import nn
from misc import CheckpointSupport, CoefficientScheduler

from spectrainer import ToyNetTrainer

from .fsm import FSMBase
from .richbar import RichProgressBar
from .testlogger import TestLogger


class Trainer(pl.Trainer):
    def __init__(
        self, misc: DictConfig, paths: dict, flag: DictConfig, *, logger_stage=None
    ):
        self.misc = misc
        self.paths = paths
        self.name = paths.get("name", "default")

        checkpoint_callback = self._getCheckpointCallback()
        pl.Trainer.__init__(
            self,
            callbacks=[checkpoint_callback, RichProgressBar()],
            default_root_dir=checkpoint_callback.dirpath,
            logger=self._getLogger(logger_stage),
            num_sanity_val_steps=0,
            terminate_on_nan=True,
            log_every_n_steps=10,
            **flag
        )

    def _getLogger(self, stage=None):
        log_dir = self.paths.get("log_dir", os.path.join("log", self.name)).format(
            date=date.today().strftime("%m%d")
        )
        return TestLogger(
            log_dir, self.name, add={}
        ) if stage == 'test' else TensorBoardLogger(
            log_dir, self.name, log_graph=True, default_hp_metric=False
        )

    def _getCheckpointCallback(self):
        model_dir = self.paths.get("model_dir",
                                   os.path.join("model", self.name)).format(
                                       date=date.today().strftime("%m%d")
                                   )
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


class CLI(LightningCLI):
    def before_fit(self) -> None:
        self.model.score_caption = self.datamodule.score_caption

    def after_fit(self) -> None:
        post = self.config.paths.get("post_training", "")
        if post and os.path.exists(post):
            with open(post) as f:
                # use exec here since
                # 1. `import` will excute the script at once
                # 2. you can modify the script when training
                exec(compile(f.read(), post, exec))


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
    conf = getConfig(conf_path)
    kwargs = dict(
        misc=conf.misc,
        op_conf=conf.optimizer,
        sg_conf=conf.scheduler,
        branch_conf=conf.branch,
    )

    cosg = CoefficientScheduler(conf.coefficients, {"piter": "x", "max_epochs": "M"})
    net = Net(cmgr=cosg, cps=CheckpointSupport(conf.misc.memory_trade), **conf.model)
    fsm = FSM(net, cosg, **kwargs)

    trainer = Trainer(conf.misc, conf.paths, conf.flag)

    if conf.misc.get("continue", True):
        model_dir = trainer.default_root_dir
        name = conf.misc.get("load_from", "latest") + ".pt"
        path = os.path.join(model_dir, name)
        fsm = fsm.load_from_checkpoint(path, **kwargs)
    else:
        fsm.seed = int(torch.empty((), dtype=torch.int64).random_(4294967295).item())

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


def runFromCLI(FSM: type[FSMBase]):
    """Given the config in all, construct 3 key components for training, 
    which are all initialed with minimal config sections.

    Args:
        FSM (type[FSMBase]): [description]

    Returns:
        LightningCLI
    """
    cli = LightningCLI(FSM, DPLSet, trainer_class=Trainer, subclass_mode_model=True)
    return cli


def getTestComponents(FSM: type[FSMBase], Net: type[nn.Module], conf_path: str):
    conf = getConfig(conf_path)
    kwargs = dict(
        misc=conf.misc,
        op_conf=conf.optimizer,
        sg_conf=conf.scheduler,
        branch_conf=conf.branch,
    )

    cosg = CoefficientScheduler(conf.coefficients, {"piter": "x", "max_epochs": "M"})
    net = Net(cmgr=cosg, cps=CheckpointSupport(conf.misc.memory_trade), **conf.model)
    fsm = FSM(net, cosg, **kwargs)

    trainer = Trainer(conf.misc, conf.paths, conf.flag, logger_stage='test')
    model_dir = trainer.default_root_dir

    path = os.path.join(model_dir, "best.pt")
    if not os.path.exists(path):
        warn(f'{path} does not exist. Will use latest.pt instead.')
        path = os.path.join(model_dir, "latest.pt")
        assert os.path.exists(path), 'no checkpoint saved.'

    fsm = fsm.load_from_checkpoint(path, **kwargs)
    pl.utilities.seed.seed_everything(fsm.seed)

    device = gpus2device(trainer.gpus)

    datamodule = DPLSet(conf.dataloader, conf.data.datasets, device=device)
    datamodule.prepare_data()
    datamodule.setup()

    return trainer, fsm, datamodule
