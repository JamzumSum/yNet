"""
A universial trainer ready to be inherited.

* author: JamzumSum
* create: 2021-1-12
"""
from datetime import date
import os

import torch
from tensorboardX import SummaryWriter
from collections import defaultdict
from rich.progress import Progress


class Trainer:
    cur_epoch = 0
    total_batch = 0
    best_mark = 1.0
    board = None

    def __init__(self, Net: torch.nn.Module, conf: dict):
        self.cls_name = Net.__name__
        self.conf = conf
        self.seed = int(torch.empty((), dtype=torch.int64).random_().item())

        self.paths = conf.get("paths", {})
        self.training = conf.get("training", {})
        op_conf: list = conf.get("optimizer", ("SGD", {}))
        self.op_name = op_conf[0]
        self.op_conf = {} if len(op_conf) == 1 else op_conf[1]
        self.dataloader = defaultdict(dict, conf.get("dataloader", {}))
        self.model_conf = conf.get("model", {})

        self.net = Net(**self.model_conf)
        self.progress = Progress(transient=True, refresh_per_second=2)

        try:
            self.load(self.training.get("load_from", "latest"))
        except ValueError as e:
            print(e)
            print("Trainer stopped.")
            return
        self.net.to(self.device)

    def __del__(self):
        self.progress.stop()
        if hasattr(self, "board") and self.board:
            self.board.close()

    @property
    def model_dir(self):
        return self.paths.get("model_dir", os.path.join("model", self.cls_name)).format(
            date=date.today().strftime("%m%d")
        )

    @property
    def log_dir(self):
        return self.paths.get("log_dir", os.path.join("log", self.cls_name)).format(
            date=date.today().strftime("%m%d")
        )

    @property
    def device(self):
        return torch.device(self.training.get("device", "cpu"))

    @property
    def max_epoch(self):
        return self.training.get("max_epoch", 1)

    @property
    def piter(self):
        return self.cur_epoch / self.max_epoch

    def save(self, name, optimizers: dict = None):
        if optimizers:
            if isinstance(optimizers, dict):
                optimizers = {k: v.state_dict() for k, v in optimizers.items()}
            elif isinstance(optimizers, torch.optim.Optimizer):
                optimizers = optimizers.state_dict()

        vconf = {
            "seed": self.seed,
            "cur_epoch": self.cur_epoch,
            "total_batch": self.total_batch,
            "best_mark": self.best_mark,
            "_op_state_dict": optimizers
        }
        
        os.makedirs(self.model_dir, exist_ok=True)
        torch.save(
            (
                self.net.state_dict(),
                self.conf,
                vconf,
            ),
            os.path.join(self.model_dir, name + ".pt"),
        )

    def load(self, name):
        if not self.training.get("continue", True):
            return
        path = os.path.join(self.model_dir, name + ".pt")
        if not os.path.exists(path):
            print("%s not exist. Start new training." % path)
            return

        state, newconf, vonf = torch.load(path)

        self.net.load_state_dict(state)
        self.solveConflict(newconf)

        for k, v in vonf.items():
            setattr(self, k, v)
        print("epoch %d, score" % self.cur_epoch, self.best_mark)
        self.cur_epoch += 1
        return True

    def solveConflict(self, newConf):
        if self.conf.get("model", None) != newConf.get("model", None):
            raise ValueError("Model args have been changed")
        self.conf = newConf  # TODO

    def prepareBoard(self):
        """fxxk tensorboard spoil the log so LAZY LOAD it"""
        if self.board is None:
            self.board = SummaryWriter(self.log_dir)

    def logSummary(self, caption, summary: dict, step=None):
        for k, v in summary.items():
            self.board.add_scalar("%s/%s" % (k, caption), v, step)

    def traceNetwork(self):
        param = self.net.named_parameters()
        for name, p in param:
            self.board.add_histogram("network/" + name, p, self.cur_epoch)
