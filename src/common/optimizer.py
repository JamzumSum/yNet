import torch
from collections import defaultdict


class DirectLR(torch.optim.lr_scheduler._LRScheduler):
    @property
    def lr(self):
        return self.get_last_lr()

    def set_lr(self, lr):
        if not isinstance(lr, list) and not isinstance(lr, tuple):
            self._lr = [lr] * len(self.optimizer.param_groups)
        else:
            if len(lr) != len(self.optimizer.param_groups):
                raise ValueError(
                    "Expected {} lr, but got {}".format(
                        len(self.optimizer.param_groups), len(lr)
                    )
                )
            self._lr = list(lr)

    def get_lr(self):
        return self._lr


class _ReduceLROnPlateauSub(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self, idx, *args, **argv):
        self.idx = idx
        torch.optim.lr_scheduler.ReduceLROnPlateau.__init__(self, *args, **argv)

    def _reduce_lr(self, epoch):
        param_group = self.optimizer.param_groups[self.idx]
        old_lr = float(param_group["lr"])
        new_lr = max(old_lr * self.factor, self.min_lrs[self.idx])
        if old_lr - new_lr > self.eps:
            param_group["lr"] = new_lr
            if self.verbose:
                print(
                    "Epoch {:5d}: reducing learning rate"
                    " of group {} to {:.4e}.".format(epoch, self.idx, new_lr)
                )


class ReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self, optimizer, arglist):
        default = {
            "mode": "min",
            "factor": 0.1,
            "patience": 10,
            "threshold": 1e-4,
            "threshold_mode": "rel",
            "cooldown": 0,
            "min_lr": 0,
            "eps": 1e-8,
            "verbose": False,
        }
        self.optimizer = optimizer
        if isinstance(arglist, (list, tuple)) and len(arglist) == 1:
            arglist = arglist * len(optimizer.parameter_groups)

        ld = lambda d, i: d.get(i, default[i])
        self.sub = [
            None
            if arg is None
            else _ReduceLROnPlateauSub(
                i,
                optimizer,
                mode=ld(arg, "mode"),
                factor=ld(arg, "factor"),
                patience=ld(arg, "patience"),
                threshold=ld(arg, "threshold"),
                threshold_mode=ld(arg, "threshold_mode"),
                cooldown=ld(arg, "cooldown"),
                min_lr=ld(arg, "min_lr"),
                eps=ld(arg, "eps"),
                verbose=ld(arg, "verbose"),
            )
            for i, arg in enumerate(arglist)
        ]

    def step(self, metrics):
        ld = lambda l, i: l[i] if isinstance(l, (list, tuple)) else l

        for i, sg in enumerate(self.sub):
            if sg:
                sg.step(ld(metrics, i))

    def setepoch(self, epoch):
        self.last_epoch = epoch
        for sg in self.sub:
            if sg:
                sg.last_epoch = epoch


def get_arg_default(func, arg: str):
    argls: list = func.__code__.co_varnames[: func.__code__.co_argcount]
    if arg in argls:
        return func.__defaults__[argls.index(arg) - len(argls)]


def no_decay(weight_decay: dict, paramdic: dict, op_arg, default_lr):
    paramdic = paramdic.copy()
    param_group_key = []
    for k, v in op_arg.items():
        for i, s in enumerate(("", "_no_decay")):
            if i & 1 and not weight_decay[k]:
                continue
            param_group_key.append(k)
            sk = k + s
            paramdic[sk] = {"params": paramdic[sk]}
            paramdic[sk].update(v)
            paramdic[sk]["initial_lr"] = v.get("lr", default_lr)
            if i & 1:
                paramdic[sk]["weight_decay"] = 0.0
    return paramdic, param_group_key
