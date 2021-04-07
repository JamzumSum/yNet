from collections import defaultdict
import torch


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


class SubOptimizer(torch.optim.Optimizer):
    __slots__ = ('__o', '__i')

    def __init__(self, optimizer, idx):
        self.__o = optimizer
        self.__i = idx

    def __getattribute__(self, name: str):
        if name == 'param_groups': return [self.__o.param_groups[self.__i]]
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return getattr(self.__o, name)

    def __instancecheck__(self, instance) -> bool:
        return isinstance(instance, type(self.__o))


def getBranchScheduler(cls: type, optimizer, arglist: list[dict], extra=None):
    class BranchScheduler(cls):
        def __init__(self, optimizer, arglist: list[dict], extra: dict = None):
            self.optimizer = optimizer
            if isinstance(arglist, (list, tuple)) and len(arglist) == 1:
                arglist = arglist * len(optimizer.parameter_groups)

            if extra is None: extra = {}
            extra = {k: v for k, v in extra.items() if k in get_arg_name(cls.__init__)}
            
            self.sub = [
                None
                if arg is None else cls(SubOptimizer(optimizer, i), **arg, **extra)
                for i, arg in enumerate(arglist)
            ]

        def step(self, metrics=None):
            if metrics is not None:
                ld = lambda l, i: l[i] if isinstance(l, (list, tuple)) else l

            for i, sg in enumerate(self.sub):
                if not sg: continue
                if metrics is None: sg.step()
                else: sg.step(ld(metrics, i))

        def setepoch(self, epoch):
            self.last_epoch = epoch
            for sg in self.sub:
                if sg: sg.last_epoch = epoch

    return BranchScheduler(optimizer, arglist, extra)


def get_arg_name(func) -> list:
    return func.__code__.co_varnames[:func.__code__.co_argcount]


def get_arg_default(func, arg: str):
    argls = get_arg_name(func)
    if arg in argls:
        return func.__defaults__[argls.index(arg) - len(argls)]
    else:
        raise ValueError(arg)


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


def split_upto_decay(need_decay: list, paramdic: dict, weight_decay: dict):
    for k in paramdic:
        paramdic[k] = list(paramdic[k])
    for branch, param in paramdic.copy().items():
        if weight_decay[branch]:
            paramdic[branch +
                     "_no_decay"] = [i for i in param if id(i) not in need_decay]
            paramdic[branch] = [i for i in param if id(i) in need_decay]
        else:
            paramdic[branch] = param
    return paramdic


def get_need_decay(module_iter, decay_weight_ge: defaultdict = None):
    if decay_weight_ge is None:
        decay_weight_ge = defaultdict(
            lambda: lambda m: [],
            {
                torch.nn.Conv2d: lambda m: [id(m.weight)],
                torch.nn.Linear: lambda m: [id(m.weight)]
            },
        )
    need_decay = (decay_weight_ge[type(m)](m) for m in module_iter)
    return sum(need_decay, [])