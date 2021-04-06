from common.decorators import checkpointed
from omegaconf import DictConfig, OmegaConf


def isfloat(s: str):
    try:
        float(s)
        return True
    except ValueError:
        return False


class CoefficientScheduler:
    __slots__ = "_fs", "_varmap", "_var", "__op__", "__curmap"

    def __init__(self, conf: dict, varmap: dict):
        self._fs = conf
        self._varmap = varmap
        self._var = {}
        self.__op__ = {}
        self.__curmap = {}
        exec("from math import *", None, self.__op__)

    def _ge(self, name, default=None):
        if isinstance(self._fs, dict):
            return self._fs.get(name, default)
        elif isinstance(self._fs, DictConfig):
            if (r := OmegaConf.select(self._fs, name)) is None:
                return default
            else:
                return r

    def update(self, **var):
        self._var.update(var)
        self.__curmap = {
            v: self._var[k]
            for k, v in self._varmap.items() if k in self._var
        }

    def get(self, varname, default: str = None) -> float:
        if varname not in self._fs and default is None:
            raise KeyError(varname)
        f = self._ge(varname, default)
        if isinstance(f, str):
            # Safety warning: eval
            r = eval(f, self.__op__, self.__curmap)
            assert isinstance(r, (float, int))
            return r
        elif isinstance(f, (float, int)):
            return f

    def isConstant(self, varname, strict=False):
        if (f := self._ge(varname)) is None:
            if strict: raise KeyError(varname)
            else: return None
        else:
            if isinstance(f, (int, float)): return True
            elif isinstance(f, str):
                return isfloat(f)
            else:
                raise TypeError(f)

    def __getitem__(self, varname):
        return self.get(varname)

    def __getattribute__(self, name):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return self._var[name]


class CheckpointSupport:
    def __init__(self, memory_trade=False):
        self.memory_trade = memory_trade

    def __call__(self, instance):
        if not self.memory_trade:
            return instance
        instance.forward = checkpointed(instance.forward)
        return instance
