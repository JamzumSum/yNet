class CoefficientScheduler:
    __slots__ = "_fs", "_varmap", "_var", "__op__", "__curmap"

    def __init__(self, conf: dict, varmap: dict):
        self._fs = conf
        self._varmap = varmap
        self._var = {}
        self.__op__ = {}
        exec("from math import *", None, self.__op__)

    def update(self, **var):
        self._var.update(var)
        self.__curmap = {v: self._var[k] for k, v in self._varmap.items() if k in self._var}

    def get(self, varname, default: str = None) -> float:
        if varname not in self._fs and default is None:
            raise KeyError(varname)
        f = self._fs.get(varname, default)
        if isinstance(f, str):
            # Safety warning: eval
            r = eval(f, self.__op__, self.__curmap)
            assert isinstance(r, (float, int))
            return r
        elif isinstance(f, (float, int)):
            return f

    def __getitem__(self, varname):
        return self.get(varname)

    def __getattribute__(self, name):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return self._var[name]

