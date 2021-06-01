from typing import Union
from omegaconf import OmegaConf, DictConfig

DictLike = Union[dict, DictConfig]


def shallow_update(default: DictLike, update: DictLike, copy=False):
    if default is None or update is None: return

    if isinstance(default, DictConfig):
        if (ro := OmegaConf.is_readonly(default)):
            default = default.copy()
            OmegaConf.set_readonly(default, False)
        default.update(update)
        OmegaConf.set_readonly(default, ro)
    else:
        if copy: default = default.copy()
        default.update(update)
    return default


def deep_update(default, update):
    f = {dict: dict.items, list: enumerate}[type(update)]
    for k, v in f(update):
        if k in default:
            if isinstance(v, dict) and isinstance(default[k], dict):
                default[k] = deep_update(default[k], v)
            elif isinstance(v, list) and isinstance(default[k], list):
                default[k] = deep_update(default[k], v)
            else:
                default[k] = v
        else:
            default[k] = v
    return default
