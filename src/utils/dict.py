
def shallow_update(default, update, copy=False):
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
            else: default[k] = v
        else: default[k] = v
    return default
