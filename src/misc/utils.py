import yaml
import os

from .dict import deep_update

def getConfig(path):
    with open(path) as f: 
        d = yaml.safe_load(f)
        if 'import' in d: 
            path = os.path.join(os.path.dirname(path), d.pop('import'))
            imd = getConfig(path)
            return deep_update(imd.copy(), d)
        else: return d