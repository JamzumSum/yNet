import yaml

from dataloader import trainValidSplit
from spectrainer import ToyNetTrainer
from toynet.toynetv1 import ToyNetV1D
from utils import soft_update
import os

(ta, tu), (va, vu) = trainValidSplit(8, 2)
print('trainset A distribution:', ta.distribution)
print('trainset U distribution:', tu.distribution)
print('validation A distribution:', va.distribution)
print('validation U distribution:', vu.distribution)

def getConfig(path):
    with open(path) as f: 
        d = yaml.safe_load(f)
        if 'import' in d: 
            path = os.path.join(os.path.dirname(path), d.pop('import'))
            imd = getConfig(path)
            return soft_update(imd.copy(), d)
        else: return d

trainer = ToyNetTrainer(ToyNetV1D, getConfig('./config/toynetv1.yml'))
trainer.train(ta, tu, va, vu)
