import yaml

from dataloader import trainValidSplit
from spectrainer import ToyNetTrainer
from toynet.toynetv1 import ToyNetV1

(ta, tu), (va, vu) = trainValidSplit(8, 2)
print('annotated train set:', len(ta))
print('unannotated train set:', len(tu))
print('annotated validation set:', len(va))
print('unannotated validation set:', len(vu))

conf = {}
with open('./config/toynetv1.yml') as f: conf = yaml.safe_load(f)

trainer = ToyNetTrainer(ToyNetV1, conf)
trainer.train(ta, tu, va, vu)
