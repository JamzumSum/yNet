import yaml

from dataloader import trainValidSplit
from spectrainer import ToyNetTrainer
from toynet.toynetv1 import ToyNetV1

(ta, tu), (va, vu) = trainValidSplit(8, 2)
print('trainset A distribution:', ta.distribution)
print('trainset U distribution:', tu.distribution)
print('validation A distribution:', va.distribution)
print('validation U distribution:', vu.distribution)

conf = {}
with open('./config/toynetv1.yml') as f: conf = yaml.safe_load(f)

trainer = ToyNetTrainer(ToyNetV1, conf)
trainer.train(ta, tu, va, vu)
