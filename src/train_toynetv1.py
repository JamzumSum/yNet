import yaml

from dataloader import trainValidSplit, count
from spectrainer import ToyNetTrainer
from toynet.toynetv1 import ToyNetV1

(ta, tu), (va, vu) = trainValidSplit(8, 2)
print('trainset A distribution:', count(ta.tensors[-1]))
print('trainset U distribution:', count(tu.tensors[-1]))
print('validation A distribution:', count(va.tensors[-1]))
print('validation U distribution:', count(vu.tensors[-1]))

conf = {}
with open('./config/toynetv1.yml') as f: conf = yaml.safe_load(f)

trainer = ToyNetTrainer(ToyNetV1, conf)
trainer.train(ta, tu, va, vu)
