from spectrainer import ToyNetTrainer, ToyNetV1
from dataloader import trainValidSplit
import yaml

(ta, tu), (va, vu) = trainValidSplit(8, 2)
print('annotated train set:', ta.shape[0])
print('unannotated train set:', tu.shape[0])
print('annotated validation set:', va.shape[0])
print('unannotated validation set:', vu.shape[0])

conf = {}
with open('./config/toynetv1.yml') as f: conf = yaml.safe_load(f)

trainer = ToyNetTrainer(ToyNetV1, conf)
trainer.train(ta, tu, va, vu)
