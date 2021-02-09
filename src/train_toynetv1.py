import os

from dataset import classSpecSplit, CachedDatasetGroup
from spectrainer import ToyNetTrainer
from toynet.toynetv1 import ToyNetV1
from utils.utils import getConfig

ta, va = classSpecSplit(CachedDatasetGroup('./data/BIRADs/ourset.pt'), 8, 2)
print('trainset A distribution:', ta.distribution)
# print('trainset U distribution:', tu.distribution)
print('validation A distribution:', va.distribution)
# print('validation U distribution:', vu.distribution)

trainer = ToyNetTrainer(ToyNetV1, getConfig('./config/toynetv1.yml'))
trainer.train(ta, vd=va)

post = trainer.paths.get('post_training', '')
if post and os.path.exists(post):
    with open(post) as f:
        # use exec here since
        # 1. `import` will excute the script at once
        # 2. you can modify the script when training
        exec(compile(f.read(), post, exec))
