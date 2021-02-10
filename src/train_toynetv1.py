import os

from data.dataset import classSpecSplit, CachedDatasetGroup, DistributedConcatSet
from data.augment import ElasticAugmentSet
from spectrainer import ToyNetTrainer
from toynet.toynetv1 import ToyNetV1
from utils.utils import getConfig

td, vd = classSpecSplit(CachedDatasetGroup('./data/BIRADs/ourset.pt'), 8, 2)
td = DistributedConcatSet(
    datasets=[td, ElasticAugmentSet(td, 'Ym', aim_size=440)], 
    tag=['realset', 'elastic']
)
print('trainset A distribution:', td.distribution)
print('validation A distribution:', vd.distribution)

trainer = ToyNetTrainer(ToyNetV1, getConfig('./config/toynetv1.yml'))
trainer.train(td, vd)

post = trainer.paths.get('post_training', '')
if post and os.path.exists(post):
    with open(post) as f:
        # use exec here since
        # 1. `import` will excute the script at once
        # 2. you can modify the script when training
        exec(compile(f.read(), post, exec))
