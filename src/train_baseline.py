import os

from data.dataset import classSpecSplit, CachedDatasetGroup, DistributedConcatSet
from data.augment import ElasticAugmentSet, augmentWith
from spectrainer import ToyNetTrainer
from baseline.resnet import Resx2
from utils.utils import getConfig

if __name__ == "__main__":
    td, vd = classSpecSplit(
        DistributedConcatSet([
            CachedDatasetGroup('./data/set2/set2.pt'), 
            CachedDatasetGroup('./data/set3/set3.pt')
        ], tag=['set2', 'set3']), 8, 2
    )
    # td = augmentWith(td, ElasticAugmentSet, 'Ym', 640)
    print('trainset distribution:', td.distribution)
    print('validation distribution:', vd.distribution)

    trainer = ToyNetTrainer(Resx2, getConfig('./config/resnet.yml'))
    trainer.train(td, vd)

    post = trainer.paths.get('post_training', '')
    if post and os.path.exists(post):
        with open(post) as f:
            # use exec here since
            # 1. `import` will excute the script at once
            # 2. you can modify the script when training
            exec(compile(f.read(), post, exec))
