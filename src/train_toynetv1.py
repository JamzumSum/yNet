import os

from spectrainer import ToyNetTrainer
from common.trainer import Trainer
from toynet.toynetv1 import ToyNetV1
from misc.utils import getConfig

conf = getConfig("./config/toynetv1.yml")
trainer = Trainer(ToyNetTrainer, ToyNetV1, conf)
trainer.fit()

post = trainer.paths.get("post_training", "")
if post and os.path.exists(post):
    with open(post) as f:
        # use exec here since
        # 1. `import` will excute the script at once
        # 2. you can modify the script when training
        exec(compile(f.read(), post, exec))
