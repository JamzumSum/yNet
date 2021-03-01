import os

from spectrainer import ToyNetTrainer
from common.trainer import Trainer, getTrainComponents
from baseline.resnet import Resx2
from common.trainer import getConfig

trainer, net, data = getTrainComponents(
    ToyNetTrainer, Resx2, "./config/resnet.yml"
)
trainer.fit(net, datamodule=data)

post = trainer.paths.get("post_training", "")
if post and os.path.exists(post):
    with open(post) as f:
        # use exec here since
        # 1. `import` will excute the script at once
        # 2. you can modify the script when training
        exec(compile(f.read(), post, exec))
