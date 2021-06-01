import os

from spectrainer import ToyNetTrainer
from common.trainer import getTrainComponents
from baseline.resnet import Resx2, SimRes
from baseline.densenet import Densex2, SimDense


def post_script(post):
    if post and os.path.exists(post):
        with open(post) as f:
            # use exec here since
            # 1. `import` will excute the script at once
            # 2. you can modify the script when training
            exec(compile(f.read(), post, exec))


def main():
    trainer, net, data = getTrainComponents(ToyNetTrainer, Resx2, "./config/simres.yml")
    trainer.fit(net, datamodule=data)

    post = trainer.paths.get("post_training", "")
    post_script(post)


if __name__ == "__main__":
    main()
