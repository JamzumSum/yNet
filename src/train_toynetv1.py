import os

from common.trainer import getTrainComponents
from spectrainer import ToyNetTrainer
from toynet.toynetv1 import ToyNetV1


def post_script(post):
    if post and os.path.exists(post):
        with open(post) as f:
            # use exec here since
            # 1. `import` will excute the script at once
            # 2. you can modify the script when training
            exec(compile(f.read(), post, exec))


def main():
    # for capability when spawn start
    trainer, net, data = getTrainComponents(
        ToyNetTrainer, ToyNetV1, "./config/toynetv1.yml"
    )
    trainer.fit(net, datamodule=data)

    post = trainer.paths.get("post_training", "")
    post_script(post)


if __name__ == "__main__":
    main()
