import os

from spectrainer import ToyNetTrainer
from common.trainer import Trainer, getTestComponents
from toynet.toynetv1 import ToyNetV1
from common.trainer import getConfig


def main():
    # for capability when spawn start
    trainer, net, data = getTestComponents(
        ToyNetTrainer, ToyNetV1, "./config/toynetv1.yml"
    )
    trainer.test(net, datamodule=data)


if __name__ == "__main__":
    main()
