from baseline.resnet import SimRes
from common.trainer import getTestComponents
from spectrainer import ToyNetTrainer
from toynet.toynetv1 import ToyNetV1


def main():
    # for capability when spawn start
    trainer, net, data = getTestComponents(
        ToyNetTrainer, SimRes, "./config/simres.yml"
    )
    trainer.test(net, datamodule=data)


if __name__ == "__main__":
    main()
