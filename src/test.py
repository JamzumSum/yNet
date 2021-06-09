from baseline.densenet import Densex2
from baseline.resnet import Resx2, SimRes
from common.trainer import getTestComponents
from spectrainer import ToyNetTrainer
from toynet.toynetv1 import ToyNetV1


def main():
    # for capability when spawn start
    trainer, net, data = getTestComponents(
        ToyNetTrainer, Densex2, "log/tmp/densenet/version_0/hparams.yaml"
    )
    trainer.test(net, datamodule=data)


if __name__ == "__main__":
    main()
