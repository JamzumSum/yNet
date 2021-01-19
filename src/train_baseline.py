import torch
import yaml


from dataloader import trainValidSplit
from spectrainer import ToyNetTrainer

if __name__ == "__main__":
    (ta, tu), (va, vu) = trainValidSplit(8, 2)
    print('annotated train set:', len(ta))
    print('unannotated train set:', len(tu))
    print('annotated validation set:', len(va))
    print('unannotated validation set:', len(vu))

    conf = {}
    with open('./config/resnet50.yml') as f: conf = yaml.safe_load(f)
    trainer = ToyNetTrainer(Resx2, conf)
    trainer.train(ta, tu, va, vu)
