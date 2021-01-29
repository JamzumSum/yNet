import torch
from dataloader import Annotated, Unannotated

def brightness(dataset):
    X = dataset.tensors[0]
    X = X.mean(dim=(1, 2, 3))
    print('mean', X.mean())
    print('std', X.std())

print('annotated:')
brightness(Annotated())
print('unannotated:')
brightness(Unannotated())