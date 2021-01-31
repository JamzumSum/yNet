'''
A collection for focal losses implementation. 

* author: JamzumSum
* create: 2021-1-11
'''

import torch
import torch.nn.functional as F

def focalCE(P, Y, gamma=2., *args, **argv):
    '''
    focal loss for classification. nll_loss implement
    - P: [N, K] NOTE: not softmax-ed
    - Y: [N]    NOTE: long
    - gamma: 
    '''
    pt = torch.softmax(P, dim=-1)   # [N, K]
    gms = (1 - pt) ** gamma         # [N, K]
    return F.nll_loss(gms * pt.log(), Y, *args, **argv)

def focalBCE(P, Y, gamma=2., K=-1, weight=None):
    '''
    focal loss for classification. BCELoss implement
    - P: [N, K] NOTE: not softmax-ed when K != 1
    - Y: [N]    NOTE: long
    - gamma: 
    '''
    weight = weight / weight.sum()
    # This is a 'pre-distribution' that needs softmax.
    # NOTE: softmax is troubling for both branches. 
    Y = F.one_hot(Y, num_classes=K).float()
    bce = F.binary_cross_entropy(P, Y, weight=weight, reduction='none')
    pt = torch.exp(-bce)            # [N, K]
    gms = (1 - pt) ** gamma         # [N, K]
    return (gms * bce).mean()