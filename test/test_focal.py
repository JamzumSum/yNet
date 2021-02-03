from unittest import TestCase

import torch
import torch.nn.functional as F
from common.loss import focalBCE

class FocalTest(TestCase):
    K = 6
    def testBCE(self):
        weight = torch.Tensor([.1, .1, .1, .2, .2, .3])

        P = torch.randn(3, self.K)
        Y = torch.randint(self.K, (3,))
        mask = F.one_hot(Y, num_classes=self.K).float()
        sP = P.sigmoid()
        man = mask * sP.log() + (1 - mask) * (1 - sP).log()
        man = -man * weight
        tim = F.binary_cross_entropy(sP, mask, reduction='none', weight=weight)
        self.assertTrue(torch.all(man == tim))

    def testBCEwithLogits(self):
        weight = torch.Tensor([.1, .1, .1, .2, .2, .3])

        P = torch.randn(3, self.K)
        Y = torch.randint(self.K, (3,))
        mask = F.one_hot(Y, num_classes=self.K).float()
        sP = P.sigmoid()
        man = mask * sP.log() + (1 - mask) * (1 - sP).log()
        man = -man * weight
        tim = F.binary_cross_entropy_with_logits(P, mask, reduction='none', weight=weight)
        self.assertTrue(torch.all(torch.abs(man - tim) / torch.max(man, tim) < 1e-5))

    def testFocalBCEBP(self):
        P = torch.empty(4, 2)
        P[:, 0] = .2
        P[:, 1] = .8
        P = P.requires_grad_(True)

        Y = torch.zeros(4).long()
        loss = focalBCE(P, Y, K=2)

        op = torch.optim.SGD((P,), lr=1)
        loss.backward()
        op.step()
        print()
        print(P)
        self.assertTrue(torch.all(P[:, 0] > .2))
        self.assertTrue(torch.all(P[:, 1] < .8))