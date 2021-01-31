from unittest import TestCase

import torch
import torch.nn.functional as F


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
        man = -man
        tim = F.binary_cross_entropy_with_logits(P, mask, reduction='none')
        self.assertTrue(torch.all(torch.abs(man - tim) / torch.max(man, tim) < 1e-5))