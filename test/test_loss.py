from unittest import TestCase

import torch
import torch.nn as nn
import torch.nn.functional as F
from common.loss import focal_smooth_loss, SemiHardTripletLoss
from toynet.toynetv1 import ToyNetV1


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
        loss = focal_smooth_loss(P, Y, smooth=.1)

        op = torch.optim.SGD((P,), lr=1)
        loss.backward()
        op.step()
        print()
        print(P)
        self.assertTrue(torch.all(P[:, 0] > .2))
        self.assertTrue(torch.all(P[:, 1] < .8))

class TripletTest(TestCase):
    def testAPN(self):
        c = torch.Tensor([
            [.2, .8, .9],
            [.21, .8, .89],
            [.22, .81, .91],
            [.3, .7, .8],
            [.4, .6, .65],
            [.9, .1, .2],
            [.91, .11, .21],
            [.92, .09, .19],
        ])
        Y = torch.LongTensor([0, 0, 0, 0, 1, 1, 1, 1])
        a, p, n = ToyNetV1.apn(c, Y, 2)
        self.assertTrue(torch.all(a == c[4]))
        self.assertTrue(torch.all(p == c[7]))
        self.assertTrue(torch.all(n == c[3]))
        loss = F.triplet_margin_loss(a, p, n, 1)
        print(loss)

    def testTriplet(self):
        c = torch.randn(8, 2) * 100
        c = c.requires_grad_()
        Y = torch.LongTensor([0] * 4 + [1] * 4)
        op = torch.optim.SGD((c,), lr=0.01)

        print()
        print(c)
        for epoch in range(10000):
            apn = ToyNetV1.apn(c, Y)
            if apn: 
                loss = F.triplet_margin_loss(*apn, margin=1., swap=True)
                loss.backward()
                op.step()
        print(c)

    def testTriplet2(self):
        c = torch.randn(8, 2, requires_grad=True)
        Y = torch.LongTensor([0] * 4 + [1] * 4)
        crit = SemiHardTripletLoss(.3)
        op = torch.optim.Adam((c,), lr=1e-4, weight_decay=0.01)

        print()
        print(c)
        loss = 1
        while loss > 0.4:
            loss = crit(c, Y)
            loss.backward()
            op.step()
        print(c)
        while loss > 0.3:
            loss = crit(c, Y)
            loss.backward()
            op.step()
        print(c)
        for i in range(1000):
            loss = crit(c, Y)
            loss.backward()
            op.step()
        print('loss =', loss)
        print(c)
        