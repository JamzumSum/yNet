from unittest import TestCase

import torch
import torch.nn as nn
from common.loss import (F, SemiHardTripletLoss, focal_smooth_bce,
                         focal_smooth_ce)


def teq(t1, t2, eps=1e-5):
    return torch.all(torch.abs(t1 - t2) / torch.max(t1, t2) < eps)


class MathVal(TestCase):
    K = 6

    def BCE(self):
        weight = torch.Tensor([0.1, 0.1, 0.1, 0.2, 0.2, 0.3])

        P = torch.randn(3, self.K)
        Y = torch.randint(self.K, (3,))
        mask = F.one_hot(Y, num_classes=self.K).float()
        sP = P.sigmoid()
        logpt = mask * sP.log() + (1 - mask) * (1 - sP).log()
        logpt = -logpt * weight
        tim = F.binary_cross_entropy(sP, mask, reduction="none", weight=weight)
        self.assertTrue(torch.all(logpt == tim))

    def BCEwithLogits(self):
        weight = torch.Tensor([0.1, 0.1, 0.1, 0.2, 0.2, 0.3])

        P = torch.randn(3, self.K)
        Y = torch.randint(self.K, (3,))
        mask = F.one_hot(Y, num_classes=self.K).float()
        sP = P.sigmoid()
        man = mask * sP.log() + (1 - mask) * (1 - sP).log()
        man = -man * weight
        tim = F.binary_cross_entropy_with_logits(
            P, mask, reduction="none", weight=weight
        )
        self.assertTrue(teq(man, tim))

    def testCE(self):
        weight = torch.Tensor([0.1, 0.1, 0.1, 0.2, 0.2, 0.3])
        P = torch.randn(3, self.K)
        Y = torch.randint(self.K, (3,))
        mask = F.one_hot(Y, num_classes=self.K).float()
        man = mask * P.log_softmax(dim=1)
        man = -man * weight
        man = man.sum(dim=1)
        tim = F.cross_entropy(P, Y, weight=weight, reduction='none')
        fim = focal_smooth_ce(P.softmax(1), Y, weight=weight, reduction='none', gamma=0)
        self.assertTrue(teq(man, tim))

class FocalBCETest(TestCase):
    K = 6

    def testvalue(self):
        P = torch.rand(4, 2)
        Y = torch.randint(2, (4,)).long()
        lossf = focal_smooth_bce(P, Y, gamma=0, weight=torch.Tensor([.4, .6]))
        lossb = F.binary_cross_entropy(P, F.one_hot(Y).float(), weight=torch.Tensor([.4, .6]))
        print(lossf, lossb)
        self.assertTrue(teq(lossf, lossb))

    def testBP(self):
        P = torch.empty(4, 2)
        P[:, 0], P[:, 1] = 0.2, 0.8
        P = P.requires_grad_(True)

        Y = torch.zeros(4).long()
        loss = focal_smooth_bce(P, Y, smooth=0.1)

        op = torch.optim.SGD((P,), lr=1)
        loss.backward()
        op.step()
        self.assertTrue(torch.all(P[:, 0] > 0.2))
        self.assertTrue(torch.all(P[:, 1] < 0.8))


class FocalCETest(TestCase):
    def testvalue(self):
        P = torch.rand(4, 2)
        Y = torch.randint(2, (4,)).long()
        lossf = focal_smooth_ce(P.softmax(dim=1), Y, gamma=0, weight=torch.Tensor([.4, .6]))
        lossc = F.cross_entropy(P, Y, weight=torch.Tensor([.4, .6]))
        print(lossf, lossc)
        self.assertTrue(teq(lossf, lossc))

    def testBP(self):
        P = torch.empty(4, 2)
        P[:, 0], P[:, 1] = 0.2, 0.8
        P = P.requires_grad_(True)

        Y = torch.zeros(4).long()
        loss = focal_smooth_ce(P, Y, smooth=0)

        op = torch.optim.SGD((P,), lr=1)
        loss.backward()
        op.step()
        self.assertTrue(torch.all(P[:, 0] > 0.2))
        self.assertTrue(torch.all(P[:, 1] < 0.8))


class TripletTest(TestCase):
    def testTriplet2(self):
        c = torch.randn(8, 2, requires_grad=True)
        Y = torch.LongTensor([0] * 4 + [1] * 4)
        crit = SemiHardTripletLoss(0.3)
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
        print("loss =", loss)
        print(c)

