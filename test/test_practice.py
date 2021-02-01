from unittest import TestCase
import torch
import torch.nn.functional as F

class BPTest(TestCase):
    def testMax(self):
        v = torch.randn(1, 4, 4) / 100
        v[:, :2, :2] = 0.8
        p = v.clone().detach_().requires_grad_(True)
        op = torch.optim.SGD((p,), lr=1)
        pooling = torch.nn.AvgPool2d(2)

        patch = pooling(p)
        m = patch.max()
        loss = (m - 1) ** 2
        loss.backward()
        op.step()
        print(v)
        print(p)

    def testAvgPool(self):
        v = torch.randn(1, 8, 8)
        pooling = torch.nn.AvgPool2d(5)

        self.assertTrue(torch.all(pooling(v) == pooling(v[:, :5, :5])))
