from unittest import TestCase

import torch
from toynet.toynetv1 import ToyNetV1


class BPTest(TestCase):
    def testMultiShape(self):
        net = ToyNetV1(1, 6, [32, 64])
        d1 = torch.randn(2, 1, 512, 512)
        d2 = torch.randn(2, 1, 304, 480)
        d3 = torch.randn(2, 1, 640, 336)
        
        r = net(d1)[0]
        self.assertEqual(r.shape, (2, 2, 512, 512))
        r = net(d2)[0]
        self.assertEqual(r.shape, (2, 2, 304, 480))
        r = net(d3)[0]
        self.assertEqual(r.shape, (2, 2, 640, 336))
