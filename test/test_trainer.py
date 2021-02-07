import os
from unittest import TestCase

import torch
import yaml
from spectrainer import ToyNetTrainer
from toynet.toynetv1 import ToyNetV1
from utils.utils import cal_parameters, getConfig

class TrainerFunctionTest(TestCase):

    def setUp(self):
        conf = getConfig('./config/toynetv1.yml')
        self.trainer = ToyNetTrainer(ToyNetV1, conf)

    def testSave(self):
        self.trainer.save('test', 0.)
        self.assertTrue(os.path.exists(
            os.path.join(self.trainer.log_dir, 'test.pt')
        ))

    def test_cal_parameter(self):
        cal_parameters(self.trainer.net)

    def test_CM(self):
        from common.utils import ConfusionMatrix
        cm = ConfusionMatrix(4)
        cm.add(2 * torch.ones(4).int(), 2 * torch.ones(4).int())
        self.trainer.prepareBoard()
        self.trainer.board.add_image('confusionmat', cm.mat(), dataformats='HW')
