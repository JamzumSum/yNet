import os
from unittest import TestCase

import yaml
from spectrainer import ToyNetTrainer
from toynet.toynetv1 import ToyNetV1
from utils import cal_parameters

class TrainerFunctionTest(TestCase):

    def setUp(self):
        with open('./config/toynetv1.yml') as f: 
            conf = yaml.safe_load(f)
            self.trainer = ToyNetTrainer(ToyNetV1, conf)

    def testSave(self):
        self.trainer.save('test', 0.)
        self.assertTrue(os.path.exists(
            os.path.join(self.trainer.log_dir, 'test.pt')
        ))

    def test_cal_parameter(self):
        cal_parameters(self.trainer.net)
