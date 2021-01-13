'''
A universial trainer ready to be inherited.

* author: JamzumSum
* create: 2021-1-12
'''
import os

import torch
from tensorboardX import SummaryWriter

class Trainer:
    cur_epoch = 0
    total_batch = 0

    def __init__(self, Net: torch.nn.Module, conf: dict):
        self.cls_name = Net.__name__
        self.conf = conf

        self.paths = conf.get('paths', {})
        self.training = conf.get('training', {})
        self.op_conf = conf.get('optimizer', {})
        self.dataloader = conf.get('dataloader', {})
        self.model_conf = conf.get('model', {})

        try:
            if not self.load(self.training.get('name', '')):
                self.net = Net(**self.model_conf)
        except ValueError as e:
            print(e)
            print('Trainer stopped.')
            return

        self.board = SummaryWriter(self.log_dir)

    def __del__(self):
        if hasattr(self, 'board') and self.board: self.board.close()

    @property 
    def model_dir(self): 
        return self.paths.get('model_dir', os.path.join('model', self.cls_name))
    @property
    def log_dir(self):
        return self.paths.get('log_dir', os.path.join('log', self.cls_name))

    @property
    def max_epoch(self):
        return self.training.get('max_epoch', 1)

    def save(self, name, score=None):
        vconf = {
            'cur_epoch': self.cur_epoch, 
            'total_batch': self.total_batch
        }

        if not os.path.exists(self.model_dir): os.mkdir(self.model_dir)
        torch.save(
            (self.net, self.conf, vconf, score), 
            os.path.join(self.paths.get(), name + '.pt')
        )
    
    def load(self, name):
        if not self.training.get('continue', True): 
            if name: raise ValueError("You've set name=%s, but continue training is off." % name)
            return
        path = os.path.join(self.model_dir, name + '.pt')
        if not os.path.exists(path): 
            print('%s not exist. Start new training.' % path)
            return

        self.net, newconf, vonf, score = self.torch.load(path)

        print(score)
        self.solveConflict(newconf)

        for k, v in vonf.items(): setattr(self, k, v)
        return True

    def solveConflict(self, newConf):
        if self.conf.get('model', None) != newConf.get('model', None): 
            raise ValueError('Model args have been changed')
        self.conf = newConf     # TODO

    
    def logSummary(self, summary: dict, step=None):
        self.board.add_scalars('training', summary, step)