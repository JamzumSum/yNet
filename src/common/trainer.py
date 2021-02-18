'''
A universial trainer ready to be inherited.

* author: JamzumSum
* create: 2021-1-12
'''
from datetime import date
import os

import torch
from tensorboardX import SummaryWriter
from collections import defaultdict
from rich.progress import Progress

class Trainer:
    cur_epoch = 0
    total_batch = 0
    best_mark = 1.
    board = None

    def __init__(self, Net: torch.nn.Module, conf: dict):
        self.cls_name = Net.__name__
        self.conf = conf

        self.paths = conf.get('paths', {})
        self.training = conf.get('training', {})
        self.op_conf = conf.get('optimizer', ('SGD', {}))
        self.dataloader = defaultdict(dict, conf.get('dataloader', {}))
        self.model_conf = conf.get('model', {})

        self.net = Net(**self.model_conf)
        self.progress = Progress(transient=True)

        try: self.load(self.training.get('load_from', 'latest'))
        except ValueError as e:
            print(e)
            print('Trainer stopped.')
            return
        self.net.to(self.device)

    def __del__(self):
        if hasattr(self, 'board') and self.board: self.board.close()

    @property 
    def model_dir(self): 
        return self.paths.get('model_dir', os.path.join('model', self.cls_name)).format(
            date=date.today().strftime('%m%d')
        )
    @property
    def log_dir(self):
        return self.paths.get('log_dir', os.path.join('log', self.cls_name)).format(
            date=date.today().strftime('%m%d')
        )
    
    @property
    def device(self):
        return torch.device(self.training.get('device', 'cpu'))
    @property
    def max_epoch(self):
        return self.training.get('max_epoch', 1)
    @property
    def piter(self):
        return self.cur_epoch / self.max_epoch
        
    def save(self, name, score=None):
        vconf = {
            'cur_epoch': self.cur_epoch, 
            'total_batch': self.total_batch, 
            'best_mark': score if score else self.best_mark
        }

        os.makedirs(self.model_dir, exist_ok=True)
        torch.save(
            (self.net.state_dict(), self.conf, vconf, score), 
            os.path.join(self.model_dir, name + '.pt')
        )
    
    def load(self, name):
        if not self.training.get('continue', True): return
        path = os.path.join(self.model_dir, name + '.pt')
        if not os.path.exists(path): 
            print('%s not exist. Start new training.' % path)
            return

        state, newconf, vonf, score = torch.load(path)

        print(score)
        self.net.load_state_dict(state)
        self.solveConflict(newconf)

        for k, v in vonf.items(): setattr(self, k, v)
        return True

    def solveConflict(self, newConf):
        if self.conf.get('model', None) != newConf.get('model', None): 
            raise ValueError('Model args have been changed')
        self.conf = newConf     # TODO

    def prepareBoard(self):
        '''fxxk tensorboard spoil the log so LAZY LOAD it'''
        if self.board is None: self.board = SummaryWriter(self.log_dir)
        
    def logSummary(self, caption, summary: dict, step=None):
        for k, v in summary.items():
            self.board.add_scalar('%s/%s' % (k, caption), v, step)

    def traceNetwork(self):
        param = self.net.named_parameters()
        for name, p in param:
            self.board.add_histogram('network/' + name, p, self.cur_epoch)

    def __del__(self):
        self.progress.stop()
        self.board.close()