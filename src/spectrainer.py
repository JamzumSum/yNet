'''
A trainer for toynets.

* author: JamzumSum
* create: 2021-1-13
'''
from common.trainer import Trainer
from toynet2 import ToyNetV1, ToyNetV2
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F

class ToyNetTrainer(Trainer):
    def train(self, anno, unanno, vanno=None, vunanno=None):
        aloader = DataLoader(anno, **self.dataloader.get('annotated', {}))
        uloader = DataLoader(unanno, **self.dataloader.get('unannotated', {}))
        anno_only_epoch = self.training.get('use_unannotated_from', self.max_epoch)

        for self.cur_epoch in range(self.cur_epoch, self.max_epoch):
            for X, Ym, Yb in aloader:
                loss, summary = self.net.loss(X, Ym, Yb)
                loss.backward()
                self.total_batch += 1
                self.logSummary(summary, self.total_batch)
            if self.cur_epoch < anno_only_epoch: continue

            for X, Ym in uloader:
                loss, summary = self.net.loss(X, Ym)
                loss.backward()
                self.total_batch += 1
                self.logSummary(summary, self.total_batch)

            self.score('annotated/training', anno)                  # training score
            self.score('unannotated/training', unanno)
            if vanno: self.score('annotated/validation', vanno)     # validation score
            if vunanno: self.score('unannotated/validation', vunanno)

    def score(self, caption, dataset):
        '''
        Calculate accuracy of the given dataset.
        '''
        sloader = DataLoader(dataset, **self.dataloader.get('scoring', {}))
        d = next(iter(sloader))
        _, _, Mp, Bp = self.net(d[0])
        macc = F.l1_loss(Mp.squeeze(-1), d[1])
        self.board.add_scalar(caption + '/malignant accuracy', macc, self.cur_epoch)

        if len(d) == 3:
            bacc = torch.argmax(Bp, -1) == d[2]
            bacc = bacc.float().mean()
            self.board.add_scalar(caption + '/BIRADs accuracy', bacc, self.cur_epoch)