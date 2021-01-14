'''
A trainer for toynets.

* author: JamzumSum
* create: 2021-1-13
'''
import torch
import torch.nn.functional as F
from progress.bar import Bar
from torch.utils.data import DataLoader

from common.trainer import Trainer
from common.utils import KeyboardInterruptWrapper
from toynetv2 import ToyNetV1, ToyNetV2

class ToyNetTrainer(Trainer):
    @KeyboardInterruptWrapper(lambda self: print('Training resumed. Start from epoch%d next time.' % self.cur_epoch))
    def train(self, anno, unanno, vanno=None, vunanno=None):
        aloader = DataLoader(anno, **self.dataloader.get('annotated', {}))
        uloader = DataLoader(unanno, **self.dataloader.get('unannotated', {}))
        anno_only_epoch = self.training.get('use_unannotated_from', self.max_epoch)
        try: op = self.getOptimizer()
        except ValueError as e:
            print(e)
            print('Trainer exits before iteration starts.')
            return
        self.prepareBoard()

        for self.cur_epoch in range(self.cur_epoch, self.max_epoch):
            bar = Bar('epoch %2d(A)' % self.cur_epoch, max=len(anno))
            for X, Ym, Yb in aloader:
                loss, summary = self.net.loss(X, Ym, Yb)
                loss.backward()
                op.step()
                self.total_batch += 1
                self.logSummary(summary, self.total_batch)
                bar.next()
            bar.finish()
            if self.cur_epoch < anno_only_epoch: continue

            bar = Bar('epoch %2d(U)' % self.cur_epoch, max=len(unanno))
            for X, Ym in uloader:
                loss, summary = self.net.loss(X, Ym)
                loss.backward()
                op.step()
                self.total_batch += 1
                self.logSummary(summary, self.total_batch)
                bar.next()
            bar.finish()

            ta_bacc = self.score('annotated/training', anno)        # training score
            self.score('unannotated/training', unanno)
            if vanno: self.score('annotated/validation', vanno)     # validation score
            if vunanno: self.score('unannotated/validation', vunanno)

            if ta_bacc > self.best_mark:
                self.best_mark = ta_bacc
                self.save('best', ta_bacc)
            self.save('latest', ta_bacc)

    def score(self, caption, dataset):
        '''
        Calculate accuracy of the given dataset.
        '''
        sloader = DataLoader(dataset, **self.dataloader.get('scoring', {}))
        d = next(iter(sloader))
        with torch.no_grad(): 
            M, B, Mp, Bp = self.net(*d[0])          # NOTE: Bp is not softmax-ed
            macc = F.l1_loss(Mp.squeeze(-1), d[1])
            self.board.add_scalar(caption + '/accuracy/malignant', macc, self.cur_epoch)
            self.board.add_image(caption + '/CAM/origin', d[0][0], self.cur_epoch)  # [3, H, W]
            self.board.add_image(caption + '/CAM/malignant', M[0], self.cur_epoch)  # [1, H, W]
            if len(d) == 2: return
        
            bacc = torch.argmax(Bp, -1) == d[2] # but argmax is enough :D
            bacc = bacc.float().mean()
            self.board.add_scalar(caption + '/accuracy/BIRADs', bacc, self.cur_epoch)
            self.board.add_image(
                caption + '/CAM/BIRADs', 
                B[0, torch.argmax(Bp[0], -1)],  # [H, W]
                self.cur_epoch, 'HW'
            )
            return bacc
