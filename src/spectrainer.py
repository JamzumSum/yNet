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
from common.utils import KeyboardInterruptWrapper, NoGrad
from toynetv2 import ToyNetV1, ToyNetV2

class ToyNetTrainer(Trainer):
    @KeyboardInterruptWrapper(lambda self: print('Training resumed. Start from epoch%d next time.' % self.cur_epoch))
    def train(self, anno, unanno, vanno=None, vunanno=None):
        aloader = DataLoader(anno, **self.dataloader.get('annotated', {}))
        uloader = DataLoader(unanno, **self.dataloader.get('unannotated', {}))
        unanno_only_epoch = self.training.get('use_annotation_from', self.max_epoch)
        try: op = self.getOptimizer()
        except ValueError as e:
            print(e)
            print('Trainer exits before iteration starts.')
            return
        self.prepareBoard()
        if self.device.type == 'cpu':
            print('Warning: You are using CPU for training. Be careful of its temperature...')

        for self.cur_epoch in range(self.cur_epoch, self.max_epoch):
            bar = Bar('epoch U%03d' % self.cur_epoch, max=len(unanno))
            for X, Ym in uloader:
                X = X.to(self.device)
                Ym = Ym.to(self.device)
                loss, summary = self.net.loss(X, Ym, piter=self.cur_epoch / self.max_epoch)
                loss.backward()
                op.step()
                self.total_batch += 1
                self.logSummary(summary, self.total_batch)
                bar.next(Ym.shape[0])
            bar.finish()

            bar = Bar('epoch A%03d' % self.cur_epoch, max=len(anno))
            for X, Ym, Yb in aloader:
                X = X.to(self.device)
                Ym = Ym.to(self.device)
                Yb = None if self.cur_epoch < unanno_only_epoch else Yb.to(self.device) 
                loss, summary = self.net.loss(X, Ym, Yb, piter=self.cur_epoch / self.max_epoch)
                loss.backward()
                op.step()
                self.total_batch += 1
                self.logSummary(summary, self.total_batch)
                bar.next(Ym.shape[0])
            bar.finish()

            ta_bacc = self.score('annotated/trainset', anno)        # trainset score
            self.score('unannotated/trainset', unanno)
            if vanno: self.score('annotated/validation', vanno)     # validation score
            if vunanno: self.score('unannotated/validation', vunanno)

            if ta_bacc < self.best_mark:
                self.best_mark = ta_bacc
                self.save('best', ta_bacc)
            self.save('latest', ta_bacc)

    @NoGrad
    def score(self, caption, dataset):
        '''
        Calculate accuracy of the given dataset.
        '''
        sloader = DataLoader(dataset, **self.dataloader.get('scoring', {}))
        merr = []; berr = []
        for d in sloader:
            X, Ym = d[:2]

            _, _, Pm, Pb = self.net(X.to(self.device))   # NOTE: Bp is not softmax-ed
            merr.append(F.l1_loss(Pm.squeeze(-1), Ym.to(self.device)))
            if len(d) == 2: continue
            else: Yb = d[2]
        
            Pb = Pb.to(Yb.device)
            be = torch.argmax(Pb, -1) != Yb # but argmax is enough :D
            berr.append(be.float().mean())
                
        merr = torch.cat(merr).mean()
        self.board.add_scalar('eval/%s/error rate/malignant' % caption, merr, self.cur_epoch)

        X, Ym = dataset[:1][:2]
        M, B, _, Pb = self.net(X.to(self.device))
        self.board.add_image('eval/%s/CAM/origin' % caption, X[0], self.cur_epoch)  # [3, H, W]
        self.board.add_image('eval/%s/CAM/malignant' % caption, M[0, 0], self.cur_epoch, dataformats='HW')

        if not berr: return
        berr = torch.cat(berr).mean()
        self.board.add_scalar('eval/%s/error rate/BIRADs' % caption, berr, self.cur_epoch)
        self.board.add_image(
            'eval/%s/CAM/BIRADs' % caption, 
            B[0, torch.argmax(Pb[0], -1)],
            self.cur_epoch, dataformats='HW'
        )
        return berr