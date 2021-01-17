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

def freeze(tensor, f=0.):
    return (1 - f) * tensor + f * tensor.detach()
class ToyNetTrainer(Trainer):
    @KeyboardInterruptWrapper(lambda self: print('Training paused. Start from epoch%d next time.' % self.cur_epoch))
    def train(self, anno, unanno, vanno=None, vunanno=None):
        aloader = DataLoader(anno, **self.dataloader.get('annotated', {}))
        uloader = DataLoader(unanno, **self.dataloader.get('unannotated', {}))
        unanno_only_epoch = self.training.get('use_annotation_from', self.max_epoch)

        # get optimizer
        try: op = self.getOptimizer()
        except ValueError as e:
            print(e)
            print('Trainer exits before iteration starts.')
            return
        # get scheduler
        if 'scheduler' in self.conf: 
            sg = torch.optim.lr_scheduler.ReduceLROnPlateau(op, 'min', **self.conf['scheduler'])
        else: sg = None
        # init tensorboard logger
        self.prepareBoard()
        # lemme see who dare to use cpu for training ???
        if self.device.type == 'cpu':
            print('Warning: You are using CPU for training. Be careful of its temperature...')

        for self.cur_epoch in range(self.cur_epoch, self.max_epoch):
            bar = Bar('epoch U%03d' % self.cur_epoch, max=len(unanno))
            for X, Ym in uloader:
                X = X.to(self.device)
                Ym = Ym.to(self.device)
                with torch.no_grad():
                    loss, summary = self.net.loss(X, Ym, piter=self.cur_epoch / self.max_epoch)
                # loss.backward()
                # op.step()
                # self.total_batch += 1
                self.logSummary('unannotated', summary, self.total_batch)
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
                self.logSummary('annotated', summary, self.total_batch)
                bar.next(Ym.shape[0])
            bar.finish()

            ta_berr = self.score('annotated/trainset', anno)        # trainset score
            self.score('unannotated/trainset', unanno)
            if vanno: self.score('annotated/validation', vanno)     # validation score
            if vunanno: self.score('unannotated/validation', vunanno)

            if ta_berr < self.best_mark:
                self.best_mark = ta_berr
                self.save('best', ta_berr)
            self.save('latest', ta_berr)

            if sg and self.cur_epoch >= unanno_only_epoch: sg.step(ta_berr)

    @NoGrad
    def score(self, caption, dataset):
        '''
        Calculate accuracy of the given dataset.
        '''
        sloader = DataLoader(dataset, **self.dataloader.get('scoring', {}))
        merr = []; berr = []
        mres = []; bres = []
        for d in sloader:
            X, Ym = d[:2]

            _, _, Pm, Pb = self.net(X.to(self.device))   # NOTE: Bp is not softmax-ed
            Pm = Pm.squeeze(-1)
            mres.append(torch.round(Pm))
            merr.append(F.l1_loss(Pm, Ym.to(self.device)))

            if len(d) == 2: continue
            else: Yb = d[2]
        
            be = torch.argmax(Pb, -1) != Yb.to(self.device) # but argmax is enough :D
            bres.append(torch.argmax(Pb))
            berr.append(be.float().mean())
                
        merr = torch.stack(merr).mean()
        mres = torch.cat(mres)
        self.board.add_scalar('eval/%s/error rate/malignant' % caption, merr, self.cur_epoch)
        self.board.add_histogram('eval/%s/distribution/malignant' % caption, mres, self.cur_epoch)

        X, Ym = dataset[:1][:2]
        M, B, _, Pb = self.net(X.to(self.device))
        self.board.add_image('eval/%s/CAM/origin' % caption, X[0], self.cur_epoch)  # [3, H, W]
        self.board.add_image('eval/%s/CAM/malignant' % caption, M[0, 0], self.cur_epoch, dataformats='HW')

        if not berr: return
        B = (B * torch.softmax(Pb, dim=-1)).sum(dim=1)     # [N, H, W]
        berr = torch.stack(berr).mean()
        bres = torch.cat(bres)
        absurd = ((bres == 0) * (mres == 1)).sum() + ((bres == 5) * (mres == 0)).sum()  # BIRAD-2 but malignant, BIRAD-5 but benign
        
        self.board.add_scalar('eval/%s/absurd' % caption, absurd / mres.shape[0], self.cur_epoch)
        self.board.add_scalar('eval/%s/error rate/BIRADs' % caption, berr, self.cur_epoch)
        self.board.add_histogram('eval/%s/CAM/BIRADs' % caption, bres, self.cur_epoch)
        self.board.add_image('eval/%s/CAM/BIRADs' % caption, B[0], self.cur_epoch, dataformats='HW')
        return berr