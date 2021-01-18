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
from common.utils import KeyboardInterruptWrapper, NoGrad, freeze, gray2JET
from toynetv2 import ToyNetV1, ToyNetV2
    
class ToyNetTrainer(Trainer):
    @property
    def logHotmap(self):
        return hasattr(self.net, 'hotmap') and bool(self.net.hotmap)
    
    def getOptimizer(self):
        name = self.op_conf.get('name', 'SGD')
        op: torch.optim.Optimizer = getattr(torch.optim, name)
        default = self.op_conf.get('default', {})

        if hasattr(self.net, 'seperatedParameters'):
            paramM, paramB = self.net.seperatedParameters()
            def merge(d1, d2):
                d1.update(d2)
                return d1
            return op(
                [merge({'params': p}, self.op_conf.get(i, {})) for i, p in zip(['M', 'B'], [paramM, paramB])], 
                **default
            )
        else: return op(self.net.parameters(), **default)

    def checkGapScheduler(self, optimizer, unanno_only_epoch, last_epoch=-1):
        if not hasattr(self.net, 'seperatedParameters'): return
        return torch.optim.lr_scheduler.LambdaLR(optimizer, [
            lambda e: 1e-4 if e == unanno_only_epoch else 1, 
            lambda e: 1
        ], last_epoch)

    @KeyboardInterruptWrapper(lambda self, *l, **d: print('Training paused. Start from epoch%d next time.' % self.cur_epoch))
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
        gap_sg = self.checkGapScheduler(op, unanno_only_epoch)
        if 'scheduler' in self.conf: 
            sg = torch.optim.lr_scheduler.ReduceLROnPlateau(op, 'min', **self.conf['scheduler'])
        else: sg = None
        # init tensorboard logger
        self.prepareBoard()
        # lemme see who dare to use cpu for training ???
        if self.device.type == 'cpu':
            print('Warning: You are using CPU for training. Be careful of its temperature...')

        for self.cur_epoch in range(self.cur_epoch, self.max_epoch):
            if gap_sg: gap_sg.step(self.cur_epoch)
            bar = Bar('epoch U%03d' % self.cur_epoch, max=len(unanno))
            for X, Ym in uloader:
                X = X.to(self.device)
                Ym = Ym.to(self.device)
                loss, summary = self.net.loss(X, Ym, piter=self.piter)
                loss = freeze(loss, 1 - self.piter)
                loss.backward()
                op.step()
                op.zero_grad()
                self.total_batch += 1
                self.logSummary('unannotated', summary, self.total_batch)
                bar.next(Ym.shape[0])
            bar.finish()

            bar = Bar('epoch A%03d' % self.cur_epoch, max=len(anno))
            for X, Ym, Yb in aloader:
                X = X.to(self.device)
                Ym = Ym.to(self.device)
                if self.cur_epoch < unanno_only_epoch:
                    loss, summary = self.net.loss(X, Ym, piter=self.piter)
                else:
                    Yb = Yb.to(self.device) 
                    loss, summary = self.net.loss(X, Ym, Yb, a=0.9, piter=self.piter)   # freeze 90% of malignant loss
                loss.backward()
                op.step()
                op.zero_grad()
                assert not torch.any(torch.isnan(next(self.net.parameters())))   # debug
                self.total_batch += 1
                self.logSummary('annotated', summary, self.total_batch)
                bar.next(Ym.shape[0])
            bar.finish()

            merr, berr = self.score('annotated/trainset', anno)         # trainset score
            self.score('unannotated/trainset', unanno)
            if vanno: 
                merr, berr = self.score('annotated/validation', vanno)  # validation score
            if vunanno: self.score('unannotated/validation', vunanno)

            if merr < self.best_mark:
                self.best_mark = merr
                self.save('best', merr)
            self.save('latest', merr)

            if sg and self.cur_epoch >= unanno_only_epoch: sg.step(merr)

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

            Pm, Pb = self.net(X.to(self.device))[-2:]   # NOTE: Pb is not softmax-ed
            Pm = torch.argmax(Pm, dim=-1).float()
            mres.append(Pm)
            merr.append(F.l1_loss(Pm, Ym.to(self.device)))

            if len(d) == 2: continue
            else: Yb = d[2]
        
            be = torch.argmax(Pb, -1) != Yb.to(self.device) # but argmax is enough :D
            bres.append(torch.argmax(Pb, dim=-1))
            berr.append(be.float().mean())
                
        merr = torch.stack(merr).mean()
        mres = torch.cat(mres)
        self.board.add_scalar('err/malignant/%s' % caption, merr, self.cur_epoch)
        self.board.add_histogram('distribution/malignant/%s' % caption, mres, self.cur_epoch)

        X, Ym = dataset[:1][:2]
        res = self.net(X.to(self.device))
        if self.logHotmap:
            M, B, _, Pb = res
            hotmap = 0.5 * X[0] + 0.3 * gray2JET(M[0, 0])
            self.board.add_image('%s/CAM malignant' % caption, hotmap, self.cur_epoch, dataformats='CHW')
        else: _, Pb = res

        if not berr: return
        berr = torch.stack(berr).mean()
        bres = torch.cat(bres)
        absurd = ((bres == 0) * (mres == 1)).sum() + ((bres == 5) * (mres == 0)).sum()  # BIRAD-2 but malignant, BIRAD-5 but benign
        
        self.board.add_scalar('absurd/%s' % caption, absurd / mres.shape[0], self.cur_epoch)
        self.board.add_scalar('err/BIRADs/%s' % caption, berr, self.cur_epoch)
        if self.logHotmap:
            B = (B.permute(0, 2, 3, 1) * torch.softmax(Pb, dim=-1)).sum(dim=-1)     # [N, H, W]
            hotmap = 0.5 * X[0] + 0.3 * gray2JET(B[0])
            self.board.add_histogram('distribution/BIRADs/%s' % caption, bres, self.cur_epoch)
            self.board.add_image('%s/CAM BIRADs' % caption, hotmap, self.cur_epoch, dataformats='CHW')
        return merr, berr