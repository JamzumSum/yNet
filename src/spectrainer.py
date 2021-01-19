'''
A trainer for toynets.

* author: JamzumSum
* create: 2021-1-13
'''
import torch
import torch.nn.functional as F
from progress.bar import Bar
from torch.utils.data import DataLoader
from itertools import chain

from common.trainer import Trainer
from common.utils import freeze, gray2JET
from common.decorators import KeyboardInterruptWrapper, NoGrad, Batched
from utils import update_default
    
class ToyNetTrainer(Trainer):
    @property
    def logHotmap(self): return hasattr(self.net, 'hotmap') and bool(self.net.hotmap)
    @property
    def sg_conf(self): return self.conf.get('scheduler', {})

    def getBranchOptimizerAndScheduler(self, branch: str):
        '''
        branch: default, M, B
        '''
        branch_conf = self.conf.get('branch', {}).get(branch, {})
        op_name, op_arg = branch_conf.get('optimizer', ('SGD', {}))
        sg_arg = branch_conf.get('scheduler', self.sg_conf)

        op: torch.optim.Optimizer = getattr(torch.optim, op_name)
        paramM, paramB = self.net.seperatedParameters()
        param = {
            'default': chain(paramM, paramB), 
            'M': paramM, 
            'B': paramB
        }
        optimizer = op(param[branch], **update_default(self.op_conf[1], op_arg, True))

        if not sg_arg: return optimizer, None
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **update_default(self.sg_conf, sg_arg, True))
        return optimizer, scheduler

    @KeyboardInterruptWrapper(lambda self, *l, **d: print('Training paused. Start from epoch%d next time.' % self.cur_epoch))
    def train(self, anno, unanno, vanno=None, vunanno=None):
        aloader = DataLoader(anno, **self.dataloader.get('annotated', {}))
        uloader = DataLoader(unanno, **self.dataloader.get('unannotated', {}))
        unanno_only_epoch = self.training.get('use_annotation_from', self.max_epoch)

        # get optimizer
        dop, dsg = self.getBranchOptimizerAndScheduler('default')
        mop, msg = self.getBranchOptimizerAndScheduler('M')
        bop, bsg = self.getBranchOptimizerAndScheduler('B')
        # init tensorboard logger
        self.prepareBoard()
        # lemme see who dare to use cpu for training ???
        if self.device.type == 'cpu':
            print('Warning: You are using CPU for training. Be careful of its temperature...')

        @Batched
        def trainInBatch(X, Ym, Yb=None):
            if self.cur_epoch < unanno_only_epoch: Yb = None
            X = X.to(self.device)
            Ym = Ym.to(self.device)
            loss, summary = self.net.loss(X, Ym, piter=self.piter)
            loss = freeze(loss, 1 - self.piter)
            loss.backward()
            for i in ops:
                i.step()
                i.zero_grad()
            self.total_batch += 1
            self.logSummary(name, summary, self.total_batch)
            bar.next(Ym.shape[0])

        for self.cur_epoch in range(self.cur_epoch, self.max_epoch):
            if self.cur_epoch < unanno_only_epoch: 
                ops = (dop, )
            else:
                ops = (mop, bop)

            bar = Bar('epoch U%03d' % self.cur_epoch, max=len(unanno))
            name = 'unannotated'
            trainInBatch(uloader)
            bar.finish()

            bar = Bar('epoch A%03d' % self.cur_epoch, max=len(anno))
            name = 'annotated'
            trainInBatch(aloader)
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

            if self.cur_epoch < unanno_only_epoch: 
                if dsg: dsg.step(merr)
            else:
                if msg: msg.step(merr)
                if bsg: bsg.step(berr)

    @NoGrad
    def score(self, caption, dataset):
        '''
        Calculate accuracy of the given dataset.
        '''
        @Batched
        def scoresInBatch(X, Ym, Yb=None):
            Pm, Pb = self.net(X.to(self.device))[-2:]   # NOTE: Pb is not softmax-ed
            Pm = torch.argmax(Pm, dim=-1).float()
            merr = F.l1_loss(Pm, Ym.to(self.device))
            if Yb is None: return Pm, merr
        
            be = torch.argmax(Pb, -1) != Yb.to(self.device) # but argmax is enough :D
            Pb = torch.argmax(Pb, dim=-1)
            berr = be.float().mean()
            return Pm, merr, Pb, berr
        
        loader = DataLoader(dataset, **self.dataloader.get('scoring', {}))
        res = scoresInBatch(loader)
        if len(res) == 2:
            Pm, merr = res
            Pb = berr = None
        elif len(res) == 4:
            Pm, merr, Pb, berr = res

        merr = merr.mean()
        # absurd: sum of predicts that are of BIRAD-2 but malignant; of BIRAD-5 but benign
        # NOTE: the criterion is class-specific. change it once class-map are changed.
        absurd = ((Pb == 0) * (Pm == 1)).sum() + ((Pb == 5) * (Pm == 0)).sum()
        
        self.board.add_scalar('absurd/%s' % caption, absurd / Pm.shape[0], self.cur_epoch)
        self.board.add_scalar('err/malignant/%s' % caption, merr, self.cur_epoch)
        self.board.add_histogram('distribution/malignant/%s' % caption, Pm, self.cur_epoch)

        X = dataset[:1][0].to(self.device)
        res = self.net(X)

        if self.logHotmap:
            M, B, _, bweights = res
            hotmap = 0.5 * X[0] + 0.3 * gray2JET(M[0, 0])
            self.board.add_image('%s/CAM malignant' % caption, hotmap, self.cur_epoch, dataformats='CHW')
            # Now we generate CAM even if dataset is BIRADS-unannotated.
            B = (B.permute(0, 2, 3, 1) * torch.softmax(bweights, dim=-1)).sum(dim=-1)     # [N, H, W]
            hotmap = 0.5 * X[0] + 0.3 * gray2JET(B[0])
            self.board.add_image('%s/CAM BIRADs' % caption, hotmap, self.cur_epoch, dataformats='CHW')

        if berr is None: return merr

        berr = berr.mean()
        # absurd: sum of predicts that are of BIRAD-2 but malignant; of BIRAD-5 but benign
        # NOTE: the criterion is class-specific. change it once class-map are changed.
        absurd = ((Pb == 0) * (Pm == 1)).sum() + ((Pb == 5) * (Pm == 0)).sum()

        self.board.add_scalar('absurd/%s' % caption, absurd / Pm.shape[0], self.cur_epoch)
        self.board.add_scalar('err/BIRADs/%s' % caption, berr, self.cur_epoch)
        self.board.add_histogram('distribution/BIRADs/%s' % caption, Pb, self.cur_epoch)
        return merr, berr
            