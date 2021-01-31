'''
A trainer for toynets.

* author: JamzumSum
* create: 2021-1-13
'''
from itertools import chain

import torch
import torch.nn.functional as F
from progress.bar import Bar
from torch.utils.data import DataLoader

from common.decorators import Batched, KeyboardInterruptWrapper, NoGrad
from common.trainer import Trainer
from common.utils import ConfusionMatrix, freeze, gray2JET, unsqueeze_as
from utils.dict import shallow_update

class ToyNetTrainer(Trainer):
    @property
    def logHotmap(self): return hasattr(self.net, 'support') and 'hotmap' in self.net.support
    @property
    def adversarial(self): return hasattr(self.net, 'support') and 'discriminator' in self.net.support
    @property
    def sg_conf(self): return self.conf.get('scheduler', {})

    def getBranchOptimizerAndScheduler(self, branch: str):
        '''
        branch: default, M, B
        '''
        branch_conf = self.conf.get('branch', {}).get(branch, {})
        op_name, op_arg = branch_conf.get('optimizer', self.op_conf)
        sg_arg = branch_conf.get('scheduler', self.sg_conf)

        op: torch.optim.Optimizer = getattr(torch.optim, op_name)
        paramM, paramB = self.net.seperatedParameters()
        paramD = None
        if self.adversarial: 
            paramD = self.net.discriminatorParameters()
            paramM = chain(paramM, paramD)
        param = {
            'default': paramM, 
            'discriminator': paramD,
            'M': paramM, 
            'B': paramB
        }
        optimizer = op(param[branch], **shallow_update(self.op_conf[1], op_arg, True))

        if not sg_arg: return optimizer, None
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **shallow_update(self.sg_conf, sg_arg, True))
        return optimizer, scheduler

    @KeyboardInterruptWrapper(lambda self, *l, **d: print('Training paused. Start from epoch%d next time.' % self.cur_epoch))
    def train(self, ta, tu, va=None, vu=None):
        # get all loaders
        aloader = DataLoader(ta, **self.dataloader.get('annotated', {}))
        uloader = DataLoader(tu, **self.dataloader.get('unannotated', {}))
        if va: 
            valoader = DataLoader(va, batch_size=aloader.batch_size)
        if vu:
            vuloader = DataLoader(vu, batch_size=uloader.batch_size)
        unanno_only_epoch = self.training.get('use_annotation_from', self.max_epoch)

        # cache all distributions, for they are constants
        normalize1 = lambda x: x / x.sum()
        annoMdistrib = normalize1(torch.Tensor(ta.getDistribution(1)))
        annoBdistrib = normalize1(torch.Tensor(ta.distribution))
        unannoDistrib = normalize1(torch.Tensor(tu.distribution))
        if va: 
            vannoMdistrib = normalize1(torch.Tensor(va.getDistribution(1)))
            vannoBdistrib = normalize1(torch.Tensor(va.distribution))
        if vu:
            vunannoDistrib = normalize1(torch.Tensor(vu.distribution))

        # get all optimizers and schedulers
        gop, gsg = self.getBranchOptimizerAndScheduler('default')
        mop, msg = self.getBranchOptimizerAndScheduler('M')
        bop, bsg = self.getBranchOptimizerAndScheduler('B')
        if self.adversarial:
            dop, dsg = self.getBranchOptimizerAndScheduler('discriminator')
        else: dsg = None

        # init tensorboard logger
        self.prepareBoard()
        # lemme see who dare to use cpu for training ???
        if self.device.type == 'cpu':
            print('Warning: You are using CPU for training. Be careful of its temperature...')

        def trainInBatch(X, Ym, Yb=None, ops=None, mweight=None, bweight=None, name=''):
            if self.cur_epoch < unanno_only_epoch: Yb = None
            X = X.to(self.device)
            Ym = Ym.to(self.device)
            if Yb is not None: Yb = Yb.to(self.device)
            if mweight is not None: mweight = mweight.to(self.device)
            if bweight is not None: bweight = bweight.to(self.device)
            loss, summary = self.net.loss(X, Ym, Yb, self.piter, mweight, bweight)
            if ops:
                loss.backward()
                for i in ops:
                    i.step()
                    i.zero_grad()
            if name:
                self.total_batch += 1
                self.logSummary(name, summary, self.total_batch)
            bar.next(Ym.shape[0])
            return loss.detach_(),
        trainWithDataset = Batched(trainInBatch)

        def trainDiscriminator(X, Ym, Yb):
            X = X.to(self.device)
            Ym = Ym.to(self.device)
            Yb = Yb.to(self.device)
            loss = self.net.discriminatorLoss(X, Ym, Yb, piter=self.piter)
            loss.backward()
            dop.step()
            dop.zero_grad()
            bar.next(Ym.shape[0])
            return loss.detach_(),

        if self.cur_epoch == 0:
            self.board.add_graph(self.net, ta.tensors[0][:2].to(self.device))
            
        ops = (gop, )
        for self.cur_epoch in range(self.cur_epoch, self.max_epoch):
            if self.cur_epoch == unanno_only_epoch:
                ops = (mop, bop)
                print('Use annotated datasets from current epoch.')
                print('Optimizer is seperated from now.')
            self.net.train()
            bar = Bar('epoch T%03d' % self.cur_epoch, max=len(tu) + (len(ta) << int(self.adversarial)))
            tull, = trainWithDataset(
                uloader, ops = ops, name='unannotated', mweight=unannoDistrib
            )
            tall, = trainWithDataset(
                aloader, ops = ops, name='annotated', mweight=annoMdistrib, bweight=annoBdistrib
            )
            if self.adversarial: dll, = Batched(trainDiscriminator)(aloader)
            bar.finish()

            amerr, berr = self.score('annotated/trainset', ta)          # trainset score
            umerr, _ = self.score('unannotated/trainset', tu)
            if va: 
                amerr, berr = self.score('annotated/validation', va)    # validation score
            if vu: 
                umerr, _ = self.score('unannotated/validation', vu)
            merr = (amerr + umerr) / 2

            if merr < self.best_mark:
                self.best_mark = merr
                self.save('best', merr)
            self.save('latest', merr)

            ll = torch.cat((tull, tall)).mean()
            if va or vu:
                vll = []
                lenn = lambda x: len(x) if x else 0
                bar = Bar('epoch V%03d' % self.cur_epoch, max=lenn(va) + lenn(vu))
                if va: vll.append(*trainWithDataset(valoader, mweight=vannoMdistrib, bweight=vannoBdistrib))
                if vu: vll.append(*trainWithDataset(vuloader, mweight=vunannoDistrib))
                bar.finish()
                ll = (ll + torch.cat(vll).mean()) / 2
            
            if dsg: dsg.step(dll.mean())
            if self.cur_epoch < unanno_only_epoch: 
                if gsg: gsg.step(ll)
            else:
                if msg: msg.step(ll)
                if bsg: bsg.step(ll)

    @NoGrad
    def score(self, caption, dataset):
        '''
        Score and evaluate the given dataset.
        '''
        self.net.eval()
        annotated = len(dataset.tensors) == 3
        mcm = ConfusionMatrix(2)
        dcm = ConfusionMatrix(2) if self.adversarial else None
        bcm = ConfusionMatrix(self.net.K) if annotated else None

        def scoresInBatch(X, Ym, Yb=None):
            nonlocal mcm, dcm, bcm
            X = X.to(self.device)
            Ym = Ym.to(self.device)
            Pm, Pb = self.net(X)[-2:]   # NOTE: Pm & Pb is not softmax-ed
            if self.adversarial:
                cy0 = self.net.D(Pm, Pb).argmax(dim=1)  # [N, 1]
                dcm.add(cy0, torch.zeros_like(cy0))

            Pm = Pm.argmax(dim=1)
            Pb = Pb.argmax(dim=1)
            mcm.add(Pm, Ym)
            if Yb is None: return Pm, Pb

            Yb = Yb.to(self.device)
            bcm.add(Pb, Yb)
            if self.adversarial:
                cy1 = self.net.D(
                    F.one_hot(Ym, num_classes=2).float(), 
                    F.one_hot(Yb, num_classes=self.net.K).float()
                )
                dcm.add(cy1, torch.ones_like(cy1))
            return Pm, Pb
        
        loader = DataLoader(dataset, **self.dataloader.get('scoring', {}))
        Pm, Pb = Batched(scoresInBatch)(loader)

        cmdic = {'malignant': mcm}; errl = []
        if annotated: cmdic['BIRADs'] = bcm
        if self.adversarial: cmdic['discriminator'] = dcm

        for k, v in cmdic.items():
            errl.append(v.err())
            self.board.add_image('ConfusionMat/%s/%s' % (k, caption), v.m, self.cur_epoch, dataformats='HW')
            self.board.add_scalar('err/%s/%s' % (k, caption), errl[-1], self.cur_epoch)
            self.board.add_scalar('f1/%s/%s' % (k, caption), v.fscore(), self.cur_epoch)
        
        self.board.add_histogram('distribution/malignant/%s' % caption, Pm, self.cur_epoch)
        self.board.add_histogram('distribution/BIRADs/%s' % caption, Pb, self.cur_epoch)

        if self.logHotmap:
            # BUG: since datasets are sorted, images extracted are biased. (Bengin & BIRADs-2)
            X = next(iter(loader))[0].to(self.device)
            M, B, mw, bw = self.net(X)
            wsum = lambda x, w: (x * unsqueeze_as(w.softmax(dim=-1), x)).sum(dim=1)
            heatmap = lambda x, w: 0.7 * X + 0.1 * gray2JET(wsum(x, w))
            # Now we generate CAM even if dataset is BIRADS-unannotated.
            self.board.add_images('%s/CAM malignant' % caption, heatmap(M, mw), self.cur_epoch)
            self.board.add_images('%s/CAM BIRADs' % caption, heatmap(B, bw), self.cur_epoch)

        return errl[:2]