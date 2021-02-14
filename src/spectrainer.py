'''
A trainer for toynets.

* author: JamzumSum
* create: 2021-1-13
'''
from itertools import chain

import torch
import torch.nn.functional as F
from progress.bar import Bar

from common.decorators import Batched, NoGrad
from common.trainer import Trainer
from common.utils import ConfusionMatrix, freeze, gray2JET, unsqueeze_as
from utils.dict import shallow_update
from data.dataloader import FixLoader


lenn = lambda x: len(x) if x else 0
first = lambda it: next(iter(it))

class ToyNetTrainer(Trainer):
    def support(self, feature: str):
        return hasattr(self.net, 'support') and feature in self.net.support
    @property
    def logHotmap(self): return self.support('hotmap')
    @property
    def adversarial(self): return self.support('discriminator')
    @property
    def logSegmap(self): return self.support('segment')
    @property
    def sg_conf(self): return self.conf.get('scheduler', {})
    @property
    def discardYbEpoch(self): return self.training.get('use_annotation_from', self.max_epoch)

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

    def trainInBatch(self, X, Ym, Yb=None, mask=None,
            name='', ops=None, dop=None, bar=None):
        if dop and not ops: raise ValueError

        N = Ym.shape[0]
        res, loss, summary = self.net.lossWithResult(
            X, Ym, 
            None if self.cur_epoch < self.discardYbEpoch else Yb, 
            self.piter
        )
        if name:
            self.total_batch += 1
            self.logSummary(name, summary, self.total_batch)
        if ops:
            loss.backward()
            for i in ops:
                i.step()
                i.zero_grad()
        bar.next(N)
        if dop:
            res = res[0][-2:]
            dloss = self.net.discriminatorLoss(*res, Ym, Yb, piter=self.piter)
            dloss.backward()
            dop.step()
            dop.zero_grad()
            bar.next(N)
            return loss.detach_(), dloss.detach_()
        return loss.detach_(),

    def train(self, td, vd, no_aug=None):
        if no_aug is None: no_aug = td
        # get all loaders
        loader = FixLoader(td, device=self.device, **self.dataloader['training'])
        vloader = FixLoader(vd, device=self.device, batch_size=loader.batch_size)

        # get all optimizers and schedulers
        gop, gsg = self.getBranchOptimizerAndScheduler('default')
        mop, msg = self.getBranchOptimizerAndScheduler('M')
        bop, bsg = self.getBranchOptimizerAndScheduler('B')
        if self.adversarial:
            dop, dsg = self.getBranchOptimizerAndScheduler('discriminator')
        else: dop = dsg = None

        # init tensorboard logger
        self.prepareBoard()
        # lemme see who dare to use cpu for training ???
        if self.device.type == 'cpu':
            print('Warning: You are using CPU for training. Be careful of its temperature...')
        else:
            torch.cuda.reset_peak_memory_stats()
            
        trainWithDataset = Batched(self.trainInBatch)

        if self.cur_epoch == 0:
            demo = first(loader)['X'][:2]
            self.board.add_graph(self.net, demo)
            del demo

        ops = (gop, )
        for self.cur_epoch in range(self.cur_epoch, self.max_epoch):
            if self.cur_epoch == self.discardYbEpoch:
                ops = (mop, bop)
                print('Train b-branch from epoch%03d.' % self.discardYbEpoch)
                print('Optimizer is seperated from now.')
            self.net.train()
            bar = Bar('epoch T%03d' % self.cur_epoch, max=(lenn(td) << int(self.adversarial)))
            res = trainWithDataset(
                loader, ops = ops, name='ourset', bar=Bar, dop=dop
            )
            bar.finish()
            if self.adversarial: 
                tll, dll = res
                dll = dll.mean()
            else: tll, = res
            tll = tll.mean()
            
            bar = Bar('epoch V%03d' % self.cur_epoch, max=lenn(vd))
            vll = trainWithDataset(vloader, bar=bar)
            vll = torch.cat(vll).mean()
            bar.finish()
            ll = (tll + vll) / 2
            
            if dsg: dsg.step(dll)
            if self.cur_epoch < self.discardYbEpoch: 
                if gsg: gsg.step(ll)
            else:
                if msg: msg.step(ll)
                if bsg: bsg.step(ll)

            merr, _ = self.score('ourset/trainset', no_aug)      # trainset score
            merr, _ = self.score('ourset/validation', vd)        # validation score
            self.traceNetwork()
            
            if merr < self.best_mark:
                self.best_mark = merr
                self.save('best', merr)
            self.save('latest', merr)

    @NoGrad
    def score(self, caption, dataset):
        '''
        Score and evaluate the given dataset.
        '''
        self.net.eval()
        measureB = 'Yb' in dataset.statTitle
        mcm = ConfusionMatrix(2)
        bcm = ConfusionMatrix(self.net.K) if measureB else None
        dcm = ConfusionMatrix(2) if self.adversarial else None

        def scoresInBatch(X, Ym, Yb=None, mask=None):
            nonlocal mcm, dcm, bcm
            Pm, Pb = self.net(X)[-2:]
            if self.adversarial:
                cy0 = self.net.D(Pm, Pb).round().int()
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
                ).round().int()
                dcm.add(cy1, torch.ones_like(cy1))
            return Pm, Pb
        
        loader = FixLoader(dataset, device=self.device, **self.dataloader['scoring'])
        Pm, Pb = Batched(scoresInBatch)(loader)

        cmdic = {
            'B-M': mcm, 
        }
        errl = []
        if bcm: cmdic['BIRAD'] = bcm
        if dcm: cmdic['discrim'] = dcm

        for k, v in cmdic.items():
            errl.append(v.err())
            self.board.add_image('ConfusionMat/%s/%s' % (k, caption), v.mat(), self.cur_epoch, dataformats='HW')
            self.board.add_scalar('err/%s/%s' % (k, caption), errl[-1], self.cur_epoch)
            self.board.add_scalar('f1/%s/%s' % (k, caption), v.fscore(), self.cur_epoch)
        
        self.board.add_histogram('distribution/malignant/%s' % caption, Pm, self.cur_epoch)
        self.board.add_histogram('distribution/BIRADs/%s' % caption, Pb, self.cur_epoch)

        if not (self.logSegmap or self.logHotmap): return errl[:2]

        # log images
        # BUG: since datasets are sorted, images extracted are biased. (Bengin & BIRADs-2)
        # EMMM: but I don't know which image to sample, cuz if images are not fixed, 
        # we cannot compare through epochs... and also, since scoring is not testing, 
        # extracting (nearly) all mis-classified samples is excessive...
        X = first(loader)['X'][:8].to(self.device)
        heatmap = lambda x, s: 0.7 * x + 0.1 * gray2JET(s, thresh=.1)
        if self.logHotmap:
            M, B, mw, bw = self.net(X)
            wsum = lambda x, w: (x * unsqueeze_as(w / w.sum(dim=1, keepdim=True), x)).sum(dim=1)
            # Now we generate CAM even if dataset is BIRADS-unannotated.
            self.board.add_images('%s/CAM malignant' % caption, heatmap(M, wsum(M, mw)), self.cur_epoch)
            self.board.add_images('%s/CAM BIRADs' % caption, heatmap(B, wsum(B, bw)), self.cur_epoch)
            
        if self.logSegmap:
            X = first(loader)['X'][:8].to(self.device)
            seg = self.net(X, segment=True)[0]
            self.board.add_images('%s/segment' % caption, heatmap(X, seg), self.cur_epoch)

        return errl[:2]