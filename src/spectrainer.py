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
from utils import update_default

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
        optimizer = op(param[branch], **update_default(self.op_conf[1], op_arg, True))

        if not sg_arg: return optimizer, None
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **update_default(self.sg_conf, sg_arg, True))
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
            return loss.detach_()
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
            return loss.detach_()

        if self.cur_epoch == 0:
            self.board.add_graph(self.net, ta.tensors[0][:2])
            
        for self.cur_epoch in range(self.cur_epoch, self.max_epoch):
            if self.cur_epoch < unanno_only_epoch: 
                mop.state['state'] = gop.state['state'].copy()
                bop.state['state'] = gop.state['state'].copy()

            self.net.train()
            bar = Bar('epoch T%03d' % self.cur_epoch, max=len(tu) + (len(ta) << int(self.adversarial)))
            tull, = trainWithDataset(
                uloader, ops = (gop, ), name='unannotated', mweight=unannoDistrib
            )
            tall, = trainWithDataset(
                aloader, ops = (mop, bop), name='annotated', mweight=annoMdistrib, bweight=annoBdistrib
            )
            if self.adversarial: dll, = Batched(trainDiscriminator)(aloader)
            bar.finish()

            amerr, berr = self.score('annotated/trainset', ta)          # trainset score
            umerr = self.score('unannotated/trainset', tu)
            if va: 
                amerr, berr = self.score('annotated/validation', va)    # validation score
            if vu: 
                umerr = self.score('unannotated/validation', vu)
            merr = (amerr + umerr) / 2

            if merr < self.best_mark:
                self.best_mark = merr
                self.save('best', merr)
            self.save('latest', merr)

            ll = torch.cat((tull, tall)).mean()
            if va or vu:
                vll = []
                bar = Bar('epoch V%03d' % self.cur_epoch)
                if va: vll.append(trainWithDataset(valoader, mweight=vannoMdistrib, bweight=vannoBdistrib))
                if vu: vll.append(trainWithDataset(vuloader, mweight=vunannoDistrib))
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
        bcm = ConfusionMatrix(self.net.K)
        mcm = ConfusionMatrix(2)

        def scoresInBatch(X, Ym, Yb=None):
            Pm, Pb = self.net(X.to(self.device))[-2:]   # NOTE: Pm & Pb is not softmax-ed
            Pm = torch.argmax(Pm, dim=-1)               # but argmax is enough :D
            mcm.add(Pm, Ym.to(self.device))
            if Yb is None: return Pm, 
        
            Pb = torch.argmax(Pb, dim=-1)   
            bcm.add(Pb, Yb.to(self.device))
            return Pm, Pb
        
        loader = DataLoader(dataset, **self.dataloader.get('scoring', {}))
        res = Batched(scoresInBatch)(loader)
        if len(res) == 1:
            (Pm, ) = res
            Pb = berr = None
        elif len(res) == 2:
            Pm, Pb = res
            berr = bcm.err()
        merr = mcm.err()
        
        self.board.add_scalar('err/malignant/%s' % caption, merr, self.cur_epoch)
        self.board.add_scalar('f1/malignant/%s' % caption, mcm.fscore(), self.cur_epoch)
        self.board.add_image('ConfusionMatrix/malignant/%s' % caption, mcm.m, self.cur_epoch, dataformats='HW')
        self.board.add_histogram('distribution/malignant/%s' % caption, Pm, self.cur_epoch)

        X = next(iter(loader))[0].to(self.device)
        res = self.net(X)

        if self.logHotmap:
            M, B, _, bweights = res
            hotmap = 0.6 * X + 0.2 * gray2JET(M[:, 0])
            self.board.add_images('%s/CAM malignant' % caption, hotmap, self.cur_epoch)
            # Now we generate CAM even if dataset is BIRADS-unannotated.
            B = (B * unsqueeze_as(torch.softmax(bweights, dim=-1), B)).sum(dim=1)     # [N, H, W]
            hotmap = 0.6 * X + 0.2 * gray2JET(B)
            self.board.add_images('%s/CAM BIRADs' % caption, hotmap, self.cur_epoch)

        if berr is None: return merr

        # absurd: sum of predicts that are of BIRAD-2 but malignant; of BIRAD-5 but benign
        # NOTE: the criterion is class-specific. change it once class-map are changed.
        # Expected to be substituted by the consistency indicated by the discriminator.
        absurd = ((Pb == 0) * (Pm == 1)).sum() + ((Pb == 5) * (Pm == 0)).sum()

        self.board.add_scalar('absurd/%s' % caption, absurd / Pm.shape[0], self.cur_epoch)
        self.board.add_scalar('err/BIRADs/%s' % caption, berr, self.cur_epoch)
        self.board.add_scalar('f1/BIRADs/%s' % caption, bcm.fscore(), self.cur_epoch)
        self.board.add_image('ConfusionMatrix/BIRADs/%s' % caption, bcm.m, self.cur_epoch, dataformats='HW')
        self.board.add_histogram('distribution/BIRADs/%s' % caption, Pb, self.cur_epoch)
        return merr, berr
            