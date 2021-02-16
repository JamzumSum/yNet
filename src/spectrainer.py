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
from common.utils import ConfusionMatrix, gray2JET, unsqueeze_as
from common.loss import diceCoefficient
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
            mask, self.piter
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
        vloader = FixLoader(vd, device=self.device, batch_size=loader.batch_size, shuffle=True)

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
                loader, ops = ops, name='ourset', bar=bar, dop=dop
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

            merr = self.score('trainset', no_aug)[0]      # trainset score
            merr = self.score('validation', vd)[0]        # validation score
            self.traceNetwork()
            
            if merr < self.best_mark:
                self.best_mark = merr
                self.save('best', merr)
            self.save('latest', merr)

    @NoGrad
    def score(self, caption, dataset, thresh=.5):
        '''
        Score and evaluate the given dataset.
        '''
        self.net.eval()
        def scoresInBatch(X, Ym, Yb=None, mask=None):
            seg, embed, Pm, Pb = self.net(X)
            res = {
                'pm': Pm,
                'pb': Pb,
                'ym': Ym,
                'c': embed
            }
            if self.adversarial:
                cy0 = self.net.D(Pm, Pb)
                res['cy'] = cy0
                res['cy-GT'] = torch.zeros_like(cy0, dtype=torch.long)
            
            if seg is not None:
                res['dice'] = diceCoefficient(seg, mask, reduction='none')

            if Yb is not None: 
                res['yb'] = Yb
            if self.adversarial and Yb is not None:
                cy1 = self.net.D(
                    # TODO: perturbations
                    F.one_hot(Ym, num_classes=2).float(), 
                    F.one_hot(Yb, num_classes=self.net.K).float()
                )
                res['cy'] = torch.cat((res['cy'], cy1), dim=0)
                res['cy-GT'] = torch.cat((res['cy-GT'], torch.ones_like(cy1, dtype=torch.long)), dim=0)
            return res
        
        loader = FixLoader(dataset, device=self.device, **self.dataloader['scoring'])
        res: dict = Batched(scoresInBatch)(loader)
        self.board.add_embedding(res['c'], metadata=res['ym'].tolist(), global_step=self.cur_epoch, tag=caption)

        errl = []
        items = {
            'B-M': ('pm', 'ym'), 
            'BIRAD': ('pb', 'yb'),
            'discrim': ('cy', 'cy-GT')
        }
        for k, (p, y) in items.items():
            if y not in res: continue
            p, y = res[p], res[y]
            
            if p.dim() == 1: argp = (p > thresh).int()
            elif p.dim() == 2: argp = p.argmax(dim=1)
            err = (argp != y).float().mean()
            self.board.add_scalar('err/%s/%s' % (k, caption), err, self.cur_epoch)
            errl.append(err)

            if p.dim() == 1: 
                self.board.add_pr_curve('%s/%s' % (k, caption), p, y, self.cur_epoch)
            elif p.dim() == 2 and p.size(1) <= 2: 
                self.board.add_pr_curve('%s/%s' % (k, caption), p[:, -1], y, self.cur_epoch)
            
        self.board.add_histogram('distribution/malignant/%s' % caption, res['pm'], self.cur_epoch)
        self.board.add_histogram('distribution/BIRADs/%s' % caption, res['pb'], self.cur_epoch)

        if not (self.logSegmap or self.logHotmap): return errl

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
            seg = self.net(X, segment=True)[0].squeeze(1)
            self.board.add_images('%s/segment' % caption, heatmap(X, seg), self.cur_epoch)

        return errl
