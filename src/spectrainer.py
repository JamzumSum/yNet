"""
A trainer for toynets.

* author: JamzumSum
* create: 2021-1-13
"""
from warnings import warn

import torch
from torch import nn
from omegaconf import DictConfig, ListConfig
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from torchmetrics.functional import accuracy

from common import deep_collate, unsqueeze_as
from common.loss import confidence, diceCoefficient
from common.optimizer import (
    get_arg_default, get_need_decay, getBranchScheduler, no_decay, split_upto_decay
)
from common.support import *
from common.trainer.fsm import FSMBase
from common.utils import gray2JET
from misc.decorators import noexcept
from misc.updatedict import shallow_update
from toynet.lossbase import MultiTask
from toynet.toynetv1 import ToyNetV1


def getPLSchedulerDict(sg, monitor=None):
    if isinstance(sg, ReduceLROnPlateau):
        return {
            "scheduler": sg,
            "interval": "epoch",
            "reduce_on_plateau": True,
            "monitor": monitor,
        }
    elif isinstance(sg, OneCycleLR):
        return {
            "scheduler": sg,
            "interval": "step",
        }
    else:
        warn(f'{type(sg)} is not implement. `step` will be called after every epoch.')
        return {
            "scheduler": sg,
            "interval": "epoch",
        }


class ToyNetTrainer(FSMBase):
    net: MultiTask

    def __init__(
        self,
        net: nn.Module,
        cmgr, 
        conf: DictConfig
    ):
        self.branch_conf = conf.get('branch', {})
        super().__init__(net, cmgr, conf)

        self.flood = self.misc.get("flood", 0)
        self.scoring_log_image_epoch_train = -1

        self.logHotmap = isinstance(self.net, HeatmapSupported)
        self.adversarial = isinstance(self.net, HasDiscriminator)
        self.logSegmap = isinstance(self.net, SegmentSupported)

        self.example_input_array = torch.zeros((2, 1, 512, 512))

    def forward(self, X):
        r = self.net(X)
        return r['pm']

    def overrided_opsg(self, branches: list):
        op_arg, sg_arg = {}, {}
        for k in branches:
            d = self.branch_conf.get(k, {})
            op_arg[k] = shallow_update(
                self.op_conf, d.get("optimizer", self.op_conf), True
            )
            sg_arg[k] = shallow_update(
                self.sg_conf, d.get("scheduler", self.sg_conf), True
            )
        return op_arg, sg_arg

    def configure_optimizers(self):
        branchstr = self.net.branches if isinstance(self.net, MultiBranch) else 'G'
        branchstr = list(branchstr)
        if self.adversarial: branchstr.append('D')
        op_arg, sg_arg = self.overrided_opsg(branchstr)

        get_op_init_default = lambda arg: get_arg_default(self.op_cls.__init__, arg)
        default_decay, default_lr = (
            get_op_init_default("weight_decay"),
            get_op_init_default("lr"),
        )
        weight_decay = {
            k: d.get("weight_decay", default_decay)
            for k, d in op_arg.items()
        }

        # add support for non-multibranch model, usually a baseline like resnet.
        if isinstance(self.net, MultiBranch):
            paramdic = self.net.branch_weight(weight_decay)
            mop_param_key = ([i, i + "_no_decay"] for i in self.net.branches)
            mop_param_key = sum(mop_param_key, [])
            msg_param_key = self.net.branches
        else:
            paramdic = {"G": self.net.parameters()}
            paramdic = split_upto_decay(
                get_need_decay(self.net.modules()), paramdic, weight_decay
            )
            mop_param_key = ["G", "G_no_decay"]
            msg_param_key = ["G"]

        paramdic, param_group_key = no_decay(
            weight_decay, paramdic, op_arg, self.op_conf.get("lr", default_lr)
        )

        mop_param = [paramdic[k] for k in mop_param_key if k in paramdic]
        mop = self.op_cls(mop_param, **self.op_conf)
        ops = [mop]
        if self.adversarial:
            dop_param = [v for k, v in paramdic.items() if k not in mop_param_key]
            ops.append(self.op_cls(dop_param, **self.op_conf))

        if not any(sg_arg.values()):
            return ops

        msg_param = [sg_arg[k] for k in param_group_key if k in msg_param_key]
        sgs = []
        extra = {
            'epochs': self.trainer.max_epochs,
            'steps_per_epoch': self.steps_per_epoch
        }
        if msg_param:
            msg = getBranchScheduler(self.sg_cls, mop, msg_param, extra)
            sgs.append(getPLSchedulerDict(msg, "val_loss"))
        if self.adversarial:
            dsg_param = [sg_arg[k] for k in param_group_key if k not in msg_param_key]
            if dsg_param:
                dsg = getBranchScheduler(self.sg_cls, ops[-1], dsg_param, extra)
                sgs.append(getPLSchedulerDict(dsg, "dis_loss"))

        return ops, sgs

    @noexcept
    def optimizer_step(
        self, current_epoch, batch_nb, optimizer, optimizer_idx, closure, *args,
        **kwargs
    ):
        # warm up lr
        warmup_conf: dict = self.misc.get('lr_warmup', {})
        if warmup_conf is None:
            return optimizer.step(closure=closure)

        interval = warmup_conf.get('interval', 'step')
        times = warmup_conf.get('times', 500)

        if interval == 'epoch':
            times *= self.steps_per_epoch

        if self.trainer.global_step < times:
            lr_scale = min(1., (self.trainer.global_step + 1.) / times)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * pg['initial_lr']

        # update params
        optimizer.step(closure=closure)

    @noexcept
    def training_step(self, batch: dict, batch_idx: int, opid=0):
        if opid == 0:

            res, lossdic = self.net.__loss__(**batch)
            loss = self.net.multiTaskLoss(lossdic)
            summary = self.net.lossSummary(lossdic)
            del lossdic

            self.log("train_loss", loss, logger=False, prog_bar=True)
            self.log_dict(summary)
            self.buf = res

            if self.flood > 0:
                loss = (loss - self.flood).abs() + self.flood

        elif opid == 1:

            Ym, Yb = batch["Ym"], batch["Yb"]
            res = self.buf[0][-2:]
            loss = self.net.discriminatorLoss(self.cosg, *res, Ym, Yb)
            self.log("dis_loss", loss, prog_bar=True, logger=False)

        return loss

    @noexcept
    def validation_step(self, batch: dict, batch_idx, dataloader_idx=0):
        if dataloader_idx == 1:
            return

        loss = self.net.loss(**batch)
        self.log("val_loss", loss, prog_bar=True, logger=False)
        return loss

    def score_step(self, batch: tuple, batch_idx, dataloader_idx=0):
        X, Ym, Yb, mask = batch["X"], batch["Ym"], batch["Yb"], batch["mask"]
        res = self.net(X)

        res['ym'] = Ym
        res["yb"] = [None] * X.size(0) if Yb is None else Yb.tolist()

        if mask is not None and self.logSegmap:
            res["dice"] = diceCoefficient(res['seg'], mask, reduction="none")
            res['confidence'] = confidence(res['seg'], reduction='none')

        # BUGGY!
        # if self.adversarial:
        #     cy0 = self.net.D(res['pm'], res['pb'])
        #     res["cy"] = cy0
        #     res["cy-GT"] = torch.zeros_like(cy0, dtype=torch.long)
        #     if Yb is not None:
        #         # BUG: perturbations
        #         cy1 = self.net.D(
        #             F.one_hot(Ym, num_classes=2).float(),
        #             F.one_hot(Yb, num_classes=self.net.K).float(),
        #         )
        #         res["cy"] = torch.cat((res["cy"], cy1), dim=0)
        #         res["cy-GT"] = torch.cat(
        #             (res["cy-GT"], torch.ones_like(cy1, dtype=torch.long)), dim=0
        #         )

        if dataloader_idx == 0:
            # validation
            if batch_idx == 0:
                self.log_images(X, dataloader_idx)
        elif dataloader_idx == 1:
            # trainset, to find a batch that exactly has gt supervision.
            if mask is not None and self.scoring_log_image_epoch_train < self.current_epoch:
                self.log_images(X, dataloader_idx)
                self.scoring_log_image_epoch_train = self.current_epoch
        return res

    def score_epoch_end(self, results: list[dict[str, torch.Tensor]]):
        """
        Score and evaluate the given dataset.
        """
        items = {
            "B-M": ("pm", "ym"),
            "BIRAD": ("pb", "yb"),
            "discrim": ("cy", "cy-GT"),
        }

        for res, caption in zip(results, self.score_caption):
            res = deep_collate(res)

            if (c := ('fi' in res and 'fi') or ('ft' in res and 'ft')):
                self.logger.experiment.add_embedding(
                    res[c],
                    metadata=res["ym"].tolist(),
                    global_step=self.current_epoch,
                    tag=caption,
                )

            if "dice" in res:
                self.log(f"segment/dice/{caption}", res["dice"].mean(), logger=True)
            if 'confidence' in res:
                self.log(
                    f"segment/confidence/{caption}",
                    res["confidence"].mean(),
                    logger=True
                )

            for k, (p, y) in items.items():
                if p not in res or y not in res: continue
                p, y = res[p], res[y]
                if all(i is None for i in y): continue

                if isinstance(y, list):
                    p = torch.stack([p[i] for i, v in enumerate(y) if v is not None])
                    y = torch.tensor([v for i, v in enumerate(y) if v is not None],
                                     dtype=torch.long,
                                     device=p.device)

                assert len(p) == len(y)
                err = 1 - accuracy(p, y)
                self.log(f"err/{k}/{caption}", err, logger=True)

                if p.dim() == 2 and p.size(1) <= 2:
                    p = p[:, -1]
                if p.dim() == 1:
                    self.logger.experiment.add_pr_curve(
                        f"{k}/{caption}", y, p, self.current_epoch
                    )
                    self.logger.experiment.add_histogram(
                        f"distribution/{k}/{caption}", p, self.current_epoch
                    )
                else:
                    self.logger.experiment.add_histogram(
                        f"distribution/{k}/{caption}", p, self.current_epoch
                    )

    def log_images(self, X, dataloader_idx):
        X = X[:8]                                                                       # log only 8 images for a pretty present.
        caption = self.score_caption[dataloader_idx]
        heatmap = lambda x, s: 0.7 * x + 0.1 * gray2JET(s, thresh=0.15)
        if self.logHotmap:
            M, B, mw, bw = self.net(X)
            wsum = lambda x, w: (x * unsqueeze_as(w / w.sum(dim=1, keepdim=True), x)
                                 ).sum(dim=1)
                                                                                        # Now we generate CAM even if dataset is BIRADS-unannotated.
            self.logger.experiment.add_images(
                f"{caption}/CAM malignant", heatmap(M, wsum(M, mw)), self.current_epoch
            )
            self.logger.experiment.add_images(
                f"{caption}/CAM BIRADs", heatmap(B, wsum(B, bw)), self.current_epoch
            )

        if self.logSegmap:
            seg = self.net(X, classify=False)['seg'].squeeze(1)
            self.logger.experiment.add_images(
                f"{caption}/segment", heatmap(X, seg), self.current_epoch
            )

    @noexcept
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        X, ym, detail = batch["X"], batch["Ym"], batch["meta"]
        extra = {}
        if isinstance(self.net, SegmentSupported):
            extra['segment'] = False
        r: dict = self.net(X, **extra)

        null = [None] * X.size(0)
        pb = r.get('pb', null)
        yb = batch.get('Yb', null)
        if yb is None: yb = null

        for meta, pmi, ymi, pbi, ybi in zip(detail, r['pm'], ym, pb, yb):
            self.logger.log_metrics(
                meta,
                pm=pmi[1].item(),
                pb=None if pbi is None else pbi.tolist(),
                ym=ymi.item(),
                yb=None if ybi is None else ybi.item(),
            )
