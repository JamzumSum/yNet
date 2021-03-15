"""
A trainer for toynets.

* author: JamzumSum
* create: 2021-1-13
"""
from itertools import chain
from warnings import warn

import torch
import torch.nn.functional as F

from common import deep_collate, unsqueeze_as
from common.loss import diceCoefficient
from common.optimizer import ReduceLROnPlateau, get_arg_default, no_decay
from common.support import *
from common.trainer.fsm import DictConfig, FSMBase, ListConfig, pl
from common.utils import gray2JET
from misc.updatedict import shallow_update

plateau_lr_dic = lambda sg, monitor: {
    "scheduler": sg,
    "interval": "epoch",
    "reduce_on_plateau": True,
    "monitor": monitor,
}


class ToyNetTrainer(FSMBase):
    def __init__(
        self,
        Net,
        model_conf: DictConfig,
        coeff_conf: DictConfig,
        misc: DictConfig,
        op_conf: ListConfig,
        sg_conf: DictConfig,
        branch_conf: DictConfig,
    ):
        self.branch_conf = branch_conf
        super().__init__(
            Net, model_conf, coeff_conf, misc, op_conf, sg_conf,
        )

        self.discardYbEpoch = self.misc.get("use_annotation_from", 0)
        self.flood = self.misc.get("flood", 0)

        self.logHotmap = isinstance(self.net, HeatmapSupported)
        self.adversarial = isinstance(self.net, HasDiscriminator)
        self.logSegmap = isinstance(self.net, SegmentSupported)

        self.example_input_array = torch.randn((2, 1, 512, 512))
        self.acc = pl.metrics.Accuracy()

    def save_hyperparameters(self):
        super().save_hyperparameters(branch=self.branch_conf)

    def forward(self, X):
        return self.net(X)

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
        branchstr = "MBD" if self.adversarial else "MB"
        op_arg, sg_arg = self.overrided_opsg(branchstr)

        get_op_init_default = lambda arg: get_arg_default(self.op_cls.__init__, arg)
        default_decay, default_lr = (
            get_op_init_default("weight_decay"),
            get_op_init_default("lr"),
        )
        weight_decay = {
            k: d.get("weight_decay", default_decay) for k, d in op_arg.items()
        }

        # add support for non-multibranch model, usually a baseline like resnet.
        if isinstance(self.net, MultiBranch):
            paramdic = self.net.branch_weight(weight_decay)
            mop_param_key = ([i, i + "_no_decay"] for i in self.net.branches)
            mop_param_key = sum(mop_param_key, [])
            msg_param_key = self.net.branches
        else:
            paramdic = {"G": self.net.parameters()}
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
        if msg_param:
            msg = ReduceLROnPlateau(mop, msg_param)
            sgs.append(plateau_lr_dic(msg, "val_loss"))
        if self.adversarial:
            dsg_param = [sg_arg[k] for k in param_group_key if k not in msg_param_key]
            if dsg_param:
                dsg = ReduceLROnPlateau(ops[-1], dsg_param)
                sgs.append(plateau_lr_dic(dsg, "dis_loss"))

        return ops, sgs

    # fmt: off
    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx, closure, *args, **kwargs):
    # fmt: on
        # warm up lr
        if self.trainer.global_step < 500:
            lr_scale = min(1., float(self.trainer.global_step + 1) / 500.)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * pg['initial_lr']

        # update params
        optimizer.step(closure=closure)

    def training_step(self, batch, batch_idx: int, opid=0):
        if opid == 0:

            X, Ym, Yb, mask = batch["X"], batch["Ym"], batch["Yb"], batch["mask"]
            if self.current_epoch < self.discardYbEpoch:
                Yb = None
            res, loss, summary = self.net.lossWithResult(self.cosg, X, Ym, Yb, mask)

            self.log("train_loss", loss, logger=False, prog_bar=True)
            self.log_dict(summary)
            self.buf = res

            if self.flood > 0:
                loss = (loss - self.flood).abs() + self.flood

        elif opid == 1:

            res = self.buf[0][-2:]
            loss = self.net.discriminatorLoss(self.cosg, *res, Ym, Yb)
            self.log("dis_loss", loss, prog_bar=True, logger=False)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 1:
            return
        X, Ym, Yb, mask = batch["X"], batch["Ym"], batch["Yb"], batch["mask"]

        if self.current_epoch < self.discardYbEpoch:
            Yb = None
        loss, _ = self.net.loss(self.cosg, X, Ym, Yb, mask)
        self.log("val_loss", loss, prog_bar=True, logger=False)
        return loss

    def score_step(self, batch: tuple, batch_idx, dataloader_idx=0):
        X, Ym, Yb, mask = batch["X"], batch["Ym"], batch["Yb"], batch["mask"]
        seg, embed, Pm, Pb = self.net(X)

        res = {
            "pm": Pm,
            "pb": Pb,
            "ym": Ym,
            "c": embed,
        }
        if self.adversarial:
            cy0 = self.net.D(Pm, Pb)
            res["cy"] = cy0
            res["cy-GT"] = torch.zeros_like(cy0, dtype=torch.long)

        if mask is not None:
            res["dice"] = diceCoefficient(seg, mask, reduction="none")

        if Yb is not None:
            res["yb"] = Yb
        if self.adversarial and Yb is not None:
            cy1 = self.net.D(
                # TODO: perturbations
                F.one_hot(Ym, num_classes=2).float(),
                F.one_hot(Yb, num_classes=self.net.K).float(),
            )
            res["cy"] = torch.cat((res["cy"], cy1), dim=0)
            res["cy-GT"] = torch.cat(
                (res["cy-GT"], torch.ones_like(cy1, dtype=torch.long)), dim=0
            )

        if batch_idx == 0:
            self.log_images(X[:8], self.score_caption[dataloader_idx])
        return res

    def score_epoch_end(self, res: list):
        """
        Score and evaluate the given dataset.
        """
        items = {
            "B-M": ("pm", "ym"),
            "BIRAD": ("pb", "yb"),
            "discrim": ("cy", "cy-GT"),
        }

        for dataset_idx, res in enumerate(res):
            res = deep_collate(res)
            caption = self.score_caption[dataset_idx]

            self.logger.experiment.add_embedding(
                res["c"],
                metadata=res["ym"].tolist(),
                global_step=self.current_epoch,
                tag=caption,
            )

            if self.logSegmap and "dice" in res:
                self.log("dice/%s" % caption, res["dice"].mean(), logger=True)

            for k, (p, y) in items.items():
                if y not in res:
                    continue
                p, y = res[p], res[y]

                err = 1 - self.acc(p, y)
                self.acc.reset()
                self.log(
                    "err/%s/%s" % (k, caption), err, logger=True,
                )

                if p.dim() == 2 and p.size(1) <= 2:
                    p = p[:, -1]
                if p.dim() == 1:
                    self.logger.experiment.add_pr_curve(
                        "%s/%s" % (k, caption), p, y, self.current_epoch
                    )
                    self.logger.experiment.add_histogram(
                        "distribution/%s/%s" % (k, caption), p, self.current_epoch,
                    )
                else:
                    self.logger.experiment.add_histogram(
                        "distribution/%s/%s" % (k, caption), p, self.current_epoch,
                    )

    def log_images(self, X, caption):
        # log images
        heatmap = lambda x, s: 0.7 * x + 0.1 * gray2JET(s, thresh=0.1)
        if self.logHotmap:
            M, B, mw, bw = self.net(X)
            wsum = lambda x, w: (
                x * unsqueeze_as(w / w.sum(dim=1, keepdim=True), x)
            ).sum(dim=1)
            # Now we generate CAM even if dataset is BIRADS-unannotated.
            self.logger.experiment.add_images(
                "%s/CAM malignant" % caption,
                heatmap(M, wsum(M, mw)),
                self.current_epoch,
            )
            self.logger.experiment.add_images(
                "%s/CAM BIRADs" % caption, heatmap(B, wsum(B, bw)), self.current_epoch,
            )

        if self.logSegmap:
            seg = self.net(X, classify=False).squeeze(1)
            self.logger.experiment.add_images(
                "%s/segment" % caption, heatmap(X, seg), self.current_epoch,
            )

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        X, ym, yb, detail = batch["X"], batch["Ym"], batch["Yb"], batch["detail"]
        pm, pb = self.net(X, segment=False)[-2:]
        ym_hat = pm.argmax(dim=1)

        err = 1 - self.acc(pm, ym)
        self.acc.reset()
        self.log("err/B-M/all", err, on_step=False, on_epoch=False, logger=True)

        for c, cname in enumerate(("benign", "malignant")):
            ym_mask = ym == c
            cerr = ((ym_mask * ym_hat) != c).sum() / ym_mask.sum()
            self.logger.experiment.add_class_err(
                cname, cerr, on_step=False, on_epoch=False, logger=True
            )

        for ymi, ymhi, di in zip(ym, ym_hat, detail):
            if ymi != ymhi:
                self.logger.experiment.add_detail(
                    di, "B/M", truth=ymi.item(), predict=ymhi.item()
                )

        if yb is None:
            return
        err = 1 - self.acc(pb, yb)
        self.acc.reset()
        self.log("err/BIRAD/all", err, on_step=False, on_epoch=False, logger=True)

        # TODO
        yb_hat = pb.argmax(dim=1)
        for ymi, ymhi, di in zip(yb, yb_hat, detail):
            if ymi != ymhi:
                self.logger.experiment.add_detail(
                    di, "BIRAD", truth=ymi.item(), predict=ymhi.item()
                )

