"""
A trainer for toynets.

* author: JamzumSum
* create: 2021-1-13
"""
from itertools import chain
from warnings import warn

import torch
import torch.nn.functional as F

from common.loss import diceCoefficient
from common.optimizer import ReduceLROnPlateau, no_decay
from common.support import *
from common.trainer import LightningBase, pl
from common.utils import deep_merge, gray2JET, unsqueeze_as
from misc.dict import shallow_update

lenn = lambda x: len(x) if x else 0
first = lambda it: next(iter(it))


plateau_lr_dic = lambda sg, monitor: {
    "scheduler": sg,
    "interval": "epoch",
    "reduce_on_plateau": True,
    "monitor": monitor,
}


class ToyNetTrainer(LightningBase):
    def __init__(self, Net, conf):
        LightningBase.__init__(self, Net, conf)
        self.sg_conf = conf.get("scheduler", {})
        self.branch_conf = conf.get("branch", {})

        self.discardYbEpoch = self.misc.get("use_annotation_from", 0)
        self.flood = self.misc.get("flood", 0)

        self.logHotmap = isinstance(self.net, HeatmapSupported)
        self.adversarial = isinstance(self.net, HasDiscriminator)
        self.logSegmap = isinstance(self.net, SegmentSupported)

        self.example_input_array = torch.randn((2, 1, 512, 512))
        self.acc = pl.metrics.Accuracy()

    @property
    def default_weight_decay(self):
        argls: list = self.op_name.__init__.__code__.co_varnames[
            : self.op_name.__init__.__code__.co_argcount
        ]
        if "weight_decay" in argls:
            return self.op_name.__init__.__defaults__[
                argls.index("weight_decay") - len(argls)
            ]
        else:
            return

    def forward(self, X):
        return self.net(X)

    def configure_optimizers(self):
        branches = ["M", "B"]
        if self.adversarial:
            branches.append("discriminator")
        op_arg, sg_arg = {}, {}
        for k in branches:
            d = self.branch_conf.get(k, {})
            op_arg[k] = d.get("optimizer", self.op_conf)
            sg_arg[k] = shallow_update(
                self.sg_conf, d.get("scheduler", self.sg_conf), True
            )

        default_decay = self.default_weight_decay
        weight_decay = {
            k: d.get("weight_decay", default_decay) for k, d in op_arg.items()
        }
        paramdic = self.net.parameter_groups(weight_decay)
        paramdic, param_group_key = no_decay(
            weight_decay, paramdic, op_arg, self.op_conf.get("lr", 1e-3)
        )

        mop_param_key = ["M", "M_no_decay", "B", "B_no_decay"]
        mop_param = [paramdic[k] for k in mop_param_key if k in paramdic]
        mop = self.op_name(mop_param, **self.op_conf)
        ops = [mop]
        if self.adversarial:
            dop_param = [v for k, v in paramdic.items() if k not in mop_param_key]
            ops.append(self.op_name(dop_param, **self.op_conf))

        if not any(sg_arg.values()):
            return ops

        msg_param_key = ["M", "B"]
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

    def training_step(self, batch, batch_idx: int, opid=0):
        if opid == 0:
            X, Ym, Yb, mask = batch["X"], batch["Ym"], batch["Yb"], batch["mask"]
            if self.current_epoch < self.discardYbEpoch:
                Yb = None
            res, loss, summary = self.net.lossWithResult(X, Ym, Yb, mask, self.piter,)

            self.log(
                "train_loss",
                loss,
                on_step=False,
                on_epoch=True,
                logger=False,
                prog_bar=True,
            )
            self.log_dict(summary)
            self.buf = res

            if self.flood > 0:
                loss = (loss - self.flood).abs() + self.flood

        elif opid == 1:
            res = self.buf[0][-2:]
            loss = self.net.discriminatorLoss(*res, Ym, Yb, piter=self.piter)
            self.log("dis_loss", loss, on_step=False, on_epoch=True, logger=False)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 1:
            return
        X, Ym, Yb, mask = batch["X"], batch["Ym"], batch["Yb"], batch["mask"]

        if self.current_epoch < self.discardYbEpoch:
            Yb = None
        loss, _ = self.net.loss(X, Ym, Yb, mask, self.piter,)
        self.buf = loss
        self.log("val_loss", loss, prog_bar=True)

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
            self.log_images(X[:8], self.test_caption[dataloader_idx])
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
            res = deep_merge(res)
            caption = self.test_caption[dataset_idx]

            self.logger.experiment.add_embedding(
                res["c"],
                metadata=res["ym"].tolist(),
                global_step=self.current_epoch,
                tag=self.test_caption[dataset_idx],
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
            seg = self.net(X, segment=True)[0].squeeze(1)
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

