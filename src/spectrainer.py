"""
A trainer for toynets.

* author: JamzumSum
* create: 2021-1-13
"""
from itertools import chain
from warnings import warn

import torch
import torch.nn.functional as F

from common.decorators import Batched, NoGrad
from common.loss import diceCoefficient
from common.trainer import Trainer
from common.utils import ConfusionMatrix, gray2JET, unsqueeze_as, ReduceLROnPlateau
from data.dataloader import FixLoader
from utils.dict import shallow_update

lenn = lambda x: len(x) if x else 0
first = lambda it: next(iter(it))


class ToyNetTrainer(Trainer):
    def __init__(self, Net, conf):
        Trainer.__init__(self, Net, conf)
        self.sg_conf = self.conf.get("scheduler", {})
        self.discardYbEpoch = self.training.get("use_annotation_from", self.max_epoch)
        self.flood = self.training.get("flood", 0)

    def support(self, feature: str):
        return hasattr(self.net, "support") and feature in self.net.support

    @property
    def logHotmap(self):
        return self.support("hotmap")

    @property
    def adversarial(self):
        return self.support("discriminator")

    @property
    def logSegmap(self):
        return self.support("segment")

    def getBranchOptimizerAndScheduler(self, branches: list):
        """
        branch: 
        - default
        - M, B
        - discriminator
        """
        if self.adversarial:
            branches.append("discriminator")
        d = self.conf.get("branch", {})
        branch_conf = {branch: d.get(branch, {}) for branch in branches}
        op_arg = {
            k: d.get("optimizer", self.op_conf[1]) for k, d in branch_conf.items()
        }
        sg_arg = {k: d.get("scheduler", self.sg_conf) for k, d in branch_conf.items()}
        if self.sg_conf is not None:
            sg_arg = {
                k: shallow_update(self.sg_conf, d, True) for k, d in sg_arg.items()
            }
        if "default" in branches:
            d = op_arg.pop("default")
            op_arg["M"], op_arg["B"] = d, d
            d = sg_arg.pop("default")
            sg_arg["M"], sg_arg["B"] = d, d

        OP: torch.optim.Optimizer = getattr(torch.optim, self.op_conf[0])
        weight_decay = {k: d.get("weight_decay", 0) for k, d in op_arg.items()}

        paramdic = self.net.parameter_groups(weight_decay)
        param_group = []
        param_group_key = []
        for k, v in op_arg.items():
            d = {"params": paramdic[k]}
            d.update(v)
            d["initial_lr"] = v.get("lr", 1e-3)
            param_group.append(d)
            param_group_key.append(k)
            if weight_decay[k]:
                d = {"params": paramdic[k + "_no_decay"]}
                d.update(v)
                d["initial_lr"] = v.get("lr", 1e-3)
                d.pop("weight_decay")
                param_group.append(d)
                param_group_key.append(k)

        optimizer = OP(param_group, **self.op_conf[1])

        if not any(sg_arg.values()):
            return optimizer, None

        scheduler = ReduceLROnPlateau(optimizer, [sg_arg[k] for k in param_group_key])
        return optimizer, scheduler

    def trainInBatch(
        self, X, Ym, Yb=None, mask=None, name="", op=None, dop=None, barstep=None
    ):
        if dop and not op:
            raise ValueError

        N = Ym.shape[0]
        res, loss, summary = self.net.lossWithResult(
            X,
            Ym,
            None if self.cur_epoch < self.discardYbEpoch else Yb,
            mask,
            self.piter,
        )
        if name:
            self.total_batch += 1
            self.logSummary(name, summary, self.total_batch)
        if op:
            if self.flood > 0:
                loss = (loss - self.flood).abs() + self.flood
            loss.backward()
            op.step()
            op.zero_grad()
        barstep(N)
        if dop:
            res = res[0][-2:]
            dloss = self.net.discriminatorLoss(*res, Ym, Yb, piter=self.piter)
            dloss.backward()
            dop.step()
            dop.zero_grad()
            barstep(N)
            return loss.detach_(), dloss.detach_()
        return (loss.detach_(),)

    def train(self, td, vd, no_aug=None):
        if no_aug is None:
            no_aug = td
        # get all loaders
        loader = FixLoader(td, device=self.device, **self.dataloader["training"])
        vloader = FixLoader(vd, device=self.device, **self.dataloader["training"])

        # get all optimizers and schedulers
        gop, gsg = self.getBranchOptimizerAndScheduler(["default"])
        mop, msg = self.getBranchOptimizerAndScheduler(["M", "B"])
        if self.adversarial:
            dop, dsg = self.getBranchOptimizerAndScheduler(["discriminator"])
        else:
            dop = dsg = None
        # init tensorboard logger
        self.prepareBoard()
        # lemme see who dare to use cpu for training ???
        if self.device.type == "cpu":
            warn("You are using CPU for training ಠಿ_ಠ")
        else:
            assert torch.cuda.is_available()
            torch.cuda.reset_peak_memory_stats()

        trainWithDataset = Batched(self.trainInBatch)

        if self.cur_epoch == 0:
            demo = first(loader)["X"][:2]
            self.board.add_graph(self.net, demo)
            del demo

        if self.cur_epoch < 5:
            warmlambda = lambda epoch: min(1, 0.2 * (epoch + 1))
            warmsg = torch.optim.lr_scheduler.LambdaLR(
                gop if self.cur_epoch < self.discardYbEpoch else mop, warmlambda
            )

        for self.cur_epoch in range(self.cur_epoch, self.max_epoch):
            # change optimizer and lr warmer
            if self.cur_epoch == self.discardYbEpoch:
                if self.cur_epoch < 5:
                    warmsg = torch.optim.lr_scheduler.LambdaLR(
                        gop if self.cur_epoch < self.discardYbEpoch else mop,
                        warmlambda,
                        self.cur_epoch,
                    )
                print("Train b-branch from epoch%03d." % self.discardYbEpoch)
                print("Optimizer is seperated from now.")
            # warm-up lr
            if self.cur_epoch < 5:
                warmsg.step()
            elif self.cur_epoch == 5:
                del warmsg, warmlambda
            # set training flag
            self.net.train()

            tid = self.progress.add_task(
                "epoch T%03d" % self.cur_epoch, total=(len(td) << int(self.adversarial))
            )
            res = trainWithDataset(
                loader,
                name="ourset",
                op=gop if self.cur_epoch < self.discardYbEpoch else mop,
                dop=dop,
                barstep=lambda s: self.progress.update(tid, advance=s),
            )
            self.progress.stop_task(tid)
            if self.adversarial:
                tll, dll = res
                dll = dll.mean()
            else:
                (tll,) = res
            tll = tll.mean()

            tid = self.progress.add_task("epoch V%03d" % self.cur_epoch, total=len(vd))
            vll = trainWithDataset(
                vloader, barstep=lambda s: self.progress.update(tid, advance=s)
            )
            vll = torch.cat(vll).mean()
            self.progress.stop_task(tid)
            ll = (tll + vll) / 2

            if dsg:
                dsg.step(dll)
            if self.cur_epoch < self.discardYbEpoch:
                if self.cur_epoch <= 5 and gsg:
                    gsg.step(ll)
            else:
                if self.cur_epoch <= 5 and msg:
                    msg.step(ll)

            merr = self.score("trainset", no_aug)[0]  # trainset score
            merr = self.score("validation", vd)[0]  # validation score
            self.traceNetwork()

            if merr < self.best_mark:
                self.best_mark = merr
                self.save("best", merr)
            self.save("latest", merr)

    @NoGrad
    def score(self, caption, dataset, thresh=0.5):
        """
        Score and evaluate the given dataset.
        """
        self.net.eval()
        tid = self.progress.add_task("evaluating...", total=3)

        def scoresInBatch(X, Ym, Yb=None, mask=None):
            seg, embed, Pm, Pb = self.net(X)
            guide_pm, guide_pb = self.net(X, mask=seg)

            res = {
                "pm": Pm,
                "pb": Pb,
                "gpm": guide_pm,
                "gpb": guide_pb,
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
            return res

        loader = FixLoader(dataset, device=self.device, **self.dataloader["scoring"])
        res: dict = Batched(scoresInBatch)(loader)
        self.progress.update(tid, completed=1)

        self.board.add_embedding(
            res["c"],
            metadata=res["ym"].tolist(),
            global_step=self.cur_epoch,
            tag=caption,
        )

        errl = []
        items = {
            "B-M": ("pm", "ym"),
            "Guided B-M": ("gpm", "ym"),
            "BIRAD": ("pb", "yb"),
            "discrim": ("cy", "cy-GT"),
        }
        for k, (p, y) in items.items():
            if y not in res:
                continue
            p, y = res[p], res[y]

            if p.dim() == 1:
                argp = (p > thresh).int()
            elif p.dim() == 2:
                argp = p.argmax(dim=1)
            err = (argp != y).float().mean()
            self.board.add_scalar("err/%s/%s" % (k, caption), err, self.cur_epoch)
            self.board.add_histogram(
                "distribution/%s/%s" % (k, caption), p, self.cur_epoch
            )
            errl.append(err)

            if p.dim() == 1:
                self.board.add_pr_curve("%s/%s" % (k, caption), p, y, self.cur_epoch)
            elif p.dim() == 2 and p.size(1) <= 2:
                self.board.add_pr_curve(
                    "%s/%s" % (k, caption), p[:, -1], y, self.cur_epoch
                )

        if not (self.logSegmap or self.logHotmap):
            self.progress.update(tid, completed=3)
            self.progress.stop_task(tid)
            return errl
        else:
            self.progress.update(tid, completed=2)

        # log images
        # BUG: since datasets are sorted, images extracted are biased. (Bengin & BIRADs-2)
        # EMMM: but I don't know which image to sample, cuz if images are not fixed,
        # we cannot compare through epochs... and also, since scoring is not testing,
        # extracting (nearly) all mis-classified samples is excessive...
        X = first(loader)["X"][:8].to(self.device)
        heatmap = lambda x, s: 0.7 * x + 0.1 * gray2JET(s, thresh=0.1)
        if self.logHotmap:
            M, B, mw, bw = self.net(X)
            wsum = lambda x, w: (
                x * unsqueeze_as(w / w.sum(dim=1, keepdim=True), x)
            ).sum(dim=1)
            # Now we generate CAM even if dataset is BIRADS-unannotated.
            self.board.add_images(
                "%s/CAM malignant" % caption, heatmap(M, wsum(M, mw)), self.cur_epoch
            )
            self.board.add_images(
                "%s/CAM BIRADs" % caption, heatmap(B, wsum(B, bw)), self.cur_epoch
            )

        if self.logSegmap:
            self.board.add_scalar(
                "dice/%s" % caption, res["dice"].mean(), self.cur_epoch
            )
            X = first(loader)["X"][:8].to(self.device)
            seg = self.net(X, segment=True)[0].squeeze(1)
            self.board.add_images(
                "%s/segment" % caption, heatmap(X, seg), self.cur_epoch
            )

        self.progress.update(tid, completed=3)
        self.progress.stop_task(tid)
        return errl
