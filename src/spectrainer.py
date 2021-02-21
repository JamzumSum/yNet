"""
A trainer for toynets.

* author: JamzumSum
* create: 2021-1-13
"""
from itertools import chain
from warnings import warn

import torch
import torch.nn.functional as F
from progress.bar import Bar

from common.decorators import Batched, NoGrad
from common.loss import diceCoefficient
from common.trainer import Trainer, pl
from common.utils import ConfusionMatrix, ReduceLROnPlateau, gray2JET, unsqueeze_as
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

        support = (
            lambda feature: hasattr(self.net, "support") and feature in self.net.support
        )
        self.logHotmap = support("hotmap")
        self.adversarial = support("discriminator")
        self.logSegmap = support("segment")

    def configure_optimizers(self):
        branchgroup = [["M", "B"]]
        if self.adversarial:
            branchgroup.append(["discriminator"])
        ops, sgs = [], []
        for branches in branchgroup:
            d = self.conf.get("branch", {})
            branch_conf = {branch: d.get(branch, {}) for branch in branches}
            op_arg = {
                k: d.get("optimizer", self.op_conf) for k, d in branch_conf.items()
            }
            sg_arg = {
                k: d.get("scheduler", self.sg_conf) for k, d in branch_conf.items()
            }
            if self.sg_conf is not None:
                sg_arg = {
                    k: shallow_update(self.sg_conf, d, True) for k, d in sg_arg.items()
                }

            argls: list = self.op_name.__init__.__code__.co_varnames[
                : self.op_name.__init__.__code__.co_argcount
            ]
            if "weight_decay" in argls:
                default_decay = self.op_name.__init__.__defaults__[
                    argls.index("weight_decay") - len(argls)
                ]
                weight_decay = {
                    k: d.get("weight_decay", default_decay) for k, d in op_arg.items()
                }
            else:
                weight_decay = {k: 0.0 for k, d in op_arg.items()}

            paramdic = self.net.parameter_groups(weight_decay)
            param_group = []
            param_group_key = []
            for k, v in op_arg.items():
                d = {"params": paramdic[k]}
                d.update(v)
                v.setdefault("lr", self.op_conf.get("lr", 1e-3))
                d["initial_lr"] = v["lr"]
                param_group.append(d)
                param_group_key.append(k)
                if weight_decay[k]:
                    d = {"params": paramdic[k + "_no_decay"]}
                    d.update(v)
                    d["initial_lr"] = v["lr"]
                    d["weight_decay"] = 0.0
                    param_group.append(d)
                    param_group_key.append(k)

            optimizer = self.op_name(param_group, **self.op_conf)
            if self.training.get("continue", True):
                optimizer.load_state_dict(self._op_state_dict[tuple(branches)])
            ops.append(optimizer)

            if not any(sg_arg.values()):
                continue

            scheduler = ReduceLROnPlateau(
                optimizer, [sg_arg[k] for k in param_group_key]
            )
            scheduler.setepoch(self.cur_epoch)
            sgs.append(scheduler)
        return ops, sgs

    def training_step(self, batch, op, opid, name="", barstep=None):
        X, Ym, Yb, mask = batch["X"], batch["Ym"], batch["Yb"], batch["mask"]
        N = Ym.size(0)

        if opid == 0:
            if self.cur_epoch < self.discardYbEpoch:
                Yb = None
            res, loss, summary = self.net.lossWithResult(X, Ym, Yb, mask, self.piter,)

            self.total_batch += 1
            self.logSummary(name, summary, self.total_batch)
            self.buf = res

            if self.flood > 0:
                loss = (loss - self.flood).abs() + self.flood

        elif opid == 1:
            res = self.buf[0][-2:]
            loss = self.net.discriminatorLoss(*res, Ym, Yb, piter=self.piter)

        loss.backward()
        op.step()
        op.zero_grad()
        barstep(N)
        return (loss,)

    def validation_step(self, batch, barstep):
        X, Ym, Yb, mask = batch["X"], batch["Ym"], batch["Yb"], batch["mask"]
        N = Ym.size(0)
        if self.cur_epoch < self.discardYbEpoch:
            Yb = None
        loss, _ = self.net.loss(X, Ym, Yb, mask, self.piter,)
        barstep(N)
        return (loss,)

    def train(self, td, vd, no_aug=None):
        if no_aug is None:
            no_aug = td
        # get all loaders
        loader = FixLoader(td, device=self.device, **self.dataloader["training"])
        vloader = FixLoader(vd, device=self.device, **self.dataloader["training"])

        # get all optimizers and schedulers
        ops, sgs = self.configure_optimizers()
        if self.adversarial:
            mop, dop = ops
        else:
            (mop,) = ops
        # init tensorboard logger
        self.prepareBoard()
        # lemme see who dare to use cpu for training ???
        if self.device.type == "cpu":
            warn("You are using CPU for training ಠಿ_ಠ")
        else:
            assert torch.cuda.is_available()
            torch.cuda.reset_peak_memory_stats()
            # torch.backends.cudnn.benchmark = True

        trainWithDataset = Batched(self.training_step)

        if self.cur_epoch == 0:
            demo = first(loader)["X"][:2]
            self.logger.add_graph(self.net, demo)
            del demo

        if self.cur_epoch < 5:
            warmlambda = lambda epoch: min(1, 0.2 * (epoch + 1))
            warmsg = [torch.optim.lr_scheduler.LambdaLR(op, warmlambda) for op in ops]
            if self.cur_epoch == 0:
                # zero grad step to avoid warning. :(
                warmsg.optimizer.zero_grad()
                warmsg.optimizer.step()

        for self.cur_epoch in range(self.cur_epoch, self.max_epoch):
            # warm-up lr
            if self.cur_epoch < 5:
                [wsg.step() for wsg in warmsg]
            elif self.cur_epoch == 5:
                del warmsg, warmlambda
                for sg in sgs:
                    sg.setepoch(5)

            # set training flag
            self.net.train()

            with Bar(
                "epoch T%03d" % self.cur_epoch, max=(len(td) << int(self.adversarial)),
            ) as bar:
                trainWithDataset(
                    loader,
                    op=mop,
                    opid=0,
                    name="ourset",
                    barstep=lambda s: bar.next(s),
                )
                if self.adversarial:
                    (dll,) = trainWithDataset(
                        loader, op=dop, opid=1, barstep=lambda s: bar.next(s),
                    )
                    dll = dll.mean()

            # fatal: set eval when testing and validating
            self.net.eval()
            with Bar("epoch V%03d" % self.cur_epoch, max=len(vd)) as bar:
                (vll,) = Batched(self.validation_step)(
                    vloader, barstep=lambda s: bar(s)
                )
            vll = vll.mean()

            for i, sg in enumerate(sgs):
                if self.cur_epoch >= 5:
                    sg.step(dll if i == 1 else vll)

            self.score("trainset", no_aug)
            merr = self.score("validation", vd)[0]  # validation score
            self.traceNetwork()

            opdic = {("M", "B"): mop}
            if self.adversarial:
                opdic["discriminator"] = dop

            if merr < self.best_mark:
                self.best_mark = merr.item()
                self.save("best", opdic)
            self.save("latest", opdic)

    def test_step(self, batch: dict, barstep=None):
        X, Ym, Yb, mask = batch["X"], batch["Ym"], batch["Yb"], batch["mask"]
        seg, embed, Pm, Pb = self.net(X)
        barstep(len(X))

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
        return res

    @NoGrad
    def score(self, caption, dataset):
        """
        Score and evaluate the given dataset.
        """
        self.net.eval()

        loader = FixLoader(dataset, device=self.device, **self.dataloader["scoring"])
        bar = Bar("score %s%03d" % ('', self.cur_epoch), max=len(dataset) + 2)
        barstep = lambda s: bar.next(s)
        res: dict = Batched(self.test_step)(loader, barstep=barstep)

        self.logger.add_embedding(
            res["c"],
            metadata=res["ym"].tolist(),
            global_step=self.cur_epoch,
            tag=caption,
        )

        errl = []
        items = {
            "B-M": ("pm", "ym"),
            "BIRAD": ("pb", "yb"),
            "discrim": ("cy", "cy-GT"),
        }
        for k, (p, y) in items.items():
            if y not in res:
                continue
            p, y = res[p], res[y]

            acc = pl.metrics.Accuracy()
            err = 1 - acc(p, y)
            self.logger.add_scalar("err/%s/%s" % (k, caption), err, self.cur_epoch)
            errl.append(err)

            if p.dim() == 1:
                self.logger.add_pr_curve("%s/%s" % (k, caption), p, y, self.cur_epoch)
                self.logger.add_histogram(
                    "distribution/%s/%s" % (k, caption), p, self.cur_epoch
                )
            elif p.dim() == 2 and p.size(1) <= 2:
                self.logger.add_pr_curve(
                    "%s/%s" % (k, caption), p[:, -1], y, self.cur_epoch
                )
                self.logger.add_histogram(
                    "distribution/%s/%s" % (k, caption), p[:, -1], self.cur_epoch
                )
            else:
                self.logger.add_histogram(
                    "distribution/%s/%s" % (k, caption), p, self.cur_epoch
                )

        if not (self.logSegmap or self.logHotmap):
            barstep(2)
            bar.finish()
            return errl
        else:
            barstep(1)

        # log images
        X = first(loader)["X"][:8].to(self.device)
        heatmap = lambda x, s: 0.7 * x + 0.1 * gray2JET(s, thresh=0.1)
        if self.logHotmap:
            M, B, mw, bw = self.net(X)
            wsum = lambda x, w: (
                x * unsqueeze_as(w / w.sum(dim=1, keepdim=True), x)
            ).sum(dim=1)
            # Now we generate CAM even if dataset is BIRADS-unannotated.
            self.logger.add_images(
                "%s/CAM malignant" % caption, heatmap(M, wsum(M, mw)), self.cur_epoch
            )
            self.logger.add_images(
                "%s/CAM BIRADs" % caption, heatmap(B, wsum(B, bw)), self.cur_epoch
            )

        if self.logSegmap:
            self.logger.add_scalar(
                "dice/%s" % caption, res["dice"].mean(), self.cur_epoch
            )
            X = first(loader)["X"][:8].to(self.device)
            seg = self.net(X, segment=True)[0].squeeze(1)
            self.logger.add_images(
                "%s/segment" % caption, heatmap(X, seg), self.cur_epoch
            )

        barstep(1)
        bar.finish()
        return errl
