from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F


def freeze(tensor, f=0.0):
    return (1 - f) * tensor + f * tensor.detach()


def unsqueeze_as(s, t, dim=-1):
    while s.dim() < t.dim():
        s = s.unsqueeze(dim)
    return s


def gray2JET(x, thresh=0.5):
    """
    - x: [..., H, W],       NOTE: float 0~1
    - O: [..., 3, H, W],    NOTE: BGR, float 0~1
    """
    x = 255 * x
    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)
    B = [
        128 + 4 * x,
        255 * ones,
        255 * ones,
        254 * ones,
        638 - 4 * x,
        ones,
        zeros,
        zeros,
    ]
    G = [
        zeros,
        zeros,
        4 * x - 128,
        255 * ones,
        255 * ones,
        255 * ones,
        892 - 4 * x,
        zeros,
    ]
    R = [
        zeros,
        zeros,
        zeros,
        2 * ones,
        4 * x - 382,
        254 * ones,
        255 * ones,
        1148 - 4 * x,
    ]
    cond = [
        x < 31,
        x == 32,
        (33 <= x) * (x <= 95),
        x == 96,
        (97 <= x) * (x <= 158),
        x == 159,
        (160 <= x) * (x <= 223),
        224 <= x,
    ]
    cond = torch.stack(cond)  # [8, :]
    B = torch.sum(torch.stack(B) * cond, dim=0)
    G = torch.sum(torch.stack(G) * cond, dim=0)
    R = torch.sum(torch.stack(R) * cond, dim=0)
    O = torch.stack([R, G, B], dim=-3) / 255
    return unsqueeze_as(x > thresh * 255, O, 1) * O


class ConfusionMatrix:
    def __init__(self, K=None, smooth=1e-8):
        if K:
            self._m = torch.zeros(K, K, dtype=torch.int)
        self.K = K
        self.eps = smooth

    @property
    def initiated(self):
        return hasattr(self, "_m")

    @property
    def N(self):
        return self._m.sum() if self.initiated else None

    def add(self, P, Y):
        """P&Y: [N]"""
        INTTYPE = (torch.int, torch.int64, torch.int16)
        assert P.dtype in INTTYPE
        assert Y.dtype in INTTYPE
        if self.K:
            K = self.K
        else:
            self.K = K = int(Y.max())

        if not self.initiated:
            self._m = torch.zeros(K, K, dtype=torch.int, device=P.device)
        if self._m.device != P.device:
            self._m = self._m.to(P.device)
        if Y.device != P.device:
            Y = Y.to(P.device)

        for i, j in product(range(K), range(K)):
            self._m[i, j] += ((P == i) * (Y == j)).sum()

    def accuracy(self):
        acc = self._m.diag().sum()
        return acc / self._m.sum()

    def err(self, *args, **kwargs):
        return 1 - self.accuracy(*args, **kwargs)

    def precision(self, reduction="none"):
        acc = self._m.diag()
        prc = acc / (self._m.sum(dim=1) + self.eps)
        if reduction == "mean":
            return prc.mean()
        else:
            return prc

    def recall(self, reduction="none"):
        acc = self._m.diag()
        rec = acc / (
            self._m.sum()
            - self._m.sum(dim=0)
            - self._m.sum(dim=1)
            + self._m.diag()
            + self.eps
        )
        if reduction == "mean":
            return rec.mean()
        else:
            return rec

    def fscore(self, beta=1, reduction="mean"):
        P = self.precision()
        R = self.recall()
        f = (1 + beta * beta) * (P * R) / (beta * beta * P + R + self.eps)
        if reduction == "mean":
            return f.mean()
        else:
            return f

    def mat(self):
        return self._m / self._m.max()


class SEBlock(nn.Sequential):
    def __init__(self, L, hs=128):
        nn.Sequential.__init__(
            self,
            nn.Linear(L, hs),
            nn.ReLU(),
            nn.Linear(hs, L),
            nn.Softmax(dim=-1)
            # use softmax instead of sigmoid here since the attention-ed channels are sumed,
            # while the sum might be greater than 1 if sum of the attention vector is not restricted.
        )
        nn.init.constant_(self[2].bias, 1 / L)

    def forward(self, X):
        """
        X: [N, K, H, W, L]
        O: [N, K, H, W]
        """
        X = X.permute(4, 0, 1, 2, 3)  # [L, N, K, H, W]
        Xp = F.adaptive_avg_pool2d(X, (1, 1))  # [L, N, K, 1, 1]
        Xp = Xp.permute(1, 2, 3, 4, 0)  # [N, K, 1, 1, L]
        Xp = nn.Sequential.forward(self, Xp).permute(4, 0, 1, 2, 3)  # [L, N, K, 1, 1]
        return (X * Xp).sum(dim=0)


class PyramidPooling(nn.Module):
    """
    Use pyramid pooling instead of max-pooling to make sure more elements in CAM can be backward. 
    Otherwise only the patch with maximum average confidence has grad while patches and small.
    Moreover, the size of patches are fixed so is hard to select. Multi-scaled patches are suitable.
    """

    def __init__(self, patch_sizes, hs=128):
        nn.Module.__init__(self)
        if any(i & 1 for i in patch_sizes):
            print(
                """Warning: At least one value in `patch_sizes` is odd. 
            Channel-wise align may behave incorrectly."""
            )
        self.patch_sizes = sorted(patch_sizes)
        self.atn = SEBlock(self.L, hs)

    @property
    def L(self):
        return len(self.patch_sizes)

    def forward(self, X):
        """
        X: [N, C, H, W]
        O: [N, K, 2 * H//P_0 -1, 2 * W//P_0 - 1]
        """
        # set stride as P/2, so that patches overlaps each other
        # hopes to counterbalance the lack-representating of edge pixels of a patch.
        ls = [
            F.avg_pool2d(X, patch_size, patch_size // 2)
            for patch_size in self.patch_sizes
        ]
        base = ls.pop(0)  # [N, K, H//P0, W//P0]
        ls = [F.interpolate(i, base.shape[-2:], mode="nearest") for i in ls]
        ls.insert(0, base)
        ls = torch.stack(ls, dim=-1)  # [N, K, H//P0, W//P0, L]
        return self.atn(ls)


class DirectLR(torch.optim.lr_scheduler._LRScheduler):
    @property
    def lr(self):
        return self.get_last_lr()

    def set_lr(self, lr):
        if not isinstance(lr, list) and not isinstance(lr, tuple):
            self._lr = [lr] * len(self.optimizer.param_groups)
        else:
            if len(lr) != len(self.optimizer.param_groups):
                raise ValueError(
                    "Expected {} lr, but got {}".format(
                        len(self.optimizer.param_groups), len(lr)
                    )
                )
            self._lr = list(lr)

    def get_lr(self):
        return self._lr


class _ReduceLROnPlateauSub(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self, idx, *args, **argv):
        self.idx = idx
        torch.optim.lr_scheduler.ReduceLROnPlateau.__init__(self, *args, **argv)

    def _reduce_lr(self, epoch):
        param_group = self.optimizer.parameter_groups[self.idx]
        old_lr = float(param_group["lr"])
        new_lr = max(old_lr * self.factor, self.min_lrs[self.idx])
        if old_lr - new_lr > self.eps:
            param_group["lr"] = new_lr
            if self.verbose:
                print(
                    "Epoch {:5d}: reducing learning rate"
                    " of group {} to {:.4e}.".format(epoch, self.idx, new_lr)
                )


class ReduceLROnPlateau:
    def __init__(self, optimizer, arglist):
        default = {
            "mode": "min",
            "factor": 0.1,
            "patience": 10,
            "threshold": 1e-4,
            "threshold_mode": "rel",
            "cooldown": 0,
            "min_lr": 0,
            "eps": 1e-8,
            "verbose": False,
        }
        self.optimizer = optimizer
        if isinstance(arglist, (list, tuple)) and len(arglist) == 1:
            arglist = arglist * len(optimizer.parameter_groups)

        ld = lambda d, i: d.get(i, default[i])
        self.sub = [
            None
            if arg is None
            else _ReduceLROnPlateauSub(
                i,
                optimizer,
                mode=ld(arg, 'mode'),
                factor=ld(arg, "factor"),
                patience=ld(arg, "patience"),
                threshold=ld(arg, 'threshold'),
                threshold_mode=ld(arg, "threshold_mode"),
                cooldown=ld(arg, "cooldown"),
                min_lr=ld(arg, "min_lr"),
                eps=ld(arg, "eps"),
                verbose=ld(arg, "verbose"),
            )
            for i, arg in enumerate(arglist)
        ]

    def step(self, metrics):
        ld = lambda l, i: l[i] if isinstance(l, (list, tuple)) else l

        for i, sg in enumerate(self.sub):
            if sg: sg.step(ld(metrics, i))

