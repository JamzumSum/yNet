from dataclasses import dataclass
from typing import Iterable

import torch
from torchmetrics import ConfusionMatrix
from collections import defaultdict

argmax = lambda l: l.index(max(l))

BIRAD_MAP = ['2', '3', '4', '5']


def _lbm():
    global BIRAD_MAP
    BIRAD_MAP = torch.load("./data/BIRADs/meta.pt")['classname']['Yb']
    # i2a = defaultdict(list)
    # for k, v in a2i.items():
    #     i2a[v].append(k)
    # del a2i
    # for k, v in i2a:
    #     v = ','.join(v)
    #     i2a[k] = '4' if v == '4a,4b,4c' else v
    # BIRAD_MAP = i2a


@dataclass(frozen=True)
class DiagBag:
    pid: str
    pm: float
    pb: list
    ym: int
    yb: int

    @staticmethod
    def header():
        return [
            'pid', 'malignant prob', 'BIRADs prob distrib', 'malignant anno',
            'BIRADs anno'
        ]

    def __iter__(self):
        yield self.pid
        yield f"{self.pm:.4f}"
        yield '-' if self.pb is None else f"{BIRAD_MAP[argmax(self.pb)]}类 ({', '.join('%.4f' % i for i in self.pb)})"
        yield str(self.ym)
        yield '-' if self.yb is None else f"{BIRAD_MAP[self.yb]}类"


class Counter:
    def __init__(self, diags: Iterable[DiagBag], thresh: float) -> None:
        self.raw = tuple(diags)
        self.K = len(BIRAD_MAP)
        self.allInOne(thresh)

    def allInOne(self, thresh):
        cm = ConfusionMatrix(2, threshold=thresh)
        cb = ConfusionMatrix(self.K)

        for d in self.raw:
            cm.update(preds=torch.Tensor([d.pm]), target=torch.LongTensor([d.ym]))
            if d.yb is not None:
                cb.update(preds=torch.Tensor([d.pb]), target=torch.LongTensor([[d.yb]]))

        self.cm = cm.compute()
        self.cb = cb.compute()

    @staticmethod
    def _acc(cf):
        return float(cf.diag().sum() / cf.sum())

    @staticmethod
    def _prec(cf: torch.Tensor):
        return (cf.diag() / cf.sum(dim=1).clamp_min_(1e-5)).tolist()

    @staticmethod
    def _recall(cf: torch.Tensor):
        return (cf.diag() / cf.sum(dim=0).clamp_min_(1e-5)).tolist()

    @property
    def pb_acc(self):
        return self._acc(self.cb)

    @property
    def pm_acc(self):
        return self._acc(self.cm)

    @property
    def pb_precision(self):
        return self._prec(self.cb)

    @property
    def pb_recall(self):
        return self._recall(self.cb)

    @property
    def pm_precision(self):
        return self._prec(self.cm)

    @property
    def pm_recall(self):
        return self._recall(self.cm)


_lbm()
