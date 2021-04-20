import os
import sys
from dataclasses import dataclass
from itertools import chain
from PySide2.QtCore import QUrl
from PySide2.QtGui import QDesktopServices, QKeySequence
from PySide2.QtWidgets import QAbstractItemView, QFileDialog, QHeaderView, QInputDialog, QTableWidgetItem
import yaml

from qtpy.QtWidgets import QApplication, QMainWindow

from tools.compile.diagui import Ui_WndMain


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
        yield '%.4f' % self.pm
        yield '-' if self.pb is None else ', '.join('%.4f' % i for i in self.pb)
        yield str(self.ym)
        yield '-' if self.yb is None else str(self.yb)


class PIDSolver:
    suffix = ('.jpg', '.png', '.bmp')
    sets = ['BUSI', 'BIRADs', 'set2', 'set3']

    def __init__(self) -> None:
        gen = (os.walk(f'./data/{i}/raw') for i in self.sets)
        fgen = chain([(i, os.path.join(root, i)) for i in files] for walk in chain(gen)
                     for root, _, files in walk)
        fgen = sum(fgen, [])
        fgen = filter(lambda p: any(p[0].endswith(ext) for ext in self.suffix), fgen)
        self.map = {os.path.splitext(n)[0]: p for n, p in fgen}

    def pid2path(self, pid: str) -> str:
        return self.map[pid]


class DiaGUI(QMainWindow):
    thresh = 0.5

    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
        self.ui = Ui_WndMain()
        self.ui.setupUi(self)
        self.ui.retranslateUi(self)

        self.solver = PIDSolver()
        self.__post_init__()

    def __post_init__(self):
        self.ui.tblDiag.setEditTriggers(QAbstractItemView.NoEditTriggers)

        self.ui.actionOpen.setShortcut(QKeySequence.Open)
        self.ui.actionThreshold.setShortcut(QKeySequence.AddTab)

    def configOpened(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Select raw result', './log', "raw results (*.yml *.yaml)"
        )
        if not path: return
        with open(path) as f:
            d: dict = yaml.safe_load(f)
            self.raw = [DiagBag(**v) for v in d.values()]
        self.refreshTable()

    def refreshTable(self):
        self.ui.tblDiag.clear()
        self.ui.tblDiag.setHorizontalHeaderLabels(['stat'] + DiagBag.header())
        self.ui.tblDiag.setRowCount(len(self.raw))
        for i, d in enumerate(self.raw):
            for c, s in enumerate(d):
                self.ui.tblDiag.setItem(i, c + 1, QTableWidgetItem(s))
        self.updateStat()
        self.ui.tblDiag.resizeColumnsToContents()

    def updateStat(self):
        emoji = ['✖', '✔']
        for i, d in enumerate(self.raw):
            j = int(d.pm > self.thresh) == d.ym
            self.ui.tblDiag.setItem(i, 0, QTableWidgetItem(emoji[int(j)]))

    def setThreshold(self):
        t, ok = QInputDialog.getDouble(self, 'New threshold', '', self.thresh, minValue=0., maxValue=1.)
        if not ok: return
        self.thresh = t
        if hasattr(self, 'raw') and self.raw:
            self.updateStat()

    def viewPID(self, row, col):
        pid = self.ui.tblDiag.item(row, 1).text()
        path = self.solver.pid2path(pid)
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = DiaGUI()
    gui.show()
    sys.exit(app.exec_())
