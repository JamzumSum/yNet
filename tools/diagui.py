import os
import sys
from dataclasses import dataclass
from itertools import chain
from PySide2.QtCore import QFile, QUrl
from PySide2.QtGui import QBrush, QColor, QDesktopServices
from PySide2.QtWidgets import QAbstractItemView, QFileDialog, QInputDialog, QTableWidgetItem
import yaml

from qtpy.QtWidgets import QApplication, QMainWindow

from tools.compile.diagui import Ui_WndMain

argmax = lambda l: l.index(max(l))
BIRAD_MAP = ['2', '3', '4', '5']


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
    filter = False

    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
        self.ui = Ui_WndMain()
        self.ui.setupUi(self)
        self.ui.retranslateUi(self)

        self.solver = PIDSolver()
        self.__post_init__()

    def __post_init__(self):
        self.ui.tblDiag.setEditTriggers(QAbstractItemView.NoEditTriggers)

    def configOpened(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Select raw result', './log', "raw results (*.yml *.yaml)"
        )
        if not path: return
        with open(path) as f:
            self.raw = {k: DiagBag(**v) for k, v in yaml.safe_load(f).items()}

        self.conf_name = os.path.split(path)[-1]
        self.setWindowTitle(f'DiagUI: {self.conf_name}')
        self.refreshTable()

    def refreshTable(self):
        self.filter = False
        self.ui.tblDiag.clear()
        self.ui.tblDiag.setHorizontalHeaderLabels(DiagBag.header())
        self.ui.tblDiag.setRowCount(len(self.raw))
        for i, d in enumerate(self.raw.values()):
            for c, s in enumerate(d):
                self.ui.tblDiag.setItem(i, c, QTableWidgetItem(s))
        self.updateStat()
        self.ui.tblDiag.resizeColumnsToContents()

    def updateStat(self):
        red = QBrush(QColor(255, 0, 0))
        for i in range(len(self.raw)):
            d = self.raw[self.ui.tblDiag.item(i, 0).text()]
            if int(d.pm > self.thresh) != d.ym:
                it = self.ui.tblDiag.item(i, 1)
                it.setForeground(red)
            if d.yb is not None and argmax(d.pb) != d.yb:
                it = self.ui.tblDiag.item(i, 2)
                it.setForeground(red)

    def setThreshold(self):
        t, ok = QInputDialog.getDouble(
            self, 'New threshold', '', self.thresh, minValue=0., maxValue=1.
        )
        if not ok: return
        self.thresh = t
        if hasattr(self, 'raw') and self.raw:
            self.updateStat()

    def errorFilter(self, enable):
        self.filter = enable
        n = 0
        for i in range(len(self.raw)):
            d = self.raw[self.ui.tblDiag.item(i, 0).text()]
            if int(d.pm > self.thresh) == d.ym and (d.yb is None
                                                    or argmax(d.pb) == d.yb):
                self.ui.tblDiag.setRowHidden(i, enable)
                n += 1
        self.ui.statusbar.showMessage(f'{len(self.raw) - (n if enable else 0)} in all')

    def viewPID(self, row, col):
        pid = self.ui.tblDiag.item(row, 0).text()
        path = self.solver.pid2path(pid)
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def exportMarkdown(self):
        dir = QFileDialog.getExistingDirectory(self, 'Select output folder', './doc')
        path = os.path.join(dir, 'summary.md')
        assets = os.path.join(dir, 'assets')
        os.makedirs(assets, exist_ok=True)

        line = lambda it: f"|{'|'.join(it)}|\n"

        with open(path, 'w', encoding='utf8') as f:
            f.write('## Error Summary\n\n')
            f.write(f'> Auto generated from {self.conf_name}.\n\n')
            # table title
            f.write(line(DiagBag.header()))
            f.write(line([':-:'] * len(DiagBag.header())))
            # content
            for i in range(len(self.raw)):
                d = self.raw[self.ui.tblDiag.item(i, 0).text()]
                if int(d.pm > self.thresh) != d.ym or (d.yb is not None
                                                       and argmax(d.pb) != d.yb):
                    pic = self.solver.pid2path(d.pid)
                    QFile.copy(pic, os.path.join(assets, os.path.split(pic)[-1]))
                    f.write(line(self.rich(d)))

        self.ui.statusbar.showMessage(f'{path} saved.', 3000)

    def rich(self, d: DiagBag):
        it = list(d)
        path = self.solver.pid2path(it[0])
        fname = os.path.split(path)[-1]
        it[0] = f'<a href="./assets/{fname}">{d.pid}</a>'

        if int(d.pm > self.thresh) != d.ym:
            it[1] = f'<p style="color: red">{it[1]}</p>'
        if d.yb is not None and argmax(d.pb) != d.yb:
            it[2] = f'<p style="color: red">{it[2]}</p>'
        return it


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = DiaGUI()
    gui.show()
    sys.exit(app.exec_())
