import os
import sys
from itertools import chain

import yaml
from PySide2.QtCore import QFile, Qt, QUrl
from PySide2.QtGui import QDesktopServices
from PySide2.QtWidgets import QAbstractItemView, QFileDialog, QInputDialog
from qtpy.QtWidgets import QApplication, QMainWindow

from tools.chart import CMPieChartView, PRBarChartView
from tools.common import *
from tools.compile.diagui import Ui_WndMain

wHolder = []


class PIDSolver:
    suffix = ('jpg', 'png', 'bmp')
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

    def __init__(self, parent=None, index=0) -> None:
        super().__init__(parent=parent)
        self.ui = Ui_WndMain()
        self.ui.setupUi(self)
        self.ui.retranslateUi(self)

        self._index = index
        self.solver = PIDSolver()
        self.__post_init__()

    def __post_init__(self):
        self.ui.tblDiag.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setWindowState(Qt.WindowMaximized)
        self.ui.bcv = PRBarChartView('BI-RADS', BIRAD_MAP)
        self.ui.mcv = PRBarChartView('B/M', ['benign', 'malignant'])
        self.ui.bpie = CMPieChartView(
            'BI-RADS Precision PieChart', [f"BIRADS-{i}" for i in BIRAD_MAP]
        )
        self.ui.scrollContent.addWidget(self.ui.bcv)
        self.ui.scrollContent.addWidget(self.ui.mcv)
        self.ui.scrollContent.addWidget(self.ui.bpie)

    def configOpened(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Select raw result', filter="raw results (*.yml *.yaml)"
        )
        if not path: return
        with open(path) as f:
            raw, conf = yaml.safe_load_all(f)
            self.raw = {k: DiagBag(**v) for k, v in raw.items()}

        self.counter = Counter(self.raw.values(), self.thresh)
        conf_path = conf.get('paths', {}) if conf else {}
        self.conf_name = f"{conf_path.get('name', '')}/{conf_path.get('version', os.path.split(path)[-1])}"
        self.setWindowTitle(f"DiagUI: {self.conf_name}")
        self.refreshTable()

    def refreshTable(self):
        self.ui.tblDiag.raw = self.raw
        self.showStatistic()

    def setThreshold(self):
        t, ok = QInputDialog.getDouble(
            self, 'New threshold', '', self.thresh, minValue=0., maxValue=1.
        )
        if not ok: return
        self.thresh = t
        if hasattr(self, 'raw') and self.raw:
            self.ui.tblDiag.thresh = t
            self.counter.allInOne(t)
            self.showStatistic()

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

    def showStatistic(self):
        self.ui.bcv.refresh(self.counter.pb_precision, self.counter.pb_recall)
        self.ui.mcv.refresh(self.counter.pm_precision, self.counter.pm_recall)
        self.ui.bpie.refresh(self.counter.cb)

    def newWindow(self):
        wHolder.append(gui := DiaGUI(index=len(wHolder)))
        gui.show()

    def closeEvent(self, event=None) -> None:
        wHolder[self._index] = None
        return super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = DiaGUI()
    gui.show()
    sys.exit(app.exec_())
