from PySide2.QtGui import QBrush, QColor
from PySide2.QtWidgets import QTableWidget, QTableWidgetItem

from tools.common import DiagBag, argmax


class DiagTable(QTableWidget):
    _t = 0.5

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        header = DiagBag.header()
        self.setColumnCount(len(header))
        self.setHorizontalHeaderLabels(header)
        self.resizeColumnsToContents()

    def _refreshDiag(self, raw: dict[str, DiagBag], thresh):
        self.filter = False
        self.clear()
        self.setHorizontalHeaderLabels(DiagBag.header())
        self.setRowCount(len(raw))
        for i, d in enumerate(raw.values()):
            for c, s in enumerate(d):
                self.setItem(i, c, QTableWidgetItem(s))
        self._updateStat(raw, thresh)
        self.resizeColumnsToContents()

    def _updateStat(self, raw, thresh):
        red = QBrush(QColor(255, 0, 0))
        for i in range(len(raw)):
            d = raw[self.item(i, 0).text()]
            if int(d.pm > thresh) != d.ym:
                it = self.item(i, 1)
                it.setForeground(red)
            if d.yb is not None and argmax(d.pb) != d.yb:
                it = self.item(i, 2)
                it.setForeground(red)

    @property
    def raw(self):
        return self._raw

    @raw.setter
    def raw(self, value):
        self._raw = value
        self._refreshDiag(self._raw, self._t)

    @property
    def thresh(self):
        return self._t

    @thresh.setter
    def thresh(self, value):
        self._t = value
        if hasattr(self, '_raw'): self._updateStat(self.raw, value)