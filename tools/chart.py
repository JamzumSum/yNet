from PySide2.QtCore import QEasingCurve
from PySide2.QtGui import QColor
from PySide2.QtWidgets import QSizePolicy
from qtpy.QtCharts import QtCharts
import torch

from tools.common import BIRAD_MAP

COLOR = ['#6abfea', '#99ca53', '#f6a625', '#6abfea', '#30401a']


class BaseChartView(QtCharts.QChartView):
    def __init__(self, title: str, axis: list) -> None:
        chart = QtCharts.QChart()
        chart.setTitle(title)
        chart.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        chart.setMinimumHeight(560)

        chart.setAnimationOptions(QtCharts.QChart.SeriesAnimations)
        chart.setAnimationDuration(500)
        chart.setAnimationEasingCurve(QEasingCurve.OutCubic)

        super().__init__(chart)

        self.xa = QtCharts.QBarCategoryAxis()
        self.xa.append(axis)

        self.setVisible(False)
        # self.setSize


class PRBarChartView(BaseChartView):
    def __init__(self, title: str, axis: list) -> None:
        super().__init__(title, axis)
        self.ya = QtCharts.QValueAxis()
        self.ya.setRange(0, 1)

    def refresh(self, p, r):
        pre = QtCharts.QBarSet('precision')
        rec = QtCharts.QBarSet('recall')
        pre.append(p)
        rec.append(r)
        pre.setLabelColor(QColor('#000000'))
        rec.setLabelColor(QColor('#000000'))

        series = QtCharts.QBarSeries()
        series.setLabelsVisible(True)
        series.setLabelsPosition(QtCharts.QAbstractBarSeries.LabelsOutsideEnd)
        series.setLabelsPrecision(4)
        series.append(pre)
        series.append(rec)

        self.chart().removeAllSeries()
        self.chart().addSeries(series)
        self.chart().setAxisX(self.xa, series)
        self.chart().setAxisY(self.ya, series)

        self.setVisible(True)


class FallacyBarView(BaseChartView):
    def __init__(self, title: str, axis: list) -> None:
        super().__init__(title, ['benign', 'malignant'])
        self.axis = axis
        self.ya = QtCharts.QValueAxis()

    @property
    def K(self):
        return len(self.axis)

    def refresh(self, B, M):
        sets = [QtCharts.QBarSet(i) for i in self.axis]
        for s, b, m in zip(sets, B, M):
            s.append([b, m])
            s.setLabelColor(QColor('black'))

        series = QtCharts.QBarSeries()
        series.setLabelsVisible(True)
        series.setLabelsPosition(QtCharts.QAbstractBarSeries.LabelsOutsideEnd)
        series.append(sets)

        self.chart().removeAllSeries()
        self.chart().addSeries(series)
        self.chart().setAxisX(self.xa, series)
        self.chart().setAxisY(self.ya, series)
        self.ya.setRange(0, max(B + M))

        self.setVisible(True)


class CMPieChartView(BaseChartView):
    def __init__(self, title: str, axis: list) -> None:
        super().__init__(title, axis)
        self.chart().legend().hide()

    def refresh(self, cm: torch.LongTensor):
        K = len(cm)
        color = [QColor(j) for j in COLOR[:K]]
        self.chart().removeAllSeries()

        for i in range(K):
            series = QtCharts.QPieSeries()
            series.setName(f'{i}')
            series.setHorizontalPosition((i + 0.5) / K)
            series.setPieSize(1 / K)

            for j in range(K):
                if (v := cm[i, j]):
                    series.append(f"pred as {BIRAD_MAP[j]}", v)
                    series.slices()[-1].setColor(color[j])

            series.setLabelsVisible(True)
            self.chart().addSeries(series)
            self.chart().setAxisX(self.xa, series)

        self.setVisible(True)
