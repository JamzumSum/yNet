# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'diagui.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from tools.table import DiagTable


class Ui_WndMain(object):
    def setupUi(self, WndMain):
        if not WndMain.objectName():
            WndMain.setObjectName(u"WndMain")
        WndMain.resize(1002, 600)
        font = QFont()
        font.setFamily(u"Microsoft YaHei UI")
        WndMain.setFont(font)
        self.actionOpen = QAction(WndMain)
        self.actionOpen.setObjectName(u"actionOpen")
        self.actionThreshold = QAction(WndMain)
        self.actionThreshold.setObjectName(u"actionThreshold")
        self.actionFilterError = QAction(WndMain)
        self.actionFilterError.setObjectName(u"actionFilterError")
        self.actionFilterError.setCheckable(True)
        self.actionFilterError.setChecked(False)
        self.actionmarkdown = QAction(WndMain)
        self.actionmarkdown.setObjectName(u"actionmarkdown")
        self.actionStat = QAction(WndMain)
        self.actionStat.setObjectName(u"actionStat")
        self.centralwidget = QWidget(WndMain)
        self.centralwidget.setObjectName(u"centralwidget")
        self.horizontalLayout = QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(8, 0, 8, 0)
        self.splitter = QSplitter(self.centralwidget)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setOrientation(Qt.Horizontal)
        self.tblDiag = DiagTable(self.splitter)
        self.tblDiag.setObjectName(u"tblDiag")
        self.tblDiag.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tblDiag.setTextElideMode(Qt.ElideNone)
        self.tblDiag.setSortingEnabled(True)
        self.splitter.addWidget(self.tblDiag)
        self.tblDiag.verticalHeader().setVisible(True)
        self.scrollArea = QScrollArea(self.splitter)
        self.scrollArea.setObjectName(u"scrollArea")
        self.scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scrollArea.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignTop)
        self.scrollAreaWidgetContents = QWidget()
        self.scrollAreaWidgetContents.setObjectName(u"scrollAreaWidgetContents")
        self.scrollAreaWidgetContents.setGeometry(QRect(0, 0, 452, 545))
        self.scrollContent = QGridLayout(self.scrollAreaWidgetContents)
        self.scrollContent.setObjectName(u"scrollContent")
        self.scrollContent.setHorizontalSpacing(0)
        self.scrollContent.setVerticalSpacing(8)
        self.scrollContent.setContentsMargins(0, 0, 0, 0)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.splitter.addWidget(self.scrollArea)

        self.horizontalLayout.addWidget(self.splitter)

        WndMain.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(WndMain)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1002, 27))
        self.menuFiles = QMenu(self.menubar)
        self.menuFiles.setObjectName(u"menuFiles")
        self.menuPreference = QMenu(self.menuFiles)
        self.menuPreference.setObjectName(u"menuPreference")
        self.menuExport = QMenu(self.menuFiles)
        self.menuExport.setObjectName(u"menuExport")
        self.menuFilter = QMenu(self.menubar)
        self.menuFilter.setObjectName(u"menuFilter")
        WndMain.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(WndMain)
        self.statusbar.setObjectName(u"statusbar")
        WndMain.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menuFiles.menuAction())
        self.menubar.addAction(self.menuFilter.menuAction())
        self.menuFiles.addAction(self.actionOpen)
        self.menuFiles.addAction(self.menuExport.menuAction())
        self.menuFiles.addSeparator()
        self.menuFiles.addAction(self.menuPreference.menuAction())
        self.menuPreference.addAction(self.actionThreshold)
        self.menuExport.addAction(self.actionmarkdown)
        self.menuFilter.addAction(self.actionFilterError)

        self.retranslateUi(WndMain)
        self.actionOpen.triggered.connect(WndMain.configOpened)
        self.actionThreshold.triggered.connect(WndMain.setThreshold)
        self.actionFilterError.toggled.connect(WndMain.errorFilter)
        self.actionmarkdown.triggered.connect(WndMain.exportMarkdown)
        self.actionStat.triggered.connect(WndMain.showStatistic)
        self.tblDiag.cellDoubleClicked.connect(WndMain.viewPID)

        QMetaObject.connectSlotsByName(WndMain)
    # setupUi

    def retranslateUi(self, WndMain):
        WndMain.setWindowTitle(QCoreApplication.translate("WndMain", u"DiagUI", None))
        self.actionOpen.setText(QCoreApplication.translate("WndMain", u"Open", None))
#if QT_CONFIG(shortcut)
        self.actionOpen.setShortcut(QCoreApplication.translate("WndMain", u"Ctrl+O", None))
#endif // QT_CONFIG(shortcut)
        self.actionThreshold.setText(QCoreApplication.translate("WndMain", u"Threshold", None))
#if QT_CONFIG(tooltip)
        self.actionThreshold.setToolTip(QCoreApplication.translate("WndMain", u"Change threshold of B/M", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(shortcut)
        self.actionThreshold.setShortcut(QCoreApplication.translate("WndMain", u"Ctrl+T", None))
#endif // QT_CONFIG(shortcut)
        self.actionFilterError.setText(QCoreApplication.translate("WndMain", u"Error only", None))
#if QT_CONFIG(tooltip)
        self.actionFilterError.setToolTip(QCoreApplication.translate("WndMain", u"Filter out items that error occurs", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(shortcut)
        self.actionFilterError.setShortcut(QCoreApplication.translate("WndMain", u"Ctrl+F", None))
#endif // QT_CONFIG(shortcut)
        self.actionmarkdown.setText(QCoreApplication.translate("WndMain", u"markdown", None))
#if QT_CONFIG(tooltip)
        self.actionmarkdown.setToolTip(QCoreApplication.translate("WndMain", u"Export markdown(*.md)", None))
#endif // QT_CONFIG(tooltip)
        self.actionStat.setText(QCoreApplication.translate("WndMain", u"Statistic", None))
#if QT_CONFIG(shortcut)
        self.actionStat.setShortcut(QCoreApplication.translate("WndMain", u"Ctrl+D", None))
#endif // QT_CONFIG(shortcut)
        self.menuFiles.setTitle(QCoreApplication.translate("WndMain", u"Files", None))
        self.menuPreference.setTitle(QCoreApplication.translate("WndMain", u"Preference", None))
        self.menuExport.setTitle(QCoreApplication.translate("WndMain", u"Export", None))
        self.menuFilter.setTitle(QCoreApplication.translate("WndMain", u"Filter", None))
    # retranslateUi

