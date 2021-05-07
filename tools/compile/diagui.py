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


class Ui_WndMain(object):
    def setupUi(self, WndMain):
        if not WndMain.objectName():
            WndMain.setObjectName(u"WndMain")
        WndMain.resize(800, 600)
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
        self.centralwidget = QWidget(WndMain)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.tblDiag = QTableWidget(self.centralwidget)
        if (self.tblDiag.columnCount() < 5):
            self.tblDiag.setColumnCount(5)
        self.tblDiag.setObjectName(u"tblDiag")
        self.tblDiag.setStyleSheet(u"a")
        self.tblDiag.setSortingEnabled(True)
        self.tblDiag.setColumnCount(5)
        self.tblDiag.verticalHeader().setVisible(True)

        self.gridLayout.addWidget(self.tblDiag, 0, 0, 1, 1)

        WndMain.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(WndMain)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 800, 27))
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
        self.tblDiag.cellDoubleClicked.connect(WndMain.viewPID)
        self.actionFilterError.toggled.connect(WndMain.errorFilter)
        self.actionmarkdown.triggered.connect(WndMain.exportMarkdown)

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
        self.menuFiles.setTitle(QCoreApplication.translate("WndMain", u"Files", None))
        self.menuPreference.setTitle(QCoreApplication.translate("WndMain", u"Preference", None))
        self.menuExport.setTitle(QCoreApplication.translate("WndMain", u"Export", None))
        self.menuFilter.setTitle(QCoreApplication.translate("WndMain", u"Filter", None))
    # retranslateUi

