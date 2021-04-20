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
        self.centralwidget = QWidget(WndMain)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.tblDiag = QTableWidget(self.centralwidget)
        if (self.tblDiag.columnCount() < 6):
            self.tblDiag.setColumnCount(6)
        self.tblDiag.setObjectName(u"tblDiag")
        self.tblDiag.setStyleSheet(u"a")
        self.tblDiag.setSortingEnabled(True)
        self.tblDiag.setColumnCount(6)
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
        WndMain.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(WndMain)
        self.statusbar.setObjectName(u"statusbar")
        WndMain.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menuFiles.menuAction())
        self.menuFiles.addAction(self.actionOpen)
        self.menuFiles.addAction(self.menuPreference.menuAction())
        self.menuPreference.addAction(self.actionThreshold)

        self.retranslateUi(WndMain)
        self.actionOpen.triggered.connect(WndMain.configOpened)
        self.actionThreshold.triggered.connect(WndMain.setThreshold)
        self.tblDiag.cellDoubleClicked.connect(WndMain.viewPID)

        QMetaObject.connectSlotsByName(WndMain)
    # setupUi

    def retranslateUi(self, WndMain):
        WndMain.setWindowTitle(QCoreApplication.translate("WndMain", u"DiagUI", None))
        self.actionOpen.setText(QCoreApplication.translate("WndMain", u"Open", None))
        self.actionThreshold.setText(QCoreApplication.translate("WndMain", u"Threshold", None))
        self.menuFiles.setTitle(QCoreApplication.translate("WndMain", u"Files", None))
        self.menuPreference.setTitle(QCoreApplication.translate("WndMain", u"Preference", None))
    # retranslateUi

