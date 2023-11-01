# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main.ui'
##
## Created by: Qt User Interface Compiler version 6.6.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QComboBox, QFrame, QGridLayout,
    QHBoxLayout, QHeaderView, QLabel, QPushButton,
    QSizePolicy, QSpacerItem, QSpinBox, QTabWidget,
    QTableWidget, QTableWidgetItem, QWidget)

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(1200, 800)
        Form.setStyleSheet(u"")
        self.horizontalLayout = QHBoxLayout(Form)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer, 0, 2, 1, 1)

        self.tabWidget = QTabWidget(Form)
        self.tabWidget.setObjectName(u"tabWidget")
        font = QFont()
        font.setPointSize(13)
        self.tabWidget.setFont(font)
        self.tab = QWidget()
        self.tab.setObjectName(u"tab")
        self.gridLayout_2 = QGridLayout(self.tab)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.label = QLabel(self.tab)
        self.label.setObjectName(u"label")

        self.gridLayout_2.addWidget(self.label, 0, 2, 1, 1)

        self.tableWidget = QTableWidget(self.tab)
        self.tableWidget.setObjectName(u"tableWidget")

        self.gridLayout_2.addWidget(self.tableWidget, 2, 0, 1, 7)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_2.addItem(self.horizontalSpacer_2, 0, 6, 1, 1)

        self.BtnDescribe = QPushButton(self.tab)
        self.BtnDescribe.setObjectName(u"BtnDescribe")
        font1 = QFont()
        font1.setPointSize(9)
        self.BtnDescribe.setFont(font1)

        self.gridLayout_2.addWidget(self.BtnDescribe, 0, 4, 1, 1)

        self.spinBox = QSpinBox(self.tab)
        self.spinBox.setObjectName(u"spinBox")
        self.spinBox.setMaximum(1000)

        self.gridLayout_2.addWidget(self.spinBox, 0, 3, 1, 1)

        self.Next_Button = QPushButton(self.tab)
        self.Next_Button.setObjectName(u"Next_Button")
        icon = QIcon()
        icon.addFile(u"Next.png", QSize(), QIcon.Normal, QIcon.Off)
        self.Next_Button.setIcon(icon)

        self.gridLayout_2.addWidget(self.Next_Button, 0, 5, 1, 1)

        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QWidget()
        self.tab_2.setObjectName(u"tab_2")
        self.gridLayout_3 = QGridLayout(self.tab_2)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.frame = QFrame(self.tab_2)
        self.frame.setObjectName(u"frame")
        self.frame.setFrameShape(QFrame.StyledPanel)
        self.frame.setFrameShadow(QFrame.Raised)

        self.gridLayout_3.addWidget(self.frame, 1, 0, 1, 2)

        self.comboBox = QComboBox(self.tab_2)
        self.comboBox.setObjectName(u"comboBox")

        self.gridLayout_3.addWidget(self.comboBox, 0, 0, 1, 1)

        self.comboBox_3 = QComboBox(self.tab_2)
        self.comboBox_3.setObjectName(u"comboBox_3")

        self.gridLayout_3.addWidget(self.comboBox_3, 0, 2, 1, 1)

        self.pushButton = QPushButton(self.tab_2)
        self.pushButton.setObjectName(u"pushButton")
        font2 = QFont()
        font2.setPointSize(11)
        self.pushButton.setFont(font2)

        self.gridLayout_3.addWidget(self.pushButton, 0, 3, 1, 1)

        self.comboBox_2 = QComboBox(self.tab_2)
        self.comboBox_2.setObjectName(u"comboBox_2")

        self.gridLayout_3.addWidget(self.comboBox_2, 0, 1, 1, 1)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_3.addItem(self.horizontalSpacer_3, 0, 4, 1, 1)

        self.tabWidget.addTab(self.tab_2, "")

        self.gridLayout.addWidget(self.tabWidget, 2, 0, 1, 3)

        self.line = QFrame(Form)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)

        self.gridLayout.addWidget(self.line, 1, 1, 1, 1)

        self.ButtonOpen = QPushButton(Form)
        self.ButtonOpen.setObjectName(u"ButtonOpen")
        sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ButtonOpen.sizePolicy().hasHeightForWidth())
        self.ButtonOpen.setSizePolicy(sizePolicy)
        self.ButtonOpen.setMinimumSize(QSize(54, 26))
        font3 = QFont()
        font3.setPointSize(13)
        font3.setBold(True)
        self.ButtonOpen.setFont(font3)
        self.ButtonOpen.setStyleSheet(u"\n"
"selection-background-color: rgb(255, 68, 21);\n"
"border-color: rgb(255, 97, 100);")
        icon1 = QIcon()
        icon1.addFile(u"../../../youtube AI creative/indexezd.png", QSize(), QIcon.Normal, QIcon.Off)
        self.ButtonOpen.setIcon(icon1)
        self.ButtonOpen.setIconSize(QSize(50, 14))
        self.ButtonOpen.setCheckable(False)
        self.ButtonOpen.setAutoRepeat(False)
        self.ButtonOpen.setFlat(True)

        self.gridLayout.addWidget(self.ButtonOpen, 0, 0, 2, 1)


        self.horizontalLayout.addLayout(self.gridLayout)


        self.retranslateUi(Form)

        self.tabWidget.setCurrentIndex(0)
        self.ButtonOpen.setDefault(True)


        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Excel Loader", None))
        self.label.setText(QCoreApplication.translate("Form", u"Columns", None))
        self.BtnDescribe.setText(QCoreApplication.translate("Form", u"Describe", None))
        self.Next_Button.setText("")
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), QCoreApplication.translate("Form", u"Describe data", None))
        self.pushButton.setText(QCoreApplication.translate("Form", u"Plot", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), QCoreApplication.translate("Form", u"Plot Data", None))
        self.ButtonOpen.setText(QCoreApplication.translate("Form", u"Open CSV", None))
    # retranslateUi

