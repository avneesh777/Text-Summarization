# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'data_processing.ui'
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
from PySide6.QtWidgets import (QApplication, QDialog, QLabel, QPushButton,
    QSizePolicy, QWidget)

class Ui_Text_Summarization(object):
    def setupUi(self, Text_Summarization):
        if not Text_Summarization.objectName():
            Text_Summarization.setObjectName(u"Text_Summarization")
        Text_Summarization.resize(1200, 800)
        self.Next_Button = QPushButton(Text_Summarization)
        self.Next_Button.setObjectName(u"Next_Button")
        self.Next_Button.setGeometry(QRect(480, 710, 251, 51))
        self.Text_Preprocessing = QPushButton(Text_Summarization)
        self.Text_Preprocessing.setObjectName(u"Text_Preprocessing")
        self.Text_Preprocessing.setGeometry(QRect(480, 230, 251, 51))
        self.Model_Building = QPushButton(Text_Summarization)
        self.Model_Building.setObjectName(u"Model_Building")
        self.Model_Building.setGeometry(QRect(480, 360, 251, 51))
        self.Training_Model = QPushButton(Text_Summarization)
        self.Training_Model.setObjectName(u"Training_Model")
        self.Training_Model.setGeometry(QRect(480, 490, 251, 51))
        self.Heading = QLabel(Text_Summarization)
        self.Heading.setObjectName(u"Heading")
        self.Heading.setGeometry(QRect(270, 40, 641, 131))
        font = QFont()
        font.setPointSize(36)
        self.Heading.setFont(font)
        self.Heading.setTextFormat(Qt.PlainText)
        self.Heading.setAlignment(Qt.AlignCenter)
        QWidget.setTabOrder(self.Text_Preprocessing, self.Model_Building)
        QWidget.setTabOrder(self.Model_Building, self.Training_Model)
        QWidget.setTabOrder(self.Training_Model, self.Next_Button)

        self.retranslateUi(Text_Summarization)

        QMetaObject.connectSlotsByName(Text_Summarization)
    # setupUi

    def retranslateUi(self, Text_Summarization):
        Text_Summarization.setWindowTitle(QCoreApplication.translate("Text_Summarization", u"Dialog", None))
        self.Next_Button.setText(QCoreApplication.translate("Text_Summarization", u"Next", None))
        self.Text_Preprocessing.setText(QCoreApplication.translate("Text_Summarization", u"Text Preprocessing", None))
        self.Model_Building.setText(QCoreApplication.translate("Text_Summarization", u"Building the Model", None))
        self.Training_Model.setText(QCoreApplication.translate("Text_Summarization", u"Training the Model", None))
        self.Heading.setText(QCoreApplication.translate("Text_Summarization", u"Pipeline Flow", None))
    # retranslateUi

