import os
import sys
from os.path import dirname, realpath, join
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QTableWidget, QTableWidgetItem
from PyQt5.uic import loadUiType
import pandas as pd
from PyQt5 import QtWidgets


# import files of various UI pages used in the Project (Workflow)
from pipeline_showcase import Ui_pipeline

scriptDir = dirname(realpath(__file__))
From_Main, _ = loadUiType(join(dirname(__file__), "Main.ui"))

class MainWindow(QWidget, From_Main):
    def __init__(self):
        super(MainWindow, self).__init__()
        QWidget.__init__(self)
        self.setupUi(self)

        self.ButtonOpen.clicked.connect(self.OpenFile)
        self.BtnDescribe.clicked.connect(self.dataHead)
        self.Next_Button.clicked.connect(self.Next_Page)

    def OpenFile(self):
        try:
            path = QFileDialog.getOpenFileName(self, 'Open CSV', os.getenv('HOME'), 'CSV(*.csv)')[0]
            self.all_data = pd.read_csv(path)
            # Basic EDA in terminal for debugging and reference purpose.

            print("\n\n\n")
            print("#############################################################################")
            print("Log for Reference purpose and Debugging")
            print("#############################################################################")
            print("\n\n\n")
            print("#############################################################################")
            print(self.all_data.head())
            print("#############################################################################")
            print("\n\n\n")

            print("#############################################################################")
            print(self.all_data.info())
            print("#############################################################################")
            print("\n\n\n")

            print("#############################################################################")
            print(self.all_data.describe())
            print("#############################################################################")
            print("\n\n\n")

            print("#############################################################################")
            print(self.all_data.dtypes)
            print("#############################################################################")
            print("\n\n\n")

            print("#############################################################################")
            print(self.all_data.shape)
            print("#############################################################################")
            print("\n\n\n")
        except:
            print(path)

    def dataHead(self):
        numColomn = self.spinBox.value()
        if numColomn == 0:
            # NumRows = len(self.all_data.index) => doesn't respond if dataset is too large.
            NumRows = len(self.all_data.head(1000))
        else:
            NumRows = numColomn
        self.tableWidget.setColumnCount(len(self.all_data.columns))
        self.tableWidget.setRowCount(NumRows)
        self.tableWidget.setHorizontalHeaderLabels(self.all_data.columns)

        for i in range(NumRows):
            for j in range(len(self.all_data.columns)):
                self.tableWidget.setItem(i, j, QTableWidgetItem(str(self.all_data.iat[i, j])))

        self.tableWidget.resizeColumnsToContents()
        self.tableWidget.resizeRowsToContents()


    def Next_Page(self):
        # Reuse the existing QDialog instance
        self.window = QtWidgets.QDialog()
        
        # Create an instance of Ui_pipeline and set it up
        self.ui = Ui_pipeline()
        self.ui.setupUi(self.window)

        # Now you can show the QDialog
        self.window.show()

        # Close the current window (optional, if needed)
        self.close()

        # # self.pipeline_dialog = Ui_pipeline()
        # # self.pipeline_dialog.exec_()


        # dialog = Ui_pipeline()  # Create an instance of the dialog
        # dialog_ui = Ui_pipeline()  # Create an instance of the dialog UI class
        # dialog_ui.setupUi(dialog)  # Set up the dialog UI
        # dialog.show()  # Show the dialog

        # dialog = Ui_pipeline()  # Create an instance of QDialog
        # dialog_ui = Ui_pipeline()  # Create an instance of the dialog UI class
        # dialog_ui.setupUi(dialog)  # Set up the dialog UI

        # dialog.show()  # Show the dialog modally


if __name__ == "__main__":
    app = QApplication(sys.argv)
    sheet = MainWindow()
    sheet.show()
    sys.exit(app.exec_())

    