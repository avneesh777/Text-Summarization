import sys
import time
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QProgressBar, QLabel, QFrame, QDialog, QVBoxLayout, QStackedWidget
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets

from data_processing import Ui_Data_processing


class SplashScreen(QWidget):
    loadingComplete = pyqtSignal()  # Define a custom signal
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Spash Screen Example')
        self.setFixedSize(1100, 500)
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self.counter = 0
        self.n = 300 # total instance

        self.initUI()

        self.timer = QTimer()
        self.timer.timeout.connect(self.loading)
        self.timer.start(30)

    def initUI(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.frame = QFrame()
        layout.addWidget(self.frame)

        self.labelTitle = QLabel(self.frame)
        self.labelTitle.setObjectName('LabelTitle')

        # center labels
        self.labelTitle.resize(self.width() - 10, 150)
        self.labelTitle.move(0, 40) # x, y
        self.labelTitle.setText('Processing')
        self.labelTitle.setAlignment(Qt.AlignCenter)

        self.labelDescription = QLabel(self.frame)
        self.labelDescription.resize(self.width() - 10, 50)
        self.labelDescription.move(0, self.labelTitle.height())
        self.labelDescription.setObjectName('LabelDesc')
        self.labelDescription.setText('<strong>Preparing Data...</strong>')
        self.labelDescription.setAlignment(Qt.AlignCenter)

        self.progressBar = QProgressBar(self.frame)
        self.progressBar.resize(self.width() - 200 - 10, 50)
        self.progressBar.move(100, self.labelDescription.y() + 130)
        self.progressBar.setAlignment(Qt.AlignCenter)
        self.progressBar.setFormat('%p%')
        self.progressBar.setTextVisible(True)
        self.progressBar.setRange(0, self.n)
        self.progressBar.setValue(20)

        self.labelLoading = QLabel(self.frame)
        self.labelLoading.resize(self.width() - 10, 50)
        self.labelLoading.move(0, self.progressBar.y() + 70)
        self.labelLoading.setObjectName('LabelLoading')
        self.labelLoading.setAlignment(Qt.AlignCenter)
        self.labelLoading.setText('loading...')

    def loading(self):
        self.progressBar.setValue(self.counter)

        if self.counter == int(self.n * 0.3):
            self.labelDescription.setText('<strong>Loading DataFrame...</strong>')
        elif self.counter == int(self.n * 0.6):
            self.labelDescription.setText('<strong>Finishing...</strong>')
        elif self.counter >= self.n:
            self.timer.stop()
            self.close()

            time.sleep(1)

            # Emit the signal when loading is complete
            self.loadingComplete.emit()

            # self.myApp = MyApp()
            # self.myApp.show()
            # Pay = MyApp()
            # widget.addWidget(Pay)
            # widget.setCurrentIndex(widget.currentIndex() + 1)

        self.counter += 1

        
class MyApp(QDialog):
    def __init__(self, stack_widget):
        super().__init__()
        self.stack_widget = stack_widget
        
        # self.window = QtWidgets.QDialog()
        
        # # Create an instance of Ui_pipeline and set it up
        # self.ui = Ui_Data_processing()
        # self.ui.show()
        # self.ui.setupUi(self.window)

        # # Now you can show the QDialog
        # self.window.show()

        # # Close the current window (optional, if needed)
        # self.close()

        # self.stack_widget = stack_widget

        # data_processing_screen = Ui_Data_processing()
        
        # # Add the Data_processing screen to the QStackedWidget
        # self.stack_widget.addWidget(data_processing_screen)
        
        # # Set the current screen to Data_processing
        # self.stack_widget.setCurrentWidget(data_processing_screen)

    def transitionToDataProcessing(self):
        # Create an instance of Ui_Data_processing
        data_processing_screen = Ui_Data_processing()
        
        # Add the Data_processing screen to the QStackedWidget
        self.stack_widget.addWidget(data_processing_screen)
        
        # Set the current screen to Data_processing
        self.stack_widget.setCurrentWidget(data_processing_screen)
        data_processing_screen.show()




# if __name__ == '__main__':
#     # don't auto scale when drag app to a different monitor.
#     # QApplication.setAttribute(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    
#     app = QApplication(sys.argv)
#     app.setStyleSheet('''
#         #LabelTitle {
#             font-size: 60px;
#             color: #93deed;
#         }

#         #LabelDesc {
#             font-size: 30px;
#             color: #c2ced1;
#         }

#         #LabelLoading {
#             font-size: 30px;
#             color: #e8e8eb;
#         }

#         QFrame {
#             background-color: #2F4454;
#             color: rgb(220, 220, 220);
#         }

#         QProgressBar {
#             background-color: #DA7B93;
#             color: rgb(200, 200, 200);
#             border-style: none;
#             border-radius: 10px;
#             text-align: center;
#             font-size: 30px;
#         }

#         QProgressBar::chunk {
#             border-radius: 10px;
#             background-color: qlineargradient(spread:pad x1:0, x2:1, y1:0.511364, y2:0.523, stop:0 #1C3334, stop:1 #376E6F);
#         }
#     ''')
    
#     # splash = SplashScreen()
#     # splash.show()
#     # widget = QStackedWidget() # Helps in moving between various screens/windows
#     # my_app = MyApp(widget)  # Create an instance of MyApp
#     # splash.loadingComplete.connect(MyApp.transitionToDataProcessing)

#     # # Show the main application window (widget)
#     # widget.setFixedHeight(801)
#     # widget.setFixedWidth(1201)
#     # widget.setWindowTitle("Flight Booking")
#     # widget.show()

#     # try:
#     #     sys.exit(app.exec_())
#     # except SystemExit:
#     #     print('Closing Window...')



#     splash = SplashScreen()
#     splash.show()

#     widget = QStackedWidget()
#     # Create an instance of MyApp
#     my_app = MyApp(widget)

#     # Connect the splash screen signal to the method of my_app
#     splash.loadingComplete.connect(my_app.transitionToDataProcessing)

#     app.exec_()

#     # When the splash screen is done, create and show the main application window (widget)
    
#     widget.setFixedHeight(801)
#     widget.setFixedWidth(1201)
#     widget.setWindowTitle("Flight Booking")
#     widget.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    


    app.setStyleSheet('''
            #LabelTitle {
                font-size: 60px;
                color: #93deed;
            }

            #LabelDesc {
                font-size: 30px;
                color: #c2ced1;
            }

            #LabelLoading {
                font-size: 30px;
                color: #e8e8eb;
            }

            QFrame {
                background-color: #2F4454;
                color: rgb(220, 220, 220);
            }

            QProgressBar {
                background-color: #DA7B93;
                color: rgb(200, 200, 200);
                border-style: none;
                border-radius: 10px;
                text-align: center;
                font-size: 30px;
            }

            QProgressBar::chunk {
                border-radius: 10px;
                background-color: qlineargradient(spread:pad x1:0, x2:1, y1:0.511364, y2:0.523, stop:0 #1C3334, stop:1 #376E6F);
            }
        ''')

    splash = SplashScreen()
    splash.show()

    widget = QStackedWidget()
    my_app = MyApp(widget)
    splash.loadingComplete.connect(my_app.transitionToDataProcessing)

    # Set the properties of the widget
    widget.setFixedHeight(801)
    widget.setFixedWidth(1201)
    widget.setWindowTitle("Flight Booking")

    # # Show the widget
    # widget.show()

    # Start the event loop
    sys.exit(app.exec_())
