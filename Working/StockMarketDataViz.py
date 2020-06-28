import sys
import PyQt5
from PyQt5.QtWidgets import * #QApplication, QLabel, QWidget, QPushButton, QVBoxLayout
from PyQt5.QtCore import *
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtWidgets import QApplication, QPushButton


# Subclass QMainWindow to customize application's main window
class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.setWindowTitle("Stock Market Data Visualization")

        label = QLabel("Company Name")

        # Qt namespace attributes to customize widgets
        # See http://doc.qt.io/qt-5.html
        label.setAlignment(Qt.AlignCenter)

        # Set central widget of Window.
        # Widget will expand to take up all window space by default
        self.setCentralWidget(label)

class CustomButton(QPushButton):
    def keyPressEvent(self, e):

        # My custome event handling
        super(CustomButton, self).keyPressEvent(e)


 # SIGNAL: The connected function will be called whenever the window
        # title is changed. The new title will be passed to the function.
        self.windowTitleChanged.connect(self.onWindowTitleChange)

        # SIGNAL: The connected function will be called whenever the window
        # title is changed. The new title is discarded in the lambda and the
        # function is called without parameters.
        self.windowTitleChanged.connect(lambda x: self.my_custom_fn())

        # SIGNAL: The connected function will be called whenever the window
        # title is changed. The new title is passed to the function
        # and replaces the default parameter
        self.windowTitleChanged.connect(lambda x: self.my_custom_fn(x))

        # SIGNAL: The connected function will be called whenever the window
        # title is changed. The new title is passed to the function
        # and replaces the default parameter. Extra data is passed from
        # within the lambda.
        self.windowTitleChanged.connect(lambda x: self.my_custom_fn(x, 25))
        
        # This sets the window title which will trigger all the above signals
        # sending the new title to the attached functions or lambdas as the
        # first parameter.
        self.setWindowTitle("Stock Market Data Visualization")
        
        label = QLabel("Company Name")
        label.setAlignment(Qt.AlignCenter)

        self.setCentralWidget(label)
        
        
    # SLOT: This accepts a string, e.g. the window title, and prints it
    def onWindowTitleChange(self, s):
        print(s)

    # SLOT: This has default parameters and can be called without a value
    def my_custom_fn(self, a="Enter Company Name", b=5):
        print(a, b)

# Create QApplication instance
app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec_()





#window = QWidget()

#app.setStyle('Fusion')

#palette = QPalette()
#app.setPalette(palette)


#qApp.setStyle("Fusion")

#dark_palette = QPalette()

#dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
#dark_palette.setColor(QPalette.WindowText, Qt.white)
#dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
#dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
#dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
#dark_palette.setColor(QPalette.ToolTipText, Qt.white)
#dark_palette.setColor(QPalette.Text, Qt.white)
#dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
#dark_palette.setColor(QPalette.ButtonText, Qt.white)
#dark_palette.setColor(QPalette.BrightText, Qt.red)
#dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
#dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
#dark_palette.setColor(QPalette.HighlightedText, Qt.black)

#qApp.setPalette(dark_palette)

#qApp.setStyleSheet("QToolTip { color: #ffffff; background-color: #2a82da; border: 1px solid white; }")


#layout = QVBoxLayout()
#layout.addWidget(QPushButton('SEARCH'))
#layout.addWidget(QPushButton('SEARCH'))

#button1 = QPushButton('Hello')
#button1.show()

#label = QLabel('Stock Market Data Visualization')
#label.show()
#window.setLayout(layout)
#window.show()


#app.exec_()

