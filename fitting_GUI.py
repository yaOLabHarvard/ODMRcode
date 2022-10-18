import sys
import os.path
import numpy as np
import csv
from lmfit import Parameters, Model
#import matplotlib
from matplotlib.backends.backend_qtagg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure

from PyQt6.QtCore import QDateTime, Qt, QTimer
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateTimeEdit,
        QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QPushButton, QRadioButton, QScrollBar, QSizePolicy,
        QSlider, QSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit,
        QVBoxLayout, QWidget)

from NV_ESR_fit_V1 import NV100_fitting_function ## the fitting functions
from NV_ESR_fit_V1 import NVFit, curve_style ## import the global variable that controls the fitting peak functions



class MainWindow(QDialog):

    def __init__(self):
        super(MainWindow, self).__init__()
        # set palette
        self.originalPalette = QApplication.palette()
        # set style
        QApplication.setStyle('Fusion')
        # initialize variables
        self.PNum = len(NVFit.PGuess)
        self.isCheckEd = np.zeros(self.PNum)
        self.gmodel = Model(NV100_fitting_function)
        self.params = self.gmodel.make_params()
        self.fitResult = None
        self.datax=[]
        self.datay=[]
        # self.filepath = 'D:\\work\\py\\test.txt'
        self.filepath = 'C:\\Users\\esthe\\OneDrive\\Desktop\\VSCode\\Plotting\\test.txt'
        # initialize x range
        self.startx = 2.65e9
        self.endx = 3.2e9
        self.Nx = 501
        self.xlist = np.linspace(self.startx, self.endx, self.Nx)
        #Generate Figurecanvas class
        self.plotInit = 1
        self.plotWidth =  5
        self.plotHeight = 5
        self.sc = FigureCanvas(Figure(figsize=(self.plotWidth, self.plotHeight), dpi=50))
        self.axes = self.sc.figure.subplots()
        #self.axes.subplots_adjust(wspace=0.6, hspace=0.6, left=0.1, bottom=0.22, right=0.96, top=0.96)
        # generate widgets for the gui subsections
        self.createTopLeftGroupBox()
        self.createTopRightGroupBox()
        self.createPlotcanvas()
        self.createBottomRightWidget()


        # create final geometry for four subsections
        mainLayout = QGridLayout()
        mainLayout.addWidget(self.topLeftGroupBox, 0, 0, 1, 4)
        mainLayout.addWidget(self.topRightGroupBox, 0, 4, 6, 2)
        mainLayout.addWidget(self.plotcanvas, 1, 0, 4, 4)
        mainLayout.addWidget(self.bottomRightGroupBox, 5, 0, 1, 4)
##        mainLayout.setRowStretch(1, 1)
##        mainLayout.setRowStretch(2, 1)
##        mainLayout.setColumnStretch(0, 1)
##        mainLayout.setColumnStretch(1, 1)
        self.setLayout(mainLayout)

        
        self.setWindowTitle("NV Exact Fitting V1")


    def createTopLeftGroupBox(self):
        self.topLeftGroupBox = QGroupBox("Load file")

        text1 = QLabel()
        text1.setText("File name:")
        # self.inputLine = QLineEdit("D:\\work\\py\\test.txt")
        self.inputLine = QLineEdit("C:\\Users\\esthe\\OneDrive\\Desktop\\VSCode\\Plotting\\test.txt")
        text2 = QLabel()
        text2.setText("e.g. C:\\Users\\yourname\\Desktop\\file.csv")
        loadButton = QPushButton("Load and plot!")
        loadButton.clicked.connect(self.loadFileButtonCallback)

        layout = QVBoxLayout()
        layout.addWidget(text1)
        layout.addWidget(self.inputLine)
        layout.addWidget(text2)
        layout.addWidget(loadButton)
        self.topLeftGroupBox.setLayout(layout)

    def createTopRightGroupBox(self):
        self.topRightGroupBox = QGroupBox("Fitting Parameters")
        layout = QGridLayout()
        textCol = {}
        for i in range(self.PNum):
            textCol[i] = QLabel()
            textCol[i].setText(self.gmodel.param_names[i])
            layout.addWidget(textCol[i], i, 0)

        self.lineCol = {}
        for i in range(self.PNum):
            self.lineCol[i] = QLineEdit(str(NVFit.PGuess[i]))
            layout.addWidget(self.lineCol[i], i, 1)

        self.checkCol = {}

        for i in range(self.PNum):
            self.checkCol[i] = QCheckBox("fixed")
            layout.addWidget(self.checkCol[i], i, 2)

        updateButton = QPushButton("Update!")
        updateButton.clicked.connect(self.updateButtonCallback)
        layout.addWidget(updateButton, self.PNum + 1, 1)

        
        self.topRightGroupBox.setLayout(layout)

    def createPlotcanvas(self):
        self.plotcanvas = QGroupBox("Plot")

        # Create toolbar, frequency range and canvas
        self.axes.set_ylabel('Intensity')
        self.axes.set_xlabel('Frequency(Hz)')
        toolbar = NavigationToolbar(self.sc, self)
        
        text1 = QLabel()
        text1.setText("start freq (Hz)")
        self.startxLine = QLineEdit("2.65e9")
        text2 = QLabel()
        text2.setText("end freq (Hz)")
        self.endxLine = QLineEdit("3.2e9")
        text3 = QLabel()
        text3.setText("Num of pts")
        self.NxLine = QLineEdit("500")

        #set up plot layout
        layout = QGridLayout()
        layout.addWidget(text1, 0, 0, 1, 1)
        layout.addWidget(text2, 0, 1, 1, 1)
        layout.addWidget(text3, 0, 2, 1, 1)
        layout.addWidget(self.startxLine, 1, 0, 1, 1)
        layout.addWidget(self.endxLine, 1, 1, 1, 1)
        layout.addWidget(self.NxLine, 1, 2, 1, 1)
        layout.addWidget(toolbar, 2, 0, 1, 3)
        layout.addWidget(self.sc, 3, 0, 1, 3)

        # add layout into the widget
        self.plotcanvas.setLayout(layout)


    def createBottomRightWidget(self):
        self.bottomRightGroupBox = QWidget()
        
        fittingButton = QPushButton("Fit!")
        fittingButton.clicked.connect(self.fittingButtonCallback)
        exportButton = QPushButton("Export!")
        exportButton.clicked.connect(self.exportButtonCallback)
        
        layout = QHBoxLayout()
        layout.addWidget(fittingButton)
        layout.addWidget(exportButton)
        self.bottomRightGroupBox.setLayout(layout)

    def updateCanvas(self, x=np.array([0,1,2,3,4]), y=np.array([0,0,0,0, 0])):
        self.axes.cla()  # Clear the canvas.
        self.axes.set_ylabel('Intensity')
        self.axes.set_xlabel('Frequency(Hz)')
        self.axes.plot(self.datax, self.datay, 'o-b', label = 'Exp')
        if self.plotInit:
            self.plotInit = 0
        else:
            self.axes.plot(x, y, '-r', label = 'Fit')
        # Trigger the canvas to update and redraw.
        self.axes.set_xlim([self.startx, self.endx])
        self.axes.legend()
        self.sc.draw()

    def esrData(self, filename):
        with open(filename) as file:
            txtFile = csv.reader(file, delimiter=" ")

            for line in txtFile:
                # print(line)
                if line != []:
                    if line[0]<'A' and line[0]>'*':
                        ## Turning the list of strings into an np array
                        nline = np.asarray(line)
                        nline = nline.astype(float)
                        ## Writing the parts of the line to x and y arrays
                        self.datax.append(nline[0])
                        self.datay.append(nline[1])

        # Normalizing the baseline of the ESR Data to 1
        datayavg=[]
        for i in self.datay: # Collect the baseline points (points within the standard deviation)
            if i > abs(np.mean(self.datay))-abs(np.std(self.datay)) and i < abs(np.mean(self.datay))+abs(np.std(self.datay)):
                datayavg.append(i)

        self.datayBaseline = np.mean(datayavg)
        self.datay=self.datay/np.mean(datayavg)
        self.datax = np.array(self.datax)
        self.datay = np.array(self.datay)

    # def loadFile(self):
        # dataArray = np.loadtxt(self.filepath)
        # print("Data contains {} lines".format(len(dataArray)))
        # dataArray = dataArray.transpose()
        # self.datax = dataArray[0]
        # self.datay = dataArray[1]

    def exportFile(self):
        (currentPath, fileName) = os.path.split(self.filePath)
        print(currentPath)
        newPath = os.path.join(currentPath, 'fit_summary.txt')
        print(newPath)
        tmpFile = open(newPath, "a")
        tmpFile.write(self.fitResult.fit_report())
        tmpFile.write("\n")
        tmpFile.write("#########################")
        tmpFile.close()
    
    def loadFileButtonCallback(self):
        self.filePath = self.inputLine.text()
        print(self.filePath)
        # self.loadFile()
        self.esrData(self.filePath)
        self.updateCanvas()

    def updateButtonCallback(self):
        #print("haha")
        #print(self.PNum)
        self.startx = float(self.startxLine.text())
        self.endx = float(self.endxLine.text())
        self.Nx = int(self.NxLine.text())
        self.xlist = np.linspace(self.startx, self.endx, self.Nx)
        
        for i in range(self.PNum):
            #print(float(self.lineCol[i].text()))
            NVFit.PGuess[i] = float(self.lineCol[i].text())
        
        MWFreq = NVFit.NV100_exact_Bonly()
        ylist = NVFit.Plot_guess_fit(self.xlist, MWFreq, 'l')
        self.updateCanvas(self.xlist, ylist)
        print("Parameters updated")


    def fittingButtonCallback(self):
        print(NVFit.PGuess)
        for i in range(self.PNum):
            self.isCheckEd[i] = self.checkCol[i].isChecked()
            if i == 3 or i == 4:
                self.params.add(self.gmodel.param_names[i], value=NVFit.PGuess[i], vary=1 - self.isCheckEd[i], min=0, max=90)
            else:
                self.params.add(self.gmodel.param_names[i], value=NVFit.PGuess[i], vary=1 - self.isCheckEd[i])
        print(self.params)
        print("Fitting parameters are set")
        if self.datay is None:
            print("Experimenral data isn't loaded yet")
            self.updateButtonCallback()
        else:
            self.fitResult = self.gmodel.fit(self.datay, self.params, xVals=self.datax)
            self.updateCanvas(self.datax, self.fitResult.best_fit)
            print("Fitting plotted")


    def exportButtonCallback(self):
        if self.fitResult is None:
            print("Data isn't fitted yet")
        else:
            self.exportFile()
            print("Report saved")
        
        
         
        
    

if __name__ == "__main__":
    # Check whether there is already a running QApplication (e.g., if running
    # from an IDE).
    qapp = QApplication.instance()
    if not qapp:
        qapp = QApplication(sys.argv)

    app = MainWindow()
    app.show()
    app.activateWindow()
    app.raise_()
    qapp.exec()
