# In[1]:
import multipeak_fit as mf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.integrate import quad
from scipy.signal import find_peaks
from scipy.signal import correlate2d
import scipy as sp
import random as rd
path = "F:/NMR/NMR/py_projects/WF/ODMRcode/WF/raw_data/"
filename= '100xobj_Bz0p3A.mat'

class WFimage:
    def __init__(self, filename):
        self.fVals, self.dat, self.xFrom, self.xTo, self.X, self.Y, self.npoints = mf.read_matfile(filename, normalize= False)
        print("mat file is loaded successfully!")
        self.originalDat = self.dat
        self.dataMask = None
        self.isNorm = False
        self.isMultfit = False

    def norm(self):
        self.dat = mf.normalize_widefield(self.dat)
        self.isNorm = True

    def sliceImage(self, Nslice = 3):
        return self.dat[:,:,Nslice].copy()
    
    def generateTmp(self, ROI = [[0, 1], [0, 1]], Nslice = 3):
        return self.dat[ROI[0,0]:ROI[0,1],ROI[1,0]:ROI[1,1],Nslice].copy()
    
    def pointESR(self, x = 0, y = 0):
        return self.dat[x, y].copy()
        
    def maskData(self, eps = 1e-3):
        if not self.isNorm:
            self.norm()
        self.dataMask = np.zeros((self.X, self.Y), dtype=int)
        for x in range(self.X):
            for y in range(self.Y):
                if min(self.dat[x, y]) < 1 - eps:
                    self.dataMask[x, y]  = 1

    def maskContourgen(self):
        if self.dataMask is not None:
            xlist = np.arange(0, self.X, 1)
            ylist = np.arange(0, self.Y, 1)
            CX, CY = np.meshgrid(xlist, ylist)
            CZ = np.zeros((self.X, self.Y), dtype=float)
            for x in range(self.X):
                for y in range(self.Y):
                    CZ[x, y] = min(self.dataMask[x, y])

            return CX, CY, CZ
        else:
            print("Mask hasn't been created! Make the mask by maskData")

    def singleESRpeakfind(self, x = 0, y = 0, max_peak = 2):
        if not self.isNorm:
            self.norm()
        yVals = self.pointESR(x, y)
        ymax = max(1 - yVals)
        # ymin = min(1 - yVals)
        peakPos, _ = find_peaks(1 - yVals, distance = 30,  height=(ymax/3, ymax), width = 10)
        #peakPos, _ = find_peaks(1 - yVals, prominence=0.8)
        peak_values = yVals[peakPos]

        #   Sort the peaks by amplitude (highest to lowest)
        sort_indices = np.argsort(peak_values)[::-1]
        peak_indices_sorted = peakPos[sort_indices]
        if len(peakPos) < max_peak:
            return peak_indices_sorted
        else:
            top_peak = peak_indices_sorted[:max_peak]
            return top_peak

    def singleESRfit(self, x = 0, y = 0, autofind = True):
        yVals = self.pointESR(x, y)
        if autofind:
            peakPos = self.singleESRpeakfind(x, y)
        else:
            peakPos = np.fromstring(input('Enter frequency (for example: 2.71, 2.81, 2.91, 3.01):'),sep=',')
        initParas = mf.generate_pinit(self.fVals[peakPos], np.zeros(len(peakPos)))
        pOpt, pCov= mf.fit_data(self.fVals, yVals, init_params= initParas, fit_function= mf.lor_fit, maxFev=100)
        if pOpt is not None:
            residuals = yVals - mf.lor_fit(self.fVals, *pOpt)
            chiSq = np.sum((residuals / mf.lor_fit(self.fVals, *pOpt)) ** 2)
        else:
            chiSq = 1
        return pOpt, pCov, chiSq

    def multiESRfit(self, xlist, ylist):
        self.optList = {}
        self.covList = {}
        self.sqList = {}
        self.multix = xlist
        self.multiy = ylist
        for x in self.multix:
            for y in self.multiy:
                pOpt, pCov, chiSq = self.singleESRfit(x, y)
                self.optList[(x, y)] = pOpt
                self.covList[(x, y)] = pCov
                self.sqList[(x, y)] = chiSq
            print("{}-th row has completed".format(x))

        self.isMultfit = True
    
    def multiNpeakplot(self):
        if self.isMultfit:
            self.Npeak = np.zeros((self.X, self.Y))
            # print(self.Npeak.shape)
            # print(self.multix)
            # print(self.multiy)
            for x in self.multix:
                for y in self.multiy:
                    popt = self.optList[(x, y)]
                    if popt is not None:
                        self.Npeak[x,y] = int(np.floor(len(popt)/3))

            return self.Npeak
        else:
            print("Run multiESRfit first to unlock this")

    def multiChisqplot(self):
        if self.isMultfit:
            self.chixy = np.zeros((self.X, self.Y))
            # print(self.Npeak.shape)
            # print(self.multix)
            # print(self.multiy)
            for x in self.multix:
                for y in self.multiy:
                    chisq = self.sqList[(x, y)]
                    self.chixy[x,y] = chisq

            return self.chixy
        else:
            print("Run multiESRfit first to unlock this")

    def imgCorr(self, tmpImg = None, roi = [[50, 60],[40, 50]]):
        targetImg = self.sliceImage()
        if tmpImg is None: 
            tmpImg = self.generateTmp(ROI = roi)
        else:
            corr=correlate2d(targetImg, tmpImg, mode='same')

            ypos, xpos = np.unravel_index(np.argmax(corr), corr.shape)
            return ypos, xpos
        
    def shiftDat(self, dx = 0, dy = 0):
        (xx, yy, _) = np.shape(self.dat)
        newdat = np.ones(np.shape(self.dat))
        ##Move it to the correct position!
        for j in np.arange(xx):
            for k in np.arange(yy):
                if j+dy < xx and k+dx < yy:
                    newdat[j+dy,k+dx,:]=self.dat[j,k,:]
        self.dat=newdat

    
