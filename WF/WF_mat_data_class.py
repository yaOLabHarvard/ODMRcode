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
        self.originalDat = self.dat.copy()
        self.dataMask = None
        self.isNorm = False
        self.isMultfit = False
        self.isMask = False
        self.isNpeak = False

    def norm(self):
        self.dat = mf.normalize_widefield(self.dat)
        self.isNorm = True

    def sliceImage(self, Nslice = 3):
        return self.originalDat[:,:,Nslice].copy()
    
    def generateTmp(self, ROI = [[0, 1], [0, 1]], Nslice = 3):
        return self.originalDat[ROI[0][0]:ROI[0][1],ROI[1][0]:ROI[1][1],Nslice].copy()
    
    def pointESR(self, x = 0, y = 0):
        return self.dat[x, y].copy()
        
    def maskData(self, eps = .5e-3):
        if not self.isNorm:
            self.norm()
        self.dataMask = np.zeros((self.X, self.Y), dtype=int)
        for x in range(self.X):
            for y in range(self.Y):
                if min(self.dat[x, y]) < 1 - eps:
                    self.dataMask[x, y]  = 1

        self.isMask = True

    def myFavoriatePlot(self, x = 0, y = 0):
        fig, ax = plt.subplots(nrows=1, ncols= 2, figsize= (12,6))
        ## plot the image
        IMGplot = ax[0].imshow(self.originalDat[:,:,3].copy())
        ax[0].add_patch(Rectangle((x - 1, y -1), 2, 2, fill=None, alpha = 1))
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(IMGplot, cax=cax)


        ## plot single ESR
        spESR = self.pointESR(x, y)
        ax[1].plot(self.fVals, spESR, '-')
        peaks = self.singleESRpeakfind(x, y, method = 'user')
        popt, pcov, chisq = self.singleESRfit(x, y)
        print("the Chi square is {}".format(chisq))
        ax[1].plot(self.fVals[peaks], spESR[peaks], 'x')
        try:
            for i in np.arange(int(np.floor(len(popt)/3))):
                params= popt[1+3*i:4+3*i]
                ax[1].plot(self.fVals,popt[0]+mf.lorentzian(self.fVals,*params), '-')
        except TypeError:
            print("No good fit was found! Try increase Maxfev or try a better initial guess")

        plt.show()
        plt.close()

    def maskContourgen(self):
        if self.isMask:
            xlist = np.arange(0, self.X, 1)
            ylist = np.arange(0, self.Y, 1)
            CX, CY = np.meshgrid(xlist, ylist)
            CZ = np.zeros((self.X, self.Y), dtype=float)
            for x in xlist:
                for y in ylist:
                    CZ[x, y] = min(self.dat[x, y])

            return CX, CY, CZ
        else:
            print("Mask hasn't been created! Make the mask by maskData")

    def singleESRpeakfind(self, x = 0, y = 0, max_peak = 2, method = 'user'):
        if not self.isNorm:
            self.norm()
        yVals = self.pointESR(x, y)
        ymax = max(1 - yVals)
        # ymin = min(1 - yVals)
        if method == 'user':
            peakPos, _ = find_peaks(1 - yVals, distance = 20,  height=(ymax/5, ymax), width = 5)
        elif method == 'pro':
            peakPos, _ = find_peaks(1 - yVals, prominence=0.1)
        peak_values = yVals[peakPos]

        #   Sort the peaks by amplitude (highest to lowest)
        sort_indices = np.argsort(peak_values)[::-1]
        peak_indices_sorted = peakPos[sort_indices]
        if len(peakPos) < max_peak:
            return peak_indices_sorted
        else:
            top_peak = peak_indices_sorted[:max_peak]
            return top_peak


    def singleESRfit(self, x = 0, y = 0, autofind = True, backupOption = None):
        yVals = self.pointESR(x, y)
        if autofind:
            peakPos = self.singleESRpeakfind(x, y)
        else:
            peakPos = np.fromstring(input('Enter frequency (for example: 2.71, 2.81, 2.91, 3.01):'),sep=',')
        initParas = mf.generate_pinit(self.fVals[peakPos], np.zeros(len(peakPos)))

        pOpt, pCov= mf.fit_data(self.fVals, yVals, init_params= initParas, fit_function= mf.lor_fit, maxFev=500)
        if pOpt is not None:
            residuals = yVals - mf.lor_fit(self.fVals, *pOpt)
            chiSq = np.sum((residuals / mf.lor_fit(self.fVals, *pOpt)) ** 2)
        else:
            if backupOption is not None:
                initParas = mf.generate_pinit(backupOption, np.zeros(len(backupOption)))
                pOpt, pCov= mf.fit_data(self.fVals, yVals, init_params= initParas, fit_function= mf.lor_fit, maxFev=500)
            chiSq = 1
        return pOpt, pCov, chiSq

    def multiESRfit(self, xlist, ylist, backupOption = None):
        self.optList = {}
        self.covList = {}
        self.sqList = {}
        self.multix = xlist
        self.multiy = ylist
        for x in self.multix:
            for y in self.multiy:
                pOpt, pCov, chiSq = self.singleESRfit(x, y, backupOption=backupOption)
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
            self.isNpeak = True
            return self.Npeak
        else:
            print("Run multiESRfit first to unlock this")

    def npeakManualCorrection(self, eps = 5e-5):
        if self.isNpeak and self.isMask:
            for x in self.multix:
                for y in self.multiy:
                    if self.sqList[(x, y)] > eps and self.dataMask[x, y]:
                        try:
                            self.myFavoriatePlot(x, y)
                            newN = int(input('Enter number of peaks you think is real:'))
                            self.Npeak[x, y] = newN
                        except KeyboardInterrupt:
                            print("Force to stop!")
                            return 0
        
        return 1
    

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
            return [ypos, xpos]
        
    def shiftDat(self, dx = 0, dy = 0):
        (xx, yy, _) = np.shape(self.dat)
        newdat = np.ones(np.shape(self.dat))
        ##Move it to the correct position!
        for j in np.arange(xx):
            for k in np.arange(yy):
                if j+dy < xx and k+dx < yy:
                    newdat[int(j+dy),int(k+dx),:]=self.dat[j,k,:]
        self.dat=newdat

    def DEmap(self):
        if self.isMultfit and self.isNpeak:
            self.Dmap = np.zeros((self.X, self.Y))
            self.Emap = np.zeros((self.X, self.Y))
            for x in self.multix:
                for y in self.multiy:
                    theopt = self.optList[(x, y)]
                    if self.Npeak[x, y] == 2 and theopt is not None:
                        f1 = self.optList[(x, y)][3]
                        f2 = self.optList[(x, y)][6]
                        self.Dmap[x, y] = (f1 + f2)/2
                        self.Emap[x, y] = np.abs((f1 - f2)/2)
        return self.Dmap, self.Emap

    

# %%
