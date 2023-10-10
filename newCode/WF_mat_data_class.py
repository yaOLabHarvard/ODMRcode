# In[1]:
import WF_data_processing as dp
import multipeak_fit as mf
import numpy as np
# import matplotlib.pyplot as plt, cm
from matplotlib import pyplot as plt, cm
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.integrate import quad, simpson
from scipy.signal import find_peaks
from scipy.signal import correlate2d
import scipy.optimize as opt
from scipy.optimize import fsolve, root
import math

## important: only the pyplot.imshow will swap x and y axis as the output
# import scipy as sp
import random as rd
import os
# path = "F:/NMR/NMR/py_projects/WF/ODMRcode/WF/raw_data/"
# filename= '100xobj_Bz0p3A.mat'
# path = "C:/Users/esthe/OneDrive/Desktop/VSCode/Plotting/"
# filename = '100xobj_Bz3A.mat'
gamma=2.8025e-3 #GHz/G


##My comment

class WFimage:
    def __init__(self, filename):
        self.WFimagename = filename
        self.fVals, self.dat, self.xFrom, self.xTo, self.X, self.Y, self.npoints = mf.read_matfile(filename, normalize= False)
        print("mat file is loaded successfully!")
        self.originalDat = self.dat.copy()
        self.dataMask = None
        self.isNorm = False
        self.isMultfit = False
        self.isMask = False
        self.isNpeak = False
        self.binned=False

    def binning(self, binsize = 1):
        if binsize == 1:
            self.binOrigDat=self.dat
            self.binned=True
        elif binsize > 1:
            self.X=np.floor(self.X/binsize).astype(int)
            self.Y=np.floor(self.Y/binsize).astype(int)

            self.xr=np.arange(0,self.X,1)
            self.yr=np.arange(0,self.Y,1)

            newdat=np.zeros((self.X, self.Y, self.npoints))
            # sumdat=np.zeros((self.X, self.Y, self.npoints))
            for pxX in self.xr:
                for pxY in self.yr:
                    binyVals=[]
                    for binX in np.arange(0, binsize, 1):
                        for binY in np.arange(0, binsize, 1):
                            binyVals.append(self.originalDat[binsize*pxX+binX, binsize*pxY+binY, :])
                    binyVals=np.array(binyVals)
                    # print("binyVals shape: "+str(np.shape(binyVals)))
                    # print("binyVals length opposite: "+str(len(binyVals[:,0])))
                    avgeddat=[]
                    # summeddat=[]
                    for i in np.arange(self.npoints):
                        avgeddat.append(np.mean(binyVals[:,i]))
                        # summeddat.append(np.sum(binyVals[:,i]))
                    newdat[pxX,pxY,:]=avgeddat
                    # sumdat[pxX,pxY,:]=summeddat
            self.dat=newdat
            # self.dat=sumdat
            self.binOrigDat=newdat
            self.binned=True
        else:
            print("wrong binsize! please retry")


    def norm(self):
        self.dat = mf.normalize_widefield(self.dat, numpoints= 10)
        self.isNorm = True

    def sliceImage(self, Nslice = 3):
        return self.originalDat[:,:,Nslice].copy()
    
    def generateTmp(self, ROI = [[0, 1], [0, 1]], Nslice = 3):
        return self.dat[ROI[0][0]:ROI[0][1],ROI[1][0]:ROI[1][1],Nslice].copy()
    
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

    def myFavoriatePlot(self, x = 0, y = 0, withFit = True, fitParas = None):
        if not self.isNorm:
            self.norm()
        if not self.binned:
            self.binning()
        fig, ax = plt.subplots(nrows=1, ncols= 2, figsize= (12,6))
        ## plot the image
        # IMGplot = ax[0].imshow(self.originalDat[:,:,3].copy())
        IMGplot = ax[0].imshow(self.binOrigDat[:,:,3].copy())
        ax[0].add_patch(Rectangle((y - 2, x -2), 4, 4, fill=None, alpha = 1))
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(IMGplot, cax=cax)


        ## plot single ESR
        spESR = self.pointESR(x, y)
        ax[1].plot(self.fVals, spESR, '-')
        # ax[1].set_xlim()
        # ax[1].set_ylim(0.996,1.002)
        if withFit:
            if fitParas is None:
                peaks = self.singleESRpeakfind(x, y, method = 'pro')

                print("Peaks found by Autofind: "+str(self.fVals[peaks]))
                print("Peak indices found by Autofind: "+ str(peaks))

                popt, pcov, chisq = self.singleESRfit(x, y)
                print("the Chi square is {}".format(chisq))
                ax[1].plot(self.fVals[peaks], spESR[peaks], 'x')
            else:
                [popt, pcov, chisq] = fitParas
        
            try:
                for i in np.arange(int(np.floor(len(popt)/3))):
                    params= popt[1+3*i:4+3*i]
                    ax[1].plot(self.fVals,popt[0]+mf.lorentzian(self.fVals,*params), '-')
            except TypeError:
                print("No good fit was found! Try increase Maxfev or try a better initial guess")

        plt.show()
        plt.close()


    def waterfallPlot(self, lineCut = [[0, 0], [1, 1]], stepSize =1,  spacing = 0.01, plotTrace = False, plot = False):
        if self.isNorm:
            ## generate lineCut
            dx = lineCut[1][0]-lineCut[0][0]
            dy = lineCut[1][1]-lineCut[0][1]
            if dx > dy:
                nn = dx
                isX = True
                slope = dy/dx
            else:
                nn = dy
                isX = False
                slope = dx/dy

            nnList = np.arange(0, nn, stepSize)
            theLine = np.zeros((len(nnList),2), dtype=int)
            flag = 0
            for i in nnList:
                if isX:
                    pts = [lineCut[0][0] + i, int(round(lineCut[0][1] + i*slope))]
                else:
                    pts = [int(round(lineCut[0][0] + i*slope)), lineCut[0][1] + i]
            
                theLine[flag] = pts
                flag += 1

            if plotTrace:
                fig, ax = plt.subplots(nrows=1, ncols= 1, figsize= (6,6))
                IMGplot = ax.imshow(self.dat[:,:,3].copy())
                ## switch x and y for imshow
                print(theLine)
                ax.plot(theLine[:, 1], theLine[:, 0], '.', color='red', markersize=15)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(IMGplot, cax=cax)
                plt.show()
                plt.close()
            ## generate plots
            fig, ax = plt.subplots(nrows=1, ncols= 1, figsize= (6,12))
            offset = 0
            flag = 0
            for i in nnList:
                datax = self.fVals
                datay = self.pointESR(theLine[flag][0], theLine[flag][1])
                ymax = 1 - datay.min()
                offset += ymax + spacing
                ax.plot(datax, datay + offset, '-o', color = 'k', markersize=2)
                flag += 1

            ax.set_ylim([1, offset+1+spacing])

            if plot:
                plt.show()
                plt.close()
            
            return fig, ax


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

    def singleESRpeakfind(self, x = 0, y = 0, max_peak = 4, method = 'user'):
        if not self.isNorm:
            self.norm()
        if not self.binned:
            self.binning()
        yVals = self.pointESR(x, y)
        ymax = max(1 - yVals)
        # ymin = min(1 - yVals)
        if method == 'user':
            #for LaH:
            max_peak = 2
            peakPos, _ = find_peaks(1 - yVals, distance = 5,  height=(ymax/7, ymax), width = 2)

            #for LuH-N:
            # max_peak = 4
            # peakPos, _ = find_peaks(1 - yVals, distance = 5,  height=(ymax/7, ymax), width = 2)
            # print(peakPos)
            #for LT CeH9:
            # peakPos, _ = find_peaks(1 - yVals, distance = 59,  height=(ymax/4.05, ymax), width = 3)
            #for RT CeH9:
            # peakPos, _ = find_peaks(1 - yVals, distance = 40,  height=(ymax/4.05, ymax), width = 4)
        elif method == 'pro':
            # peakPos, _ = find_peaks(1 - yVals, prominence=0.1)
            # peakPos, _ = find_peaks(1 - yVals, prominence=(0.001,0.2))
            # if min(yVals)>0.992:
            if min(yVals)>0.996:
                # print('super low prom')
                # print('0.0008')
                peakPos, _ = find_peaks(1 - yVals, prominence=(0.0012,0.3))
            # elif min(yVals)<0.997 and min(yVals)>0.996:
            #     print('med-low prom')
            #     peakPos, _ = find_peaks(1 - yVals, prominence=(0.0014,0.3))
            elif min(yVals)<0.996 and min(yVals)>0.994:
                # print('low prom')
                peakPos, _ = find_peaks(1 - yVals, prominence=(0.0015,0.3))
            elif min(yVals)<0.994 and min(yVals)>0.990:
                # print('med prom')
                peakPos, _ = find_peaks(1 - yVals, prominence=(0.0018,0.3))
            elif min(yVals)<0.990 and min(yVals)>0.975:
                # print('high prom')
                peakPos, _ = find_peaks(1 - yVals, prominence=(0.0026,0.3))
            # # elif min(yVals)<0.992 and min(yVals)>0.985:
            else:
                # print('ultra high prom')
                peakPos, _ = find_peaks(1 - yVals, prominence=(0.004,0.4))
        peak_values = yVals[peakPos]
        posPeaks = np.array([i for i in peak_values if i > 0])
        #   Sort the peaks by amplitude (highest to lowest)
        sort_indices = np.argsort(posPeaks)[::-1]
        peak_indices_sorted = peakPos[sort_indices]
        if len(peakPos) < max_peak:
            return peak_indices_sorted
        else:
            top_peak = peak_indices_sorted[:max_peak]
            return top_peak

    def singleESRfit(self, x = 0, y = 0,max_peak = 4, autofind = True, initGuess = None):
        if not self.isNorm:
            print("normalize first to enable the fit")
            exit(0)
        else:
            if not self.binned:
                print("please bin")
                exit(0)
            else:
                yVals = self.pointESR(x, y)
                # print(yVals)
                if initGuess is not None:
                    self.guessFreq = initGuess              
                    self.peakPos = np.array([min(range(len(self.fVals)), key=lambda i: abs(self.fVals[i]-f)) for f in self.guessFreq])
                elif autofind:
                    self.peakPos = self.singleESRpeakfind(x, y, method='pro', max_peak = max_peak)
                    # print(self.peakPos)
                else:
                    self.guessFreq = np.fromstring(input('Enter frequency (for example: 2.71, 2.81, 2.91, 3.01):'),sep=',')
                    self.peakPos = np.array([min(range(len(self.fVals)), key=lambda i: abs(self.fVals[i]-f)) for f in self.guessFreq])
                ## generate real peaks based on the center freqs
                initParas = mf.generate_pinit(self.fVals[self.peakPos], yVals[self.peakPos])

                pOpt, pCov= mf.fit_data(self.fVals, yVals, init_params= initParas, fit_function= mf.lor_fit, maxFev=1000)
                # print(pOpt)
                self.pOptPrint=pOpt
                if pOpt is not None:
                    residuals = yVals - mf.lor_fit(self.fVals, *pOpt)
                    chiSq = np.sum((residuals / mf.lor_fit(self.fVals, *pOpt)) ** 2)
                else:
                    chiSq = 1
                return pOpt, pCov, chiSq

    def multiESRfit(self, xlist, ylist, max_peak = 4, initGuess = None):
        if not self.isNorm: #Added by Esther 20230524
            self.norm()
        if not self.binned:
            self.binning()
        self.optList = {}
        self.covList = {}
        self.sqList = {}
        self.multix = xlist
        self.multiy = ylist
        for x in self.multix:
            for y in self.multiy:
                if initGuess is not None:
                    # print('mutliesrfit')
                    pOpt, pCov, chiSq = self.singleESRfit(x, y,max_peak = max_peak, initGuess=initGuess)
                else:
                    pOpt, pCov, chiSq = self.singleESRfit(x, y,max_peak = max_peak)
                self.optList[(x, y)] = pOpt
                self.covList[(x, y)] = pCov
                self.sqList[(x, y)] = chiSq
            print("{}-th row has completed".format(x))

        self.isMultfit = True

    
    def multiESRfitManualCorrection(self, eps = 5e-5,  isResume = False):
        if self.isMultfit and self.isMask:
            if not isResume:
                xrange = self.multix
                yrange = self.multiy
            else:
                xindex = np.where(self.multix == self.resumeX)[0][0]
                xrange = self.multix[xindex:]
            for x in xrange:
                for y in yrange:
                    if self.sqList[(x, y)] > eps and self.dataMask[x, y]:
                        try:
                            self.myFavoriatePlot(x, y)
                            self.guessFreq = np.fromstring(input('Enter frequency (for example: 2.71, 2.81, 2.91, 3.01):'),sep=',')
                            pOpt, pCov, chiSq = self.singleESRfit(x, y,max_peak = len(self.guessFreq), initGuess=self.guessFreq)
                            self.myFavoriatePlot(x, y, fitParas=[pOpt, pCov, chiSq])
                            asw = int(input("Accept?(1/0)"))
                            if asw:
                                self.optList[(x, y)] = pOpt
                                self.covList[(x, y)] = pCov
                                self.sqList[(x, y)] = chiSq
                        except KeyboardInterrupt:
                            self.resumeX = x
                            print("Force to stop!")
                            return 0
        
        return 1

    
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
                    # else:
                    #     self.myFavoriatePlot(x, y)
            self.isNpeak = True
            return self.Npeak
        else:
            print("Run multiESRfit first to unlock this")

    def npeakManualCorrection(self, eps = 5e-5, isResume = False):
        if self.isNpeak and self.isMask:
            for x in self.multix:
                for y in self.multiy:
                    if self.sqList[(x, y)] > eps and self.dataMask[x, y]:
                        try:
                            self.myFavoriatePlot(x, y)
                            newN = int(input('Enter number of peaks you think is real:'))
                            self.Npeak[x, y] = newN
                        except KeyboardInterrupt:
                            self.resumeXY = (x, y)
                            print("Force to stop!")
                            return 0
        
        return 1
    
    def MWintmap(self, plot = False):
        if self.isNorm:
            (xx, yy) = np.shape(self.dat[:, :, 3])
            self.MWmap = np.zeros((xx, yy))
            fVals = self.fVals
            for x in range(xx):
                for y in range(yy):
                    yVals = np.array(self.dat[x,y])
                    self.MWmap[x,y] = simpson(1 - yVals, fVals)

        if plot:
            fig, ax = plt.subplots(nrows=1, ncols= 1, figsize= (6,6))
            img1 = ax.imshow(self.MWmap)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(img1, cax=cax)

            plt.show()
            plt.close()

    

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

    def imgCorr(self, bigImg, smallImg, roi, rr = 20, Nslice = 3, plot = False):

        corr=correlate2d(bigImg, smallImg, mode='same')
        localCorr = corr[roi[0][0]-rr:roi[0][1]+rr, roi[1][0]-rr:roi[1][1]+rr]
        xpos, ypos = np.unravel_index(np.argmax(localCorr), localCorr.shape)
        xpos += roi[0][0]-rr
        ypos += roi[1][0]-rr

        if plot:
            fig, ax = plt.subplots(nrows=1, ncols= 3, figsize= (6*3,6))
            img1 = ax[0].imshow(smallImg)
            divider = make_axes_locatable(ax[0])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(img1, cax=cax)

            cr = ax[1].imshow(corr)
            divider = make_axes_locatable(ax[1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(cr, cax=cax)

            #print("x is found at {} px; y is found at {} px".format(xpos, ypos))
            img2 = ax[2].imshow(bigImg)
            ax[2].add_patch(Rectangle((ypos-2, xpos-2), 4, 4, fill=None, alpha = 1, color = 'red'))
            divider = make_axes_locatable(ax[2])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(img2, cax=cax)

            plt.show()
            plt.close()
        return [xpos, ypos]
        
    def shiftDat(self, dx = 0, dy = 0):
        (xx, yy, _) = np.shape(self.dat)
        newdat = np.ones(np.shape(self.dat))
        ##Move it to the correct position!
        for j in np.arange(xx):
            for k in np.arange(yy):
                if j+dx < xx and k+dy < yy:
                    newdat[int(j+dx),int(k+dy),:]=self.dat[j,k,:]
        self.dat=newdat

    def DandE(self, realx, realy):
        if self.isMultfit:
            theopt = self.optList[(realx, realy)]
            if theopt is not None and len(self.optList[(realx, realy)][0::3][1:])>=1:
                peakFreqs = self.optList[(realx, realy)][0::3][1:]
                Fmax = max(peakFreqs)
                Fmin = min(peakFreqs)
                DD = (Fmin + Fmax)/2
                EE = np.abs((Fmax - Fmin)/2)
                return DD,EE
            else:
                return math.nan, math.nan
        else:
            print("Need to operate multiesrfit first!!")


    def DEmap(self):
        if self.isMultfit:
            self.Dmap = np.zeros((self.X, self.Y))
            self.Emap = np.zeros((self.X, self.Y))
            for x in self.multix:
                for y in self.multiy:
                    D, E =  self.DandE(x, y)
                    self.Dmap[x, y] = D
                    self.Emap[x, y] = E

            return self.Dmap, self.Emap
        else:
            print("Need to operate multiesrfit first!!")

    # def NVESRexactfit(self, Fitfreqs, isFourpeaks = True):
    #     ## freq unit in GHz; field unit in G
    #     Fitfreqs = np.sort(Fitfreqs)
    #     if isFourpeaks:
    #         newFitfreqs = np.zeros(8)
    #         newFitfreqs[0] = Fitfreqs[0]
    #         newFitfreqs[-1] = Fitfreqs[-1]
    #         newFitfreqs[1:4] = [Fitfreqs[1], Fitfreqs[1], Fitfreqs[1]]
    #         newFitfreqs[4:7] = [Fitfreqs[2], Fitfreqs[2], Fitfreqs[2]]

    #         func = mf.lsolver(newFitfreqs)
    #         ##print(func([0,0,50,2.87]))
    #         result = root(func, x0 = [0, 0, 50, 2.87], method='lm')
    #         return result.x
    #     else:
    #         print("not applicable right now")
    #         exit(0)
    
    def zfsTP(self, T, P):
        zfsT = 2.8771+ -4.625e-6*T+ 1.067e-7*T*T+ -9.325e-10*T*T*T+ 1.739e-12*T*T*T*T+ -1.838e-15*T*T*T*T*T
        zfsP = 1e-2*P
        return zfsT+zfsP
# %% class MultiFileWF

class multiWFImage:
    def __init__(self, folderpath):
        # self.temp=temp
        self.folderPath = folderpath
        filenamearr=os.listdir(self.folderPath)
        self.filenamearr=filenamearr.sort
        ##print(filenamearr)
        self.WFList=[]
        self.fileDir = {}
        self.Nfile = 0
        for filename in filenamearr:
            self.fileDir[self.Nfile] = filename
            tmpWF = WFimage(self.folderPath + filename)
            tmpWF.norm()
            tmpWF.binning()
            self.WFList.append(tmpWF)
            self.Nfile += 1
        print("WFList Size: " + str(self.Nfile))
        print("all mat files have been loaded successfully! The data has been normalized and binned!")
    
            
        self.isROI = False
        self.isAlign = False
        self.um=False #If false, it doesn't convert pixels to micron.
        self.isROIfit = False


    def addFile(self, filename = None):
        if filename is None:
            print("Current Iamge list: \n")
            print(self.fileDir)
        else:
            tmpWF = WFimage(self.folderPath + filename)
            tmpWF.norm()
            tmpWF.binning()
            self.WFList.append(tmpWF)
            self.fileDir[self.Nfile] = filename
            self.Nfile += 1
            print("The file has been added! Current Iamge list: \n")
            print(self.fileDir)

    def test(self):
        fig, ax = plt.subplots(nrows=self.Nfile, ncols= 1, figsize= (6,6*self.Nfile))
        for i in range(self.Nfile):
            tmp = self.WFList[i].dat
            testimg = tmp[:,:,3].copy()
            img = ax[i].imshow(testimg)
            divider = make_axes_locatable(ax[i])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(img, cax=cax)

        plt.show()

    def myFavoriatePlot(self, x = 0, y = 0, spacing = 0.01, withFit = True, fitParas = None):
        fig, ax = plt.subplots(nrows=1, ncols= 2, figsize= (12,6))
        ## plot the image
        # IMGplot = ax[0].imshow(self.originalDat[:,:,3].copy())
        Nslice = 3
        IMGplot = ax[0].imshow(self.imageStack(Nslice))
        ax[0].add_patch(Rectangle((y - 1, x - 1), 1, 1, fill=None, alpha = 1))
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(IMGplot, cax=cax)


        ## plot single ESR
        offset = 0
        for i in range(self.Nfile):
            tmpWF = self.WFList[i]
            spESR = tmpWF.pointESR(x, y)
            ymax = 1 - spESR.min()
            offset += ymax + spacing
            ax[1].plot(tmpWF.fVals, spESR + offset, '.', color='k', markersize = 2)

            if withFit:
                if fitParas is None:
                    peaks = tmpWF.singleESRpeakfind(x, y, method = 'pro')

                print("Peaks found by Autofind: "+str(tmpWF.fVals[peaks]))
                print("Peak indices found by Autofind: "+ str(peaks))

                popt, pcov, chisq = tmpWF.singleESRfit(x, y)
                print("the Chi square is {}".format(chisq))
                ax[1].plot(tmpWF.fVals[peaks], spESR[peaks]+offset, 'x')
            else:
                [popt, pcov, chisq] = fitParas
        
            try:
                for j in np.arange(int(np.floor(len(popt)/3))):
                    params= popt[1+3*j:4+3*j]
                    ax[1].plot(tmpWF.fVals,popt[0]+mf.lorentzian(tmpWF.fVals,*params)+offset, '-', color = 'r')
            except TypeError:
                print("No good fit was found! Try increase Maxfev or try a better initial guess")

        plt.show()
        plt.close()

    def imageStack(self, Nslice = 3, plot = False):
        for i in range(self.Nfile):
            tmpDat = self.WFList[i].dat.copy()
            if i == 0:
                stackImage = tmpDat[:,:, Nslice]
            else:
                stackImage += tmpDat[:,:, Nslice]
        stackImage = stackImage/self.Nfile 
        if plot:
            fig,ax=plt.subplots(1,1)
            ax.set_title("average image stack")
            imgs=ax.imshow(stackImage, interpolation=None)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(imgs, cax=cax)
            plt.show()
            plt.close()
        
        return stackImage

    def roi(self, xlow=0, ylow=0, xrange=None, yrange=None, plot = False):
        self.xlow=xlow
        self.ylow=ylow
        totalxr=len(self.WFList[0].dat[:][0][0])
        totalyr=len(self.WFList[0].dat[0][:][0])
        if xrange is None or yrange is None:
            self.xhigh=totalxr
            self.yhigh=totalyr
        else:
            self.xhigh=xlow+xrange
            self.yhigh=ylow+yrange
        self.xr=np.arange(self.xlow,self.xhigh,1)
        self.yr=np.arange(self.ylow,self.yhigh,1)
        self.rroi= [[self.xlow,self.xhigh],[self.ylow,self.yhigh]]
        self.isROI = True

        if plot:
            image = self.imageStack()
            fig,ax=plt.subplots(1,1)
            ax.set_title("average image stack with roi")
            imgs=ax.imshow(image, interpolation=None)
            ax.add_patch(Rectangle((self.ylow, self.xlow), self.yhigh-self.ylow, self.xhigh-self.xlow, fill=None, alpha = 1))
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(imgs, cax=cax)
            plt.show()
            plt.close()
    
    def normImg(self, img):
        img = np.array(img)
        maxx = img.max()
        minn = img.min()
        return (2*img - maxx - minn)/(maxx - minn)

    def imageAlign(self, nslice = 3, referN = 0, rr = 20, plot= False):
        if self.isROI:
            beforeAlignImg = self.imageStack(Nslice = nslice)
            tmpDat = self.WFList[referN].dat[:,:,nslice].copy()
            tmpDat = self.normImg(tmpDat)
            smallImg = tmpDat[self.rroi[0][0]:self.rroi[0][1], self.rroi[1][0]:self.rroi[1][1]].copy()
            shiftXY = []
            for i in range(self.Nfile):
                print("{}th image corr is processing...".format(i))
                tmpWF = self.WFList[i]
                tmp = tmpWF.dat[:,:, nslice].copy()
                bigImg = self.normImg(tmp)
                [rx, ry] = tmpWF.imgCorr(bigImg, smallImg, self.rroi, plot = False, rr = rr)
                print("WF {}: x is found at {} px; y is found at {} px".format(i , rx, ry))
                shiftXY.append([rx, ry])
            shiftXY = np.array(shiftXY)
            shiftXY = np.array([item - shiftXY[referN] for item in shiftXY])
            for i in range(self.Nfile):
                tmpWF = self.WFList[i]
                tmpWF.shiftDat(dx = -shiftXY[i][0], dy = -shiftXY[i][1])

            aftAlignImg = self.imageStack(Nslice = nslice) 

            fig, ax = plt.subplots(nrows=1, ncols= 2, figsize= (12,6))
            bef = ax[0].imshow(beforeAlignImg)
            divider = make_axes_locatable(ax[0])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(bef, cax=cax)

            aft = ax[1].imshow(aftAlignImg)
            divider = make_axes_locatable(ax[1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(aft, cax=cax)

            plt.show()
            plt.close()
            self.isAlign = True
        else:
            print("This function can only operate after assgining a roi!!")

    def roiMultiESRfit(self,max_peak = 4, initGuess = None):
        if self.isROI:
            for i in range(self.Nfile):
                tmpWF = self.WFList[i]
                print("Fitting {}th WFImage...".format(i))
                if initGuess is None:
                    tmpWF.multiESRfit(self.xr, self.yr, max_peak = max_peak)
                else:
                    tmpWF.multiESRfit(self.xr, self.yr, max_peak = max_peak, initGuess = initGuess[i])
            
            print("ROI fitting completed!")
            self.isROIfit = True
        else:
            print("This function can only operate after assgining a roi!!")

    def roiDEmap(self, plot = False):
        if self.isROIfit:
            for i in range(self.Nfile):
                tmpWF = self.WFList[i]
                self.roiDmap = np.zeros((len(self.xr), len(self.yr), self.Nfile))
                self.roiEmap = np.zeros((len(self.xr), len(self.yr), self.Nfile))
                xindex = 0
                yindex = 0
                for x in self.xr:
                    for y in self.yr:
                        D, E = tmpWF.DandE(x, y)
                        self.roiDmap[xindex, yindex, i] = D
                        self.roiEmap[xindex, yindex, i] = E

            if plot:
                fig, ax = plt.subplots(nrows=1, ncols= 1, figsize= (6,6))
                image = self.imageStack()
                ax.set_title("average image stack with roi")
                imgs=ax.imshow(image, interpolation=None)
                ax.add_patch(Rectangle((self.ylow, self.xlow), self.yhigh-self.ylow, self.xhigh-self.xlow, fill=None, alpha = 1))
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(imgs, cax=cax)

                fig, ax = plt.subplots(nrows=2, ncols= self.Nfile, figsize= (6*self.Nfile,6*2))
                for i in range(self.Nfile):
                    dmap = ax[0, i].imshow(self.roiDmap[:, :, i])
                    ax[0, i].set_title("roi D map")
                    divider = make_axes_locatable(ax[0, i])
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(dmap, cax=cax)

                    emap = ax[1, i].imshow(self.roiEmap[:, :, i])
                    ax[1, i].set_title("roi E map")
                    divider = make_axes_locatable(ax[1, i])
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(emap, cax=cax)

                plt.show()
                plt.close()
        else:
            print("This function can only operate after operating ROIfit!!")