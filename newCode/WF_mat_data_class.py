# In[1]:
##import WF_data_processing as dp
import multipeak_fit as mf
import numpy as np
import itertools
import random
# import matplotlib.pyplot as plt, cm
from matplotlib import pyplot as plt, cm
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.integrate import quad, simpson
from scipy.signal import find_peaks, argrelmin, correlate2d
from scipy.optimize import curve_fit, fsolve, root
from scipy.spatial import cKDTree
import pickle

## important: only the pyplot.imshow will swap x and y axis as the output
# import scipy as sp
import random as rd
import os
gamma=2.8025e-3 #GHz/G

## fitting parameters
epslion_y = .5e-3
peak_width = 0.015 ## GHz
max_repeat = 1000

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
        self.isSeedfit = False
        self.isErrordetect = False
        self.isManualFit = False
        self.isMask = False
        self.isNpeak = False
        self.isSeeded = False
        self.binned=False
        self.resumeX = 0
        self.FitCheck = 0
        self.presaveFig = None
        self.plotXx = 0
        self.plotYy = 0
        ## -2 means cannot fit error
        ## -1 means  bad fit occurs -- either positive amp or really broad (> 10x)
        ## 0 means default
        ## 1 means good fit but has not been corrected
        ## 2 means good fit and has been auto corrected
        ## 3 means good fit and has been manually corrected

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
        self.dat = mf.normalize_widefield(self.dat, numpoints= 3, from_back=False)
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

    def myFavoriatePlot(self, x = 0, y = 0, maxPeak = 6, withFit = True, fitParas = None):
        if not self.isNorm:
            self.norm()
        if not self.binned:
            self.binning()
        fig = plt.figure(num = 1, clear = True, figsize= (15,6))
        ax1 = fig.add_subplot(1, 2, 1)
        ## plot the image
        # IMGplot = ax[0].imshow(self.originalDat[:,:,3].copy())
        IMGplot = ax1.imshow(self.dat[:,:,3])
        ax1.add_patch(Rectangle((y - 2, x -2), 4, 4, fill=None, alpha = 1))
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(IMGplot, cax=cax)


        ## plot single ESR
        ax2 = fig.add_subplot(1, 2, 2)
        spESR = self.pointESR(x, y)
        ax2.plot(self.fVals, spESR, '-')
        # ax2.set_xlim()
        # ax2.set_ylim(0.996,1.002)
        if withFit:
            if fitParas is None:
                peaks = self.singleESRpeakfind(x, y, max_peak = maxPeak, method = 'pro')
                ax2.plot(self.fVals[peaks], spESR[peaks], 'x')
                print("Peaks found by Autofind: "+str(self.fVals[peaks]))
                print("Peak indices found by Autofind: "+ str(peaks))

                try:
                    popt, pcov, chisq = self.singleESRfit(x, y, max_peak = maxPeak)
                    print("the Chi square is {}".format(chisq))
                except ValueError:
                    print("cannot find singleesr")
                    popt = None
            else:
                [popt, pcov, chisq] = fitParas

            if popt is not None:
                npeak = int(np.floor(len(popt)/3))
            elif fitParas is None:
                npeak = len(peaks)
                popt = mf.generate_pinit(self.fVals[peaks], spESR[peaks])
            else:
                npeak = 0

            for i in range(npeak):
                params = popt[1+3*i:4+3*i]
                print("peak number {}; amp {:.3e}; width {:.3e}; freq {:.3e};".format(i, popt[3*i+1], popt[3*i+2], popt[3*i+3]))
                ax2.plot(self.fVals,popt[0]+mf.lorentzian(self.fVals,*params), '-')

        plt.show()

    def onpick(self, event):
        self.plotXx = int(event.xdata) 
        self.plotYy = int(event.ydata)
        ## x and y s are flipped in imshow
        print("Picked x {}; y {}".format(self.plotYy, self.plotXx))
        plt.clf()
        self.myFavoriatePlotMousepick()
        plt.show()
        plt.draw()

    def myFavoriatePlotMousepick(self, maxPeak = 6, withFit = True, fitParas = None, isChoosefig = False):
        if not self.isNorm:
            self.norm()
        if not self.binned:
            self.binning()
        fig = plt.figure(num = 1, clear = True, figsize= (15,6))
        ax1 = fig.add_subplot(1, 2, 1)
        ## plot the image
        # IMGplot = ax[0].imshow(self.originalDat[:,:,3].copy())
        if isChoosefig and self.presaveFig is not None:
            thefig = self.presaveFig
        else:
            thefig = self.originalDat[:,:,3].copy()

        IMGplot = ax1.imshow(thefig)
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(IMGplot, cax=cax)
    
        ## add the cross
        ax1.hlines(y = self.plotYy, xmin=self.plotXx-10, xmax=self.plotXx+10, linewidth=1, color='r')
        ax1.vlines(x = self.plotXx, ymin=self.plotYy-10, ymax=self.plotYy+10, linewidth=1, color='r')

        ## plot single ESR
        ax2 = fig.add_subplot(1, 2, 2)
        spESR = self.pointESR(self.plotYy, self.plotXx)
        ax2.plot(self.fVals, spESR, '-')
        # ax2.set_xlim()
        # ax2.set_ylim(0.996,1.002)
        if withFit:
            if fitParas is None:
                peaks = self.singleESRpeakfind(self.plotYy, self.plotXx, max_peak = maxPeak, method = 'pro')
                ax2.plot(self.fVals[peaks], spESR[peaks], 'x')
                print("Peaks found by Autofind: "+str(self.fVals[peaks]))
                print("Peak indices found by Autofind: "+ str(peaks))

                try:
                    popt, pcov, chisq = self.singleESRfit(self.plotYy, self.plotXx, max_peak = maxPeak)
                    print("the Chi square is {}".format(chisq))
                except ValueError:
                    print("cannot find singleesr")
                    popt = None
            else:
                [popt, pcov, chisq] = fitParas

            if popt is not None:
                npeak = int(np.floor(len(popt)/3))
            elif fitParas is None:
                npeak = len(peaks)
                popt = mf.generate_pinit(self.fVals[peaks], spESR[peaks])
            else:
                npeak = 0

            for i in range(npeak):
                params = popt[1+3*i:4+3*i]
                print("peak number {}; amp {:.3e}; width {:.3e}; freq {:.3e};".format(i, popt[3*i+1], popt[3*i+2], popt[3*i+3]))
                ax2.plot(self.fVals,popt[0]+mf.lorentzian(self.fVals,*params), '-')

        fig.canvas.mpl_connect('button_press_event', self.onpick)
        plt.show()
        plt.draw()


    def waterfallPlot(self, lineCut = [[0, 0], [1, 1]], stepSize =1,  spacing = 0.01, plotTrace = False, plotFit = False):
        
            theLine, nnList = lineCutGen(lineCut = lineCut, stepSize = stepSize)
            if plotTrace:
                fig = plt.figure(num = 1, clear = True, figsize= (6,6))
                ax = fig.add_subplot(1, 1, 1)
                IMGplot = ax.imshow(self.dat[:,:,3].copy())
                ## switch x and y for imshow
                print(theLine)
                ax.plot(theLine[:, 1], theLine[:, 0], '.', color='red', markersize=15)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(IMGplot, cax=cax)
                plt.show()

            ## generate plots
            fig = plt.figure(num = 1, clear = True, figsize= (6,6))
            ax = fig.add_subplot(1, 1, 1)
            offset = 0
            for i in range(len(nnList)):
                datax = self.fVals
                datay = self.pointESR(theLine[i][0], theLine[i][1])
                ymax = 1 - datay.min()
                offset += ymax + spacing
                ax.plot(datax, datay + offset, '.', color = 'k', markersize=2)
                if plotFit and self.isMultfit:
                    popt = self.optList[(theLine[i][0], theLine[i][1])]
                    try:
                        for j in np.arange(int(np.floor(len(popt)/3))):
                            params= popt[1+3*j:4+3*j]
                            ax.plot(self.fVals,popt[0]+mf.lorentzian(self.fVals,*params)+ offset, '-', color = 'r')
                    except TypeError:
                        print("No good fit was found! Try increase Maxfev or try a better initial guess")

            ax.set_ylim([1, offset+1+spacing])

            plt.show()


    def waterfallMap(self, lineCut = [[0, 0], [1, 1]], stepSize =1, plotTrace = False, localmin = False, flipped = False):
        
            theLine, nnList = lineCutGen(lineCut = lineCut, stepSize = stepSize)
            if flipped:
                theLine = theLine[::-1]
            if plotTrace:
                fig = plt.figure(num = 1, clear = True, figsize= (6,6))
                ax = fig.add_subplot(1, 1, 1)
                IMGplot = ax.imshow(self.dat[:,:,3].copy())
                ## switch x and y for imshow
                ##print(theLine)
                ax.plot(theLine[:, 1], theLine[:, 0], '.', color='red', markersize=15)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(IMGplot, cax=cax)
                plt.show()

            ## generate plots
            theimage = np.zeros((len(nnList), self.npoints))
            for i in range(len(nnList)):
                theimage[i] = self.pointESR(theLine[i][0], theLine[i][1])
            fig = plt.figure(num = 1, clear = True, figsize= (12,6))
            ax = fig.add_subplot(1, 1, 1)
            themap = ax.imshow(theimage, extent=[self.fVals[0], self.fVals[-1], 0, len(nnList)], aspect='auto', cmap='rainbow')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(themap, cax=cax)
            plt.show()


            if localmin:
                initFreq = np.fromstring(input("input the init pos (for example: 10, 20, 30):"), sep=',')
                span = int(input("input the search span you wish:"))
                Nmin = len(initFreq)
                minArray = np.zeros((len(nnList), Nmin))
                curentFreq = initFreq
                for i in range(len(nnList)):
                    array = self.pointESR(theLine[i][0], theLine[i][1])
                    tmpMin = []
                    for j in range(len(curentFreq)):
                        try:
                            lowf = int(curentFreq[j]-span)
                            highf = int(curentFreq[j]+span)
                            findrange = array[lowf:highf]
                            ##print(findrange)
                            theMin = np.array(argrelmin(findrange, order = int(span/4)))
                            theMin = theMin+ lowf
                            tmpMin.append(theMin)
                        except IndexError:
                            print("the range is twoo big!")
                            return
                    tmpMin = np.hstack(tmpMin)
                    tmpMin = np.sort(np.unique(tmpMin))
                    minValsorder = array[tmpMin].argsort()
                    finalMin = tmpMin[minValsorder]
                    if len(finalMin) >= Nmin:
                        curentFreq = finalMin[:Nmin]
                        minArray[i] = finalMin[:Nmin]
                    else:
                        for k in range(Nmin):
                            if k < len(finalMin):
                                minArray[i][k] = finalMin[k]
                            else:
                                minArray[i][k] = finalMin[-1]



                        
                minArray = minArray.transpose()
                ydata = np.arange(len(nnList))
                plt.imshow(theimage)
                for i in range(len(curentFreq)):
                    plt.scatter(minArray[i], ydata)

                plt.show()



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
            if min(yVals)>0.998:
                # print('super low prom')
                # print('0.0008')
                peakPos, _ = find_peaks(1 - yVals, prominence=(0.0005,0.1))
            # elif min(yVals)<0.997 and min(yVals)>0.996:
            #     print('med-low prom')
            #     peakPos, _ = find_peaks(1 - yVals, prominence=(0.0014,0.3))
            elif min(yVals)<0.998 and min(yVals)>0.995:
                # print('low prom')
                peakPos, _ = find_peaks(1 - yVals, prominence=(0.00075,0.2))
            elif min(yVals)<0.995 and min(yVals)>0.990:
                # print('med prom')
                peakPos, _ = find_peaks(1 - yVals, prominence=(0.001,0.3))
            elif min(yVals)<0.990 and min(yVals)>0.975:
                # print('high prom')
                peakPos, _ = find_peaks(1 - yVals, prominence=(0.005,0.3))
            # # elif min(yVals)<0.992 and min(yVals)>0.985:
            else:
                # print('ultra high prom')
                peakPos, _ = find_peaks(1 - yVals, prominence=(0.01,0.4))
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

    def singleESRfit(self, x = 0, y = 0,max_peak = 4, epsy = epslion_y, autofind = True, initGuess = None):
        if not self.isNorm:
            print("normalize first to enable the fit")
            exit(0)
        else:
            if not self.binned:
                print("please bin")
                exit(0)
            else:
                yVals = self.pointESR(x, y)
                if yVals.max() - yVals.min() < epsy:
                    print("data is too flat! skipping..")
                    return None, None, None
                # print(yVals)
                if initGuess is not None:
                    self.guessFreq = initGuess              
                    self.peakPos = np.array([min(range(len(self.fVals)), key=lambda i: abs(self.fVals[i]-f)) for f in self.guessFreq])
                    ##print(self.peakPos)
                elif autofind:
                    self.peakPos = self.singleESRpeakfind(x, y, method='pro', max_peak = max_peak)
                    # print(self.peakPos)
                else:
                    self.guessFreq = np.fromstring(input('Enter frequency (for example: 2.71, 2.81, 2.91, 3.01):'),sep=',')
                    self.peakPos = np.array([min(range(len(self.fVals)), key=lambda i: abs(self.fVals[i]-f)) for f in self.guessFreq])
                    ##print(self.peakPos)
                ## generate real peaks based on the center freqs
                initParas = mf.generate_pinit(self.fVals[self.peakPos], yVals[self.peakPos])

                if initGuess is None and autofind is True:## initGuess is not None or (initGuess is None and autofind is False):
                    try:
                        pOpt, pCov= mf.fit_data(self.fVals, yVals, init_params= initParas, fit_function= mf.lor_fit, maxFev=max_repeat)
                    except ValueError:
                        print("auto singleESRfit fails.. set opt to be none")
                        self.FitCheck = -2
                        pOpt = None
                        pCov = None

                else:
                    try:
                        pOpt, pCov= mf.fit_data(self.fVals, yVals, init_params= initParas, fit_function= mf.lor_fit, maxFev=max_repeat)
                    except ValueError:
                        self.FitCheck = -2
                        print("The guess singleESRfit fails.. Now will try auto method")
                        self.peakPos = self.singleESRpeakfind(x, y, method='pro', max_peak = max_peak)
                        initParas = mf.generate_pinit(self.fVals[self.peakPos], yVals[self.peakPos])
                
                        try:
                            pOpt, pCov= mf.fit_data(self.fVals, yVals, init_params= initParas, fit_function= mf.lor_fit, maxFev=max_repeat)
                        except ValueError:
                            print("auto singleESRfit fails.. set opt to be none")
                            self.FitCheck = -2
                            pOpt = None
                            pCov = None


                # print(pOpt)
                self.pOptPrint=pOpt
                if pOpt is not None:
                    residuals = yVals - mf.lor_fit(self.fVals, *pOpt)
                    chiSq = np.sum((residuals**2 / mf.lor_fit(self.fVals, *pOpt)))
                    cVals= pOpt[1::3]
                    widthcompare = [a/b for a,b in zip(pOpt[2::3], initParas[2::3])]
                    if len([*filter(lambda x: x > 0, cVals)]) > 0 or len([*filter(lambda x: x > 10, widthcompare)]) > 0:
                        self.FitCheck = -1
                    else:
                        self.FitCheck = 1
                else:
                    chiSq = 1
                return pOpt, pCov, chiSq

    def multiESRfit(self, xlist, ylist, max_peak = 4, epsy = epslion_y, initGuess = None, lineFit = False):
        if not self.isNorm: #Added by Esther 20230524
            self.norm()
        if not self.binned:
            self.binning()
        if not self.isMultfit:
            self.optList = {}
            self.covList = {}
            self.sqList = {}
            self.ckList = {}
        self.multix = xlist
        self.multiy = ylist
        guessFreqs = initGuess
        if lineFit:
           theloopList = zip(xlist, ylist)
           totalLen = len(xlist)
        else:
            theloopList = itertools.product(xlist, ylist)
            totalLen = len(xlist)*len(ylist)

        flag = 0
        for (x,y) in theloopList:
                if initGuess is not None:
                    try:
                        pOpt, pCov, chiSq = self.singleESRfit(x, y,max_peak = max_peak, epsy = epsy, initGuess=guessFreqs)
                        if pOpt is not None:
                            guessFreqs = pOpt[0::3][1:]
                        else:
                            guessFreqs = initGuess
                    except ValueError:
                        print("manual guess fails.. trying init guess")
                        try:
                            pOpt, pCov, chiSq = self.singleESRfit(x, y,max_peak = max_peak, epsy = epsy, initGuess=initGuess)
                        except ValueError:
                            print("init guess fails.. trying auto guess")
                            try:
                                pOpt, pCov, chiSq = self.singleESRfit(x, y,max_peak = max_peak, epsy = epsy)
                            except ValueError:
                                print("auto guess fails.. Popt set to be none")
                                

                    
                        
                else:
                    try:
                        pOpt, pCov, chiSq = self.singleESRfit(x, y,epsy = epsy, max_peak = max_peak)
                    except ValueError:
                        print("auto guess fails.. Popt set to be none")
                        pOpt = None
                        pCov = None
                        chiSq = 1

                self.optList[(x, y)] = pOpt
                self.covList[(x, y)] = pCov
                self.sqList[(x, y)] = chiSq
                self.ckList[(x, y)] = self.FitCheck

                ## counting the progress
                flag += 1
                pp = int(flag/totalLen*100)
                if pp%5 == 0:
                    print("{} percent has completed".format(pp))

        self.isMultfit = True
    
    def plotAndCorrect(self, x, y):
        print("current point: x {}; y {}".format(x, y))
        self.myFavoriatePlot(x, y, fitParas=[self.optList[(x, y)], self.covList[(x, y)], self.sqList[(x, y)]])
        guessFreq = np.fromstring(input('Enter frequency (for example: 2.71, 2.81, 2.91, 3.01):'),sep=',')
        pOpt, pCov, chiSq = self.singleESRfit(x, y,max_peak = len(guessFreq), initGuess=guessFreq)
        return [pOpt, pCov, chiSq]
    
    def CorrectQ(self, x, y, fitList):
        print("current point: x {}; y {}".format(x, y))
        self.myFavoriatePlot(x, y, fitParas=fitList)
        asw1 = int(input("Looks good?(0/1)"))
        if asw1 == 1:
            return asw1
        else:
            self.fitList = self.plotAndCorrect(x, y)            
            self.myFavoriatePlot(x, y, fitParas=self.fitList)
            asw2 = int(input("Accept?(-1/0/1)"))                
            return asw2
    
    def fitErrordetection(self, xyarray, epschi = 5e-5, epsy = epslion_y):
        if self.isMultfit:
            self.errorIndex = []
            for [x, y] in xyarray:
                if self.sqList[(x, y)] is None:
                    self.sqList[(x, y)] = 1
                if self.sqList[(x, y)] > epschi and min(self.dat[x, y]) < 1 - epsy and self.ckList[(x, y)] < 3:
                    self.errorIndex.append([x,y])
            
            self.isErrordetect = True
        else:
            print("Please do multiesr first!")

    def deleteFitpeaks(self, x, y):
        fitList = self.optList[(x, y)]
        if self.optList[(x, y)] is not None:
            peakNumber = np.arange(int((len(fitList) - 1)/3))
            for i in peakNumber:

                flag = 3*i+1
                print("peak number {}; amp {:.3e}; width {:.3e}; freq {:.3e};".format(i, fitList[flag], fitList[flag+1], fitList[flag+2]))

            nums = np.fromstring(input("input the peak number you wish to delete (for example: 0, 1, 2):"), sep=',')
            nums = np.array(nums)
            try:
                dlList =np.zeros(len(nums)*3, dtype = int)
                for i in range(len(nums)):

                    flag = 3*nums[i]+1
                    dlList[3*i:3*i+3]=[flag, flag+1, flag+2]

                self.optList[(x, y)] = np.delete(self.optList[(x, y)], dlList)


            except (ValueError, TypeError, IndexError) as e:
                print("no deletion! exit")
                return
            

    def multiESRfitManualCorrection(self, isResume = False, seedScan = False):
        if isResume:
            currentIndex = self.resumeIndex
        else:
            currentIndex = 0
            
        if seedScan and self.isSeeded:
            errorList = self.seedList[currentIndex:]
        elif self.isErrordetect:
            errorList = self.errorIndex[currentIndex:]
        else:
            print("not responding! Do errordetection or seedgeneration first!!!")

        for [x, y] in errorList:
            try:
                isRetry = True
                self.fitList = [self.optList[(x, y)], self.covList[(x, y)], self.sqList[(x, y)]]
                while isRetry:
                    condition = self.CorrectQ(x, y, self.fitList)
                    ##fitList = self.optList[(x, y)]
                    if condition == 1:
                        self.optList[(x, y)] = self.fitList[0]
                        self.covList[(x, y)] = self.fitList[1]
                        self.sqList[(x, y)] = self.fitList[2]
                        self.ckList[(x, y)] = 3

                        isRetry = False
                    elif condition < 0:
                        self.deleteFitpeaks(x, y)
                    else:
                        self.fitList = self.plotAndCorrect(x, y)
                        

                    # asw1, fitList = self.plotAndCorrect(x, y)
                    # if asw1 == 1:
                    #     self.optList[(x, y)] = fitList[0]
                    #     self.covList[(x, y)] = fitList[1]
                    #     self.sqList[(x, y)] = fitList[2]
                    #     self.ckList[(x, y)] = 3
                    #     asw2 = 0
                    # elif asw1 == 2:
                    #     self.optList[(x, y)] = fitList[0]
                    #     self.deleteFitpeaks(x, y)
                    #     asw2, fitList = self.plotAndCorrect(x, y)
                    # elif asw1 == 0:
                    #     if fitList == 0:
                    #         asw2 = 0
                    #     else:
                    #         asw2 = int(input("Want to retry?(1/0)"))

                    # if asw2 == 0:
                    #     isRetry = 0

            except(ValueError, RuntimeError, IndexError) as e:
                print(e)
                self.resumeIndex = currentIndex
                print("Force to stop!")
                return 0
                
            currentIndex += 1
        self.resumeIndex = 0
        self.isManualFit = True
        print("manual error correction finished!")


    def randomSeedGen(self, xyarray, pointRatio = 0.01, plot = False):
               
        randomN = int(pointRatio*len(xyarray))
        self.seedList = np.zeros((randomN, 2), dtype=int)
        for i in range(randomN):
            self.seedList[i] = random.choice(xyarray)

        if plot:
            fig = plt.figure(num = 1, clear = True, figsize= (6,6))
            ax = fig.add_subplot(1, 1, 1)
            IMGplot = ax.imshow(self.dat[:,:,3].copy())
            sx,sy = self.seedList.transpose()
            ax.scatter(sx, sy, color = 'r')

            plt.show()
        
        self.isSeeded = True
    
    def fitCheckQ(self, xyarray):
        correctArray = []
        for [x, y] in xyarray:
            if self.ckList[(x, y)]<2:
                correctArray.append([x, y])
        
        return correctArray

    def multiESRSeedfit(self, xlist, ylist, iter = 5, dist = 3, epschi = 5e-5, debug = True):
        if not self.isNorm: #Added by Esther 20230524
            self.norm()
        ## initialize checks
        for i in xlist:
            for j in ylist:
                self.ckList[(i,j)] = 0
        
        if self.isSeeded and self.isManualFit:
            bathArray = np.array(list(itertools.product(xlist, ylist)))
            theTree = cKDTree(bathArray)
            ##currentLevel = []
            for it in range(iter):
                currentLevel = []
                for [x, y] in self.seedList:
                    theGroup = bathArray[theTree.query_ball_point([x, y], dist)]
                    theGroup = self.fitCheckQ(theGroup)
                    currentLevel.append(theGroup)
                    for [xx, yy] in theGroup:
                        theGuess = self.optList[(x, y)][3::3]
                        try:
                            pOpt, pCov, chiSq = self.singleESRfit(xx, yy, initGuess=theGuess)
                            self.FitCheck = 2    
                        except ValueError:
                            print("The fitting fails.. please correct!")
                            pOpt = None
                            pCov = None
                            chiSq = 1
                            self.FitCheck = -2
                                                                          
                        self.optList[(xx, yy)] = pOpt
                        self.covList[(xx, yy)] = pCov
                        self.sqList[(xx, yy)] = chiSq
                        self.ckList[(xx, yy)] = self.FitCheck
                    print("{} {} has completed".format(x, y))
                
                currentLevel = [item for sublist in currentLevel for item in sublist]
                self.seedList = np.array(currentLevel)
                if debug:
                    fig = plt.figure(num = 1, clear = True, figsize= (6,6))
                    ax = fig.add_subplot(1, 1, 1)
                    IMGplot = ax.imshow(self.dat[:,:,3].copy())
                    sx,sy = self.seedList.transpose()
                    ax.scatter(sx, sy, color = 'r')
                
                self.fitErrordetection(self.seedList, epschi = 1e-3)
                self.multiESRfitManualCorrection(isResume = False, seedScan = False)




        self.isSeedfit = True

    
    def multiESRfitAutoCorrection(self, guessFreq, isResume = False, forced = False):
        if self.isErrordetect:
            if isResume:
                currentIndex = self.resumeIndex
            else:
                currentIndex = 0
            errorList = self.errorIndex[currentIndex:]
            for [x, y] in errorList:
                try:
                    pOpt, pCov, chiSq = self.singleESRfit(x, y,max_peak = len(guessFreq), initGuess=guessFreq)
                    if chiSq is None:
                        chiSq = 1
                    if forced:
                        condition = True
                    else:
                        condition = (chiSq < self.sqList[(x, y)])

                    if condition:
                        self.optList[(x, y)] = pOpt
                        self.covList[(x, y)] = pCov
                        self.sqList[(x, y)] = chiSq
                        self.ckList[(x, y)] = 2
                        print("{} {} has been auto corrected..".format(x, y))

                except(ValueError, RuntimeError, IndexError) as e:
                    print("{} {} is not working..".format(x, y))
                    continue
            print("auto error correction finished!")
    
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
            fig = plt.figure(num = 1, clear = True, figsize= (6,6))
            ax = fig.add_subplot(1, 1, 1)
            img1 = ax.imshow(self.MWmap)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(img1, cax=cax)

            plt.show()


    

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
            fig = plt.figure(num = 1, clear = True, figsize= (6*3,6))
            ax1 = fig.add_subplot(1, 3, 1)
            img1 = ax1.imshow(smallImg)
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(img1, cax=cax)
            
            ax2 = fig.add_subplot(1, 3, 2)
            cr = ax2.imshow(corr)
            divider = make_axes_locatable(ax2)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(cr, cax=cax)

            #print("x is found at {} px; y is found at {} px".format(xpos, ypos))
            ax3 = fig.add_subplot(1, 3, 3)
            img2 = ax3.imshow(bigImg)
            ax3.add_patch(Rectangle((ypos-2, xpos-2), 4, 4, fill=None, alpha = 1, color = 'red'))
            divider = make_axes_locatable(ax3)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(img2, cax=cax)

            plt.show()

        return [xpos, ypos]
        
    def shiftDat(self, dx = 0, dy = 0):
        if dx == 0 and dy == 0:
            print("no shifting!")
            return
        else:
            (xx, yy, _) = np.shape(self.dat)
            newdat = np.ones(np.shape(self.dat))
            ##Move it to the correct position!
            for j in np.arange(xx):
                for k in np.arange(yy):
                    jj = int((j+dx) % xx)
                    kk = int((k+dy) % yy)
                    newdat[jj, kk,:]=self.dat[j,k,:]
            self.dat=newdat

    def DandE(self, realx, realy):
        if self.isMultfit:
            theopt = self.optList[(realx, realy)]
            
            if theopt is not None and len(theopt[0::3][1:])>=1:
                peakFreqs = theopt[0::3][1:]
                Fmax = max(peakFreqs)
                Fmin = min(peakFreqs)
                DD = (Fmin + Fmax)/2
                EE = np.abs((Fmax - Fmin)/2)
                return DD,EE
            else:
                return 0, 0
        else:
            print("Need to operate multiesrfit first!!")
            return -99,-99

    def customDandE(self, realx, realy, peakNumber = [0,-1]):
        if self.isMultfit:
            theopt = self.optList[(realx, realy)]
            
            if theopt is not None and len(theopt[0::3][1:])>=1:
                peakFreqs = np.sort(theopt[0::3][1:])
                try:
                    Fmax = peakFreqs[int(peakNumber[0])]
                    Fmin = peakFreqs[int(peakNumber[-1])]
                except IndexError:
                    print("The requested d e does not exist! Current peak number: {}".format(len(peakFreqs)))
                    Fmax = -99
                    Fmin = -99
                DD = (Fmin + Fmax)/2
                EE = np.abs((Fmax - Fmin)/2)
                return DD,EE
            else:
                return 0, 0
        else:
            print("Need to operate multiesrfit first!!")
            return -99,-99

    def theWidth(self, realx, realy):
        if self.isMultfit:
            theopt = self.optList[(realx, realy)]
            if theopt is not None and len(theopt[2::3])>=1:
                peakWidth = theopt[2::3]
                Wfirst = peakWidth[0]
                Wlast = peakWidth[-1]

                return (Wfirst + Wlast)/2
            else:
                return 0
        else:
            print("Need to operate multiesrfit first!!")


    def DEmap(self, plot = False, iscustom = False):
        if self.isMultfit:
            self.Dmap = np.zeros((self.X, self.Y))
            self.Emap = np.ones((self.X, self.Y))
            self.Dmax = 0
            self.Dmin = 1e6
            self.Emax = 0
            self.Emin = 1e6
            if iscustom:
                theList = np.fromstring(input('Enter peak numbers (for example: 0, 3):'),sep=',')
            for x in self.multix:
                for y in self.multiy:
                    if iscustom:
                        D, E =  self.customDandE(x, y, peakNumber=theList)
                    else:
                        D, E =  self.DandE(x, y)
                    if D:
                        if D > self.Dmax:
                            self.Dmax = D
                        elif D < self.Dmin:
                            self.Dmin = D
                    if E:
                        if E > self.Emax:
                            self.Emax = E
                        elif E < self.Emin:
                            self.Emin = E
                    self.Dmap[x, y] = D
                    self.Emap[x, y] = E

            if plot:
                
                fig = plt.figure(num = 1, clear = True, figsize= (15,6))
                ax = fig.add_subplot(1, 2, 1)
                img1 = ax.imshow(self.Dmap, vmax = self.Dmax, vmin = self.Dmin)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(img1, cax=cax)

                ax = fig.add_subplot(1, 2, 2)
                img2 = ax.imshow(self.Emap, vmax = self.Emax, vmin = self.Emin)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(img2, cax=cax)

                plt.show()


            return self.Dmap, self.Emap
        else:
            print("Need to operate multiesrfit first!!")

    def DElineplot(self, lineCut = [[0, 0], [1, 1]], stepSize =1, plotTrace = False, plotD = False, plotE = False):
        if self.isMultfit:
            theLine, nnList = lineCutGen(lineCut = lineCut, stepSize = stepSize)
            if plotTrace:
                fig = plt.figure(num = 1, clear = True, figsize= (6,6))
                ax = fig.add_subplot(1, 1, 1)
                IMGplot = ax.imshow(self.dat[:,:,3].copy())
                ## switch x and y for imshow

                ax.plot(theLine[:, 1], theLine[:, 0], '.', color='red', markersize=15)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(IMGplot, cax=cax)
                plt.show()

                ## generate plots
            fig = plt.figure(num = 2, clear = True, figsize= (15,6))
            ax = fig.add_subplot(1, 1, 1)
            Dlist = []
            Elist = []
            for i in range(len(nnList)):
                DD, EE = self.DandE(theLine[i][0], theLine[i][1])
                
                Dlist.append(DD)
                Elist.append(EE)
            if plotD:
                ax.plot(nnList, Dlist, '.', color = 'r', markersize=2, label = "D (GHz)")
                ax.set_ylim(0, max(Dlist)*1.01)
            if plotE:
                ax.plot(nnList, Elist, '.', color = 'b', markersize=2, label = "E (GHz)")
                ax.set_ylim(0, max(Elist)*1.01)

            

            plt.show()



    def DEwidthplot(self, lineCut = [[0, 0], [1, 1]], stepSize =1, plotTrace = False):
        if self.isMultfit:
            theLine, nnList = lineCutGen(lineCut = lineCut, stepSize = stepSize)
            if plotTrace:
                fig = plt.figure(num = 1, clear = True, figsize= (6,6))
                ax = fig.add_subplot(1, 1, 1)
                IMGplot = ax.imshow(self.dat[:,:,3].copy())
                ## switch x and y for imshow

                ax.plot(theLine[:, 1], theLine[:, 0], '.', color='red', markersize=15)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(IMGplot, cax=cax)
                plt.show()
                plt.close()
                ## generate plots
            fig = plt.figure(num = 2, clear = True, figsize= (6,12))
            ax = fig.add_subplot(1, 1, 1)
            widthList = []
            for i in range(len(nnList)):
                width = self.theWidth(theLine[i][0], theLine[i][1])
               
                widthList.append(width)

            ax.plot(nnList, widthList, '.', color = 'r', markersize=2, label = "Width (GHz)")

            plt.show()


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
        filenamearr = [f for f in os.listdir(self.folderPath) if os.path.isfile(self.folderPath + f)]
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
    
        (self.X, self.Y, _) = np.shape(self.WFList[0].dat)
        self.isROI = False
        self.isAlign = False
        self.um=False #If false, it doesn't convert pixels to micron.
        self.isROIfit = False
        self.ROIfitloaded = False
        self.ROIfitsaved = False
        self.isDEmap = False
        self.isPara = False

        self.imgShift = np.zeros((self.Nfile, 2))
        self.roiShape = 'point'


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
            np.append(self.imgShift,[[0,0]],axis = 0)
            print("The file has been added! Current Iamge list: \n")
            print(self.fileDir)

    def setFileParameters(self, parameters = None):
        print("The current file list:")
        print(self.fileDir)
        if parameters is None:
            self.ParaList = np.fromstring(input('Enter parameters (for example: 0, 1, 2, 3):'),sep=',')
        else:
            self.ParaList = np.array(parameters)
        print("The parameters: {}".format(self.ParaList))
        self.isPara = True

    def dumpFitResult(self, picklepath = None):
        if self.isROIfit or self.ROIfitloaded:
            for i in range(self.Nfile):
                tmp = [self.WFList[i].optList, self.WFList[i].sqList, self.WFList[i].ckList, self.imgShift[i]]
                filename = self.fileDir[i].split('.')[0] + '_fit.pkl'
                if picklepath is None:
                    picklepath = self.folderPath + 'pickle/'
                    if not os.path.exists(picklepath):
                        os.makedirs(picklepath)
                with open(picklepath + filename, 'wb') as f:
                    pickle.dump(tmp, f)
                    print("{} file has been dumped!".format(i))

            self.ROIfitsaved = True

    def loadFitResult(self, picklepath = None, refreshChecks = False):
        for i in range(self.Nfile):
            filename = self.fileDir[i].split('.')[0] + '_fit.pkl'
            if picklepath is None:
                picklepath = self.folderPath + 'pickle/'
                if not os.path.exists(picklepath):
                    os.makedirs(picklepath)
            with open(picklepath + filename, 'rb') as f:
                tmpWF = self.WFList[i]
                [tmpWF.optList, tmpWF.sqList, tmpWF.ckList, self.imgShift[i]] = pickle.load(f)
                tmpWF.covList = {}
                for k in tmpWF.ckList.keys():
                    tmpWF.covList[k] = 0
                    if refreshChecks:
                        tmpWF.ckList[k] = 0
                tmpWF.isMultfit = True
                print("{} file has been loaded!".format(i))
                if not self.isAlign:
                    tmpWF.shiftDat(dx = -self.imgShift[i][0], dy = -self.imgShift[i][1])

        self.ROIfitloaded = True



    def test(self):
        fig = plt.figure(num = 1, clear = True, figsize= (6,6*self.Nfile))
        for i in range(self.Nfile):
            ax = fig.add_subplot(self.Nfile, 1, i+1)
            tmpImg = self.WFList[i].originalDat[:,:,3].copy()
            tmpImg = self.normImg(tmpImg)
            img = ax.imshow(tmpImg)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(img, cax=cax)

        plt.show()

    def myFavoriatePlot(self, x = 0, y = 0, spacing = 0.01, withFit = True, fitParas = None):
        fig = plt.figure(num = 1, clear = True, figsize= (15,6))
        
        ## plot the image
        # IMGplot = ax[0].imshow(self.originalDat[:,:,3].copy())
        ax = fig.add_subplot(1, 2, 1)
        Nslice = 3
        IMGplot = ax.imshow(self.imageStack(Nslice))
        ax.add_patch(Rectangle((y - 1, x - 1), 1, 1, fill=None, alpha = 1))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(IMGplot, cax=cax)


        ## plot single ESR
        ax = fig.add_subplot(1, 2, 2)
        offset = 0
        for i in range(self.Nfile):
            tmpWF = self.WFList[i]
            spESR = tmpWF.pointESR(x, y)
            ymax = 1 - spESR.min()
            offset += ymax + spacing
            ax.plot(tmpWF.fVals, spESR + offset, '.', color='k', markersize = 2)

            if withFit:
                if fitParas is None:
                    peaks = tmpWF.singleESRpeakfind(x, y, method = 'pro')

                print("Peaks found by Autofind: "+str(tmpWF.fVals[peaks]))
                print("Peak indices found by Autofind: "+ str(peaks))

                popt, pcov, chisq = tmpWF.singleESRfit(x, y)
                print("the Chi square is {}".format(chisq))
                ax.plot(tmpWF.fVals[peaks], spESR[peaks]+offset, 'x')
            else:
                [popt, pcov, chisq] = fitParas
        
            try:
                for j in np.arange(int(np.floor(len(popt)/3))):
                    params= popt[1+3*j:4+3*j]
                    ax.plot(tmpWF.fVals,popt[0]+mf.lorentzian(tmpWF.fVals,*params)+offset, '-', color = 'r')
            except TypeError:
                print("No good fit was found! Try increase Maxfev or try a better initial guess")

        plt.show()

    def imageStack(self, Nslice = 3, plot = False, imageList = None):
        if imageList is None:
            fileLen = self.Nfile
            imageList = np.arange(self.Nfile)
        else:
            fileLen = len(imageList)

        for i in range(fileLen):
            tmpDat = self.WFList[i].originalDat[:,:,Nslice].copy()
            if i == 0:
                stackImage = self.normImg(tmpDat)
            else:
                stackImage += self.normImg(tmpDat)
        stackImage = stackImage/self.Nfile 
        if plot:
            fig = plt.figure(num = 1, clear = True, figsize= (6,6))
            ax = fig.add_subplot(1, 1, 1)
            ax.set_title("average image stack")
            imgs=ax.imshow(stackImage, interpolation=None)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(imgs, cax=cax)
            plt.show()

        
        return stackImage

    def roi(self, xlow=0, ylow=0, xrange=None, yrange=None, plot = False, lineCut = False):
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
        if lineCut:
            theLine, _ = lineCutGen(lineCut = [[xlow, ylow], [self.xhigh, self.yhigh]], stepSize = 2)
            [self.xr, self.yr] = theLine.transpose()
            self.xyArray = theLine
        else:
            self.xr=np.arange(self.xlow,self.xhigh,1)
            self.yr=np.arange(self.ylow,self.yhigh,1)
            self.xyArray = np.array(list(itertools.product(self.xr, self.yr)))
        self.rroi= [[self.xlow,self.xhigh],[self.ylow,self.yhigh]]
        self.isROI = True
        self.mgSize = 3

        if len(self.xr) > 1 and len(self.yr) > 1 and lineCut == False:
            self.roiShape = 'square'
        elif len(self.xr) == 1 and len(self.yr) == 1:
            self.roiShape = 'point'
        else:
            self.roiShape = 'line'

        if plot:
            image = self.imageStack()
            fig = plt.figure(num = 1, clear = True, figsize= (6,6))
            ax = fig.add_subplot(1, 1, 1)
            ax.set_title("average image stack with roi")
            imgs=ax.imshow(image, interpolation=None)
            ax.add_patch(Rectangle((self.ylow, self.xlow), self.yhigh-self.ylow+self.mgSize, self.xhigh-self.xlow+self.mgSize, fill=None, alpha = 1))
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(imgs, cax=cax)
            plt.show()

    
    def normImg(self, img):
        img = np.array(img)
        maxx = img.max()
        minn = img.min()
        return (img - minn)/(maxx - minn)

    def imageAlign(self, nslice = 3, referN = 0, rr = 20, debug = False):
        if self.isROI:
            beforeAlignImg = self.imageStack(Nslice = nslice)
            tmpDat = self.WFList[referN].originalDat[:,:,nslice].copy()
            tmpDat = self.normImg(tmpDat)
            smallImg = tmpDat[self.rroi[0][0]:self.rroi[0][1], self.rroi[1][0]:self.rroi[1][1]].copy()
            shiftXY = []
            for i in range(self.Nfile):
                print(self.fileDir[i] + " file is processing...")
                tmpWF = self.WFList[i]
                tmp = tmpWF.originalDat[:,:, nslice].copy()
                bigImg = self.normImg(tmp)
                [tmpx, tmpy] = tmpWF.imgCorr(bigImg, smallImg, self.rroi, plot = debug, rr = rr)
                print("WF {}: x is found at {} px; y is found at {} px".format(i , tmpx, tmpy))
                if debug:
                    flag = int(input("Accept?(0/1)"))
                    if flag:
                        rx = tmpx
                        ry = tmpy
                    else:
                        rx = 0
                        ry = 0

                    shiftXY.append([rx, ry])
                else:
                    shiftXY.append([tmpx, tmpy])
            shiftXY = np.array(shiftXY)
            referr = shiftXY[referN].copy()
            for i in range(self.Nfile):
               if shiftXY[i][0]*shiftXY[i][1] != 0:
                    shiftXY[i] -= referr
                    if debug:
                        print(shiftXY[i])
                    tmpWF = self.WFList[i]
                    tmpWF.shiftDat(dx = -shiftXY[i][0], dy = -shiftXY[i][1])
            
            self.imgShift = shiftXY
            aftAlignImg = self.imageStack(Nslice = nslice) 

            fig = plt.figure(num = 1, clear = True, figsize= (15,6))
            ax = fig.add_subplot(1, 2, 1)
            
            bef = ax.imshow(beforeAlignImg)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(bef, cax=cax)

            ax = fig.add_subplot(1, 2, 2)
            aft = ax.imshow(aftAlignImg)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(aft, cax=cax)

            plt.show()

            self.isAlign = True
        else:
            print("This function can only operate after assgining a roi!!")

    
    def manualAlign(self, nslice = 3, referN = 0):
        beforeAlignImg = self.imageStack(Nslice = nslice)
        shiftXY = []
        for i in range(self.Nfile):
            if i != referN:
                print(self.fileDir[i] + " file is processing...")
                imageList = [referN, i]
                tmpWF = self.WFList[i]
                isretry = 1
                while isretry:
                    currentImg = self.imageStack(Nslice = nslice, imageList = imageList, plot=True)
                    [dx, dy] = np.fromstring(input('Enter the x, y shift in pixels (for example: 0, 1)(the imshow has the x and y axis swaped):'),sep=',')
                    tmpWF.shiftDat(dx = -dx, dy = -dy)
                    currentImg = self.imageStack(Nslice = nslice, imageList = imageList, plot=True)
                    asw = int(input("Accept?(0/1)"))
                    if asw:
                        isretry = 0
                        shiftXY.append([dx, dy])
                    else:
                        tmpWF.shiftDat(dx = dx, dy = dy)
            else:
                shiftXY.append([0, 0])
        
        self.imgShift = np.array(shiftXY)
        aftAlignImg = self.imageStack(Nslice = nslice)
        
        fig = plt.figure(num = 1, clear = True, figsize= (15,6))
        ax = fig.add_subplot(1, 2, 1)
        bef = ax.imshow(beforeAlignImg)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(bef, cax=cax)

        ax = fig.add_subplot(1, 2, 2)
        aft = ax.imshow(aftAlignImg)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(aft, cax=cax)

        plt.show()

        self.isAlign = True




    def roiMultiESRfit(self,max_peak = 4, initGuess = None, lineFit = False):
        if self.isROI:
            for i in range(self.Nfile):
                tmpWF = self.WFList[i]
                print("Fitting {}th WFImage...".format(i))
                if initGuess is None:
                    tmpWF.multiESRfit(self.xr, self.yr, max_peak = max_peak, lineFit = lineFit)
                else:
                    tmpWF.multiESRfit(self.xr, self.yr, max_peak = max_peak, initGuess = initGuess[i], lineFit = lineFit)
            
            print("ROI fitting completed!")
            self.isROIfit = True
        else:
            print("This function can only operate after assgining a roi!!")

    def generateroiDEmap(self):
        if self.isROIfit or self.ROIfitloaded:
            self.roiDmap = np.zeros((self.X, self.Y, self.Nfile))
            self.roiEmap = np.zeros((self.X, self.Y, self.Nfile))

            for i in range(self.Nfile):
                tmpWF = self.WFList[i]
                for x in self.xr:
                    for y in self.yr:
                        D, E = tmpWF.DandE(x, y)
                        self.roiDmap[x, y, i] = D
                        self.roiEmap[x, y, i] = E
                print("D & E maps for image {} has been generated!".format(i))
            self.isDEmap = True
            
    def plotroiDEmap(self, refN = None, withroi = False):
        if self.isDEmap:
            if refN is None:
                fig = plt.figure(num = 1, clear = True, figsize= (7*self.Nfile,6*2))
                for i in range(self.Nfile):
                    ax1 = fig.add_subplot(2, self.Nfile, 2*i+1)
                    dmap = ax1.imshow(self.roiDmap[:, :, i], vmax = 3, vmin = 2.8)
                    ax1.set_title("roi D map")
                    divider = make_axes_locatable(ax1)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(dmap, cax=cax)
                    
                    ax2 = fig.add_subplot(2, self.Nfile, 2*i+2)
                    emap = ax2.imshow(self.roiEmap[:, :, i], vmax = 0.2, vmin = 0)
                    ax2.set_title("roi E map")
                    divider = make_axes_locatable(ax2)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(emap, cax=cax)

                plt.show()

            else:
                fig = plt.figure(num = 1, clear = True, figsize= (15,6))
                ax = fig.add_subplot(1, 2, 1)
                dmap = ax.imshow(self.roiDmap[:, :, refN], vmax = 3.3, vmin = 2.8)
                if withroi:
                    ax.add_patch(Rectangle((self.ylow, self.xlow), self.yhigh-self.ylow+self.mgSize, self.xhigh-self.xlow+self.mgSize, fill=None, alpha = 1))
                ax.title.set_text("D map (GHz)")
                divider = make_axes_locatable(ax[0])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(dmap, cax=cax)

                ax = fig.add_subplot(1, 2, 2)
                emap = ax.imshow(self.roiEmap[:, :, refN], vmax = 0.2, vmin = 0)
                if withroi:
                    ax.add_patch(Rectangle((self.ylow, self.xlow), self.yhigh-self.ylow+self.mgSize, self.xhigh-self.xlow+self.mgSize, fill=None, alpha = 1))
                ax.title.set_text("E map (GHz)")
                divider = make_axes_locatable(ax[1])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(emap, cax=cax)

                plt.show()

        else:
            print("This function can only operate after operating ROIDEmaps!!")

    def roiDEvsParas(self, Eymax = 0.2, Dymax = 3):
        if self.isDEmap and self.isPara:
            Emeans = []
            Estds = []
            Dmeans = []
            Dstds = []

            for i in range(self.Nfile):
                tmpEs = self.roiEmap[self.rroi[0][0]:self.rroi[0][1],self.rroi[1][0]:self.rroi[1][1], i]
                tmpEMean = np.mean(tmpEs)
                Emeans.append(tmpEMean)
                tmpEstd = np.std(tmpEs)
                Estds.append(tmpEstd)

                tmpDs = self.roiDmap[self.rroi[0][0]:self.rroi[0][1],self.rroi[1][0]:self.rroi[1][1], i]
                tmpDMean = np.mean(tmpDs)
                Dmeans.append(tmpDMean)
                tmpDstd = np.std(tmpDs)
                Dstds.append(tmpDstd)

            fig = plt.figure(num = 1, clear = True, figsize= (15,6))
            ax = fig.add_subplot(1, 2, 1)
            ax.plot(self.ParaList, Emeans, '-', color = 'r')
            ax.set_ylim(0, Eymax)
            ax.title.set_text("E (GHz)")
            ax.errorbar(self.ParaList, Emeans, yerr = Estds, fmt ='o')

            ax = fig.add_subplot(1, 2, 2)
            ax.plot(self.ParaList, Dmeans, '-', color = 'r')
            ax.set_ylim(0, Dymax)
            ax.title.set_text("D (GHz)")
            ax.errorbar(self.ParaList, Dmeans, yerr = Dstds, fmt ='o')
            plt.show()


        else:
            print("please generate DE map and input the parameter list")

    def lineroiDEvsParas(self, Espacing = 0.1, Dspacing = 0.1):
        if self.isDEmap and self.isPara and self.roiShape == 'line':
            fig = plt.figure(num = 1, clear = True, figsize= (12,12))
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 1)
            
            Eoffset = 0
            Doffset = 0
            if len(self.xr) == 1:
                datax = self.yr
            else:
                datax = self.xr
            for i in range(self.Nfile):
                
                dataE = self.roiEmap[self.rroi[0][0]:self.rroi[0][1],self.rroi[1][0]:self.rroi[1][1], i].flatten()
                dataD = self.roiDmap[self.rroi[0][0]:self.rroi[0][1],self.rroi[1][0]:self.rroi[1][1], i].flatten()
                Espan = dataE.max() - dataE.min()
                Dspan = dataD.max() - dataD.min()

                ax1.plot(datax, dataE + Eoffset, '-.', color = 'k', markersize=2)
                ax2.plot(datax, dataD + Doffset, '-.', color = 'k', markersize=2)
                Eoffset += Espan + Espacing
                Doffset += Dspan + Dspacing
                
            plt.show()

        else:
            print("please generate DE map and input the parameter list also make sure the roi is a line")



#########################################################################
## support functions
#########################################################################
def lineCutGen(lineCut = [[0, 0], [1, 1]], stepSize =1):
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
    return theLine, nnList