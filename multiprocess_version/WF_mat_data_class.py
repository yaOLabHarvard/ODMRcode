# In[1]:
import multipeak_fit as mf
import multiprocessing as mp
import numpy as np
# import matplotlib.pyplot as plt, cm
from matplotlib import pyplot as plt, cm
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.integrate import quad, simpson
from scipy.signal import find_peaks
from scipy.signal import correlate2d

## important: only the pyplot.imshow will swap x and y axis as the output
gamma=2.8025e-3 #GHz/G
nprocs = mp.cpu_count()
## fitting parameters
epslion_y = 1e-3
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
        self.isErrordetect = False
        self.isMask = False
        self.isNpeak = False
        self.binned=False
        self.resumeX = 0
        self.FitCheck = 0
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

    def myFavoriatePlot(self, x = 0, y = 0, maxPeak = 6, withFit = True, fitParas = None):
        if not self.isNorm:
            self.norm()
        if not self.binned:
            self.binning()
        fig, ax = plt.subplots(nrows=1, ncols= 2, figsize= (12,6))
        ## plot the image
        # IMGplot = ax[0].imshow(self.originalDat[:,:,3].copy())
        IMGplot = ax[0].imshow(self.dat[:,:,3])
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
                peaks = self.singleESRpeakfind(x, y, max_peak = maxPeak, method = 'pro')
                ax[1].plot(self.fVals[peaks], spESR[peaks], 'x')
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
                print("peak parameters: {}".format(popt))

            if popt is not None:
                npeak = int(np.floor(len(popt)/3))
            elif fitParas is None:
                npeak = len(peaks)
                popt = mf.generate_pinit(self.fVals[peaks], spESR[peaks])
            else:
                npeak = 0

            for i in np.arange(npeak):
                params = popt[1+3*i:4+3*i]
                ax[1].plot(self.fVals,popt[0]+mf.lorentzian(self.fVals,*params), '-')

        plt.show()
        plt.close()

    def lineCutGen(self, lineCut = [[0, 0], [1, 1]], stepSize =1):
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
            return theLine, nnList

    def waterfallPlot(self, lineCut = [[0, 0], [1, 1]], stepSize =1,  spacing = 0.01, plotTrace = False, plotFit = False, plot = False):
        
            theLine, nnList = self.lineCutGen(lineCut = lineCut, stepSize = stepSize)
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

            if plot:
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
                peakPos, _ = find_peaks(1 - yVals, prominence=(0.0005,0.3))
            # elif min(yVals)<0.997 and min(yVals)>0.996:
            #     print('med-low prom')
            #     peakPos, _ = find_peaks(1 - yVals, prominence=(0.0014,0.3))
            elif min(yVals)<0.996 and min(yVals)>0.994:
                # print('low prom')
                peakPos, _ = find_peaks(1 - yVals, prominence=(0.0008,0.3))
            elif min(yVals)<0.994 and min(yVals)>0.990:
                # print('med prom')
                peakPos, _ = find_peaks(1 - yVals, prominence=(0.0008,0.3))
            elif min(yVals)<0.990 and min(yVals)>0.975:
                # print('high prom')
                peakPos, _ = find_peaks(1 - yVals, prominence=(0.0005,0.3))
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
                if 1 - yVals.min() < epsy:
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
                try:
                    pOpt, pCov= mf.fit_data(self.fVals, yVals, init_params= initParas, fit_function= mf.lor_fit, maxFev=max_repeat)
                except ValueError:
                    self.FitCheck = -2
                    raise ValueError("The singleESRfit fails.. pOpt set to be none")
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

    def multiESRfit(self, xlist, ylist, max_peak = 4, epsy = epslion_y, initGuess = None):
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
        for x in self.multix:
            for y in self.multiy:
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
                        chiSq = None

                self.optList[(x, y)] = pOpt
                self.covList[(x, y)] = pCov
                self.sqList[(x, y)] = chiSq
                self.ckList[(x, y)] = self.FitCheck
            print("{}-th row has completed".format(x))

        self.isMultfit = True
    
    def plotAndCorrect(self, x, y):
        print("current point: x {}; y {}".format(x, y))
        self.myFavoriatePlot(x, y, fitParas=[self.optList[(x, y)], self.covList[(x, y)], self.sqList[(x, y)]])
        if int(input("Looks good?(1/0)")):
            return -1, 0
        else:
            guessFreq = np.fromstring(input('Enter frequency (for example: 2.71, 2.81, 2.91, 3.01):'),sep=',')
            pOpt, pCov, chiSq = self.singleESRfit(x, y,max_peak = len(guessFreq), initGuess=guessFreq)
            self.myFavoriatePlot(x, y, fitParas=[pOpt, pCov, chiSq])
            asw = int(input("Accept?(1/0)"))
        return asw, [pOpt, pCov, chiSq]
    
    def fitErrordetection(self, xrange, yrange, epschi = 5e-5, epsy = epslion_y):
        if self.isMultfit:
            self.errorIndex = []
            for x in xrange:
                for y in yrange:
                    if self.sqList[(x, y)] is None:
                        self.sqList[(x, y)] = 1
                    if self.sqList[(x, y)] > epschi and min(self.dat[x, y]) < 1 - epsy and self.ckList[(x, y)] < 3:
                        self.errorIndex.append([x,y])
            
            self.isErrordetect = True
        else:
            print("Please do multiesr first!")

    def multiESRfitManualCorrection(self, isResume = False):
        if self.isErrordetect:
            if isResume:
                currentIndex = self.resumeIndex
            else:
                currentIndex = 0
            errorList = self.errorIndex[currentIndex:]
            for [x, y] in errorList:
                try:
                    isRetry = 1
                    while isRetry:
                        asw1, fitList = self.plotAndCorrect(x, y)
                        if asw1 > 0:
                            self.optList[(x, y)] = fitList[0]
                            self.covList[(x, y)] = fitList[1]
                            self.sqList[(x, y)] = fitList[2]
                            self.ckList[(x, y)] = 3
                            isRetry = 0
                        elif asw1 == 0:
                            asw2 = int(input("Want to retry?(1/0)"))
                            if asw2 == 0:
                                isRetry = 0
                        else:
                            self.ckList[(x, y)] = 3
                            break

                except(ValueError, RuntimeError, IndexError) as e:
                    print(e)
                    self.resumeIndex = currentIndex
                    print("Force to stop!")
                    return 0
                
                currentIndex += 1
            self.resumeIndex = 0
        print("manual error correction finished!")

    
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
        if dx == 0 and dy == 0:
            print("no shifting!")
            return
        else:
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


    def DEmap(self, plot = False):
        if self.isMultfit:
            self.Dmap = np.zeros((self.X, self.Y))
            self.Emap = np.zeros((self.X, self.Y))
            self.Dmax = 0
            self.Dmin = 1e6
            self.Emax = 0
            self.Emin = 1e6
            for x in self.multix:
                for y in self.multiy:
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
                fig, ax = plt.subplots(nrows=1, ncols= 2, figsize= (6*2,6))
                img1 = ax[0].imshow(self.Dmap, vmax = self.Dmax, vmin = self.Dmin)
                divider = make_axes_locatable(ax[0])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(img1, cax=cax)


                img2 = ax[1].imshow(self.Emap, vmax = self.Emax, vmin = self.Emin)
                divider = make_axes_locatable(ax[1])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(img2, cax=cax)

                plt.show()
                plt.close()

            return self.Dmap, self.Emap
        else:
            print("Need to operate multiesrfit first!!")

    def DElineplot(self, lineCut = [[0, 0], [1, 1]], stepSize =1, plotTrace = False, plotD = False, plotE = False):
        if self.isMultfit:
            theLine, nnList = self.lineCutGen(lineCut = lineCut, stepSize = stepSize)
            if plotTrace:
                fig, ax = plt.subplots(nrows=1, ncols= 1, figsize= (6,6))
                IMGplot = ax.imshow(self.dat[:,:,3].copy())
                ## switch x and y for imshow

                ax.plot(theLine[:, 1], theLine[:, 0], '.', color='red', markersize=15)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(IMGplot, cax=cax)
                plt.show()
                plt.close()
                ## generate plots
            fig, ax = plt.subplots(nrows=1, ncols= 1, figsize= (6,12))
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
            plt.close()


    def DEwidthplot(self, lineCut = [[0, 0], [1, 1]], stepSize =1, plotTrace = False):
        if self.isMultfit:
            theLine, nnList = self.lineCutGen(lineCut = lineCut, stepSize = stepSize)
            if plotTrace:
                fig, ax = plt.subplots(nrows=1, ncols= 1, figsize= (6,6))
                IMGplot = ax.imshow(self.dat[:,:,3].copy())
                ## switch x and y for imshow

                ax.plot(theLine[:, 1], theLine[:, 0], '.', color='red', markersize=15)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(IMGplot, cax=cax)
                plt.show()
                plt.close()
                ## generate plots
            fig, ax = plt.subplots(nrows=1, ncols= 1, figsize= (6,12))
            widthList = []
            for i in range(len(nnList)):
                width = self.theWidth(theLine[i][0], theLine[i][1])
               
                widthList.append(width)

            ax.plot(nnList, widthList, '.', color = 'r', markersize=2, label = "Width (GHz)")

            plt.show()
            plt.close()


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

