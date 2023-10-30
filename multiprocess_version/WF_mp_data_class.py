from WF_mat_data_class import WFimage
import multipeak_fit as mf
import multiprocessing as mp
import numpy as np
# import matplotlib.pyplot as plt, cm
from matplotlib import pyplot as plt, cm
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import time

## important: only the pyplot.imshow will swap x and y axis as the output
# import scipy as sp
import os
gamma=2.8025e-3 #GHz/G
nprocs = mp.cpu_count()
## fitting parameters
epslion_y = 1e-3
peak_width = 0.015 ## GHz
max_repeat = 1000

##My comment

def WFmatrixESRfit(wf, xlist, ylist, max_peak = 4, epsy = epslion_y, initGuess = None):
        print("file name: "+wf.WFimagename)
        if not wf.isNorm:
            wf.norm()
        if not wf.binned:
            wf.binning()
        if not wf.isMultfit:
            wf.optList = {}
            wf.covList = {}
            wf.sqList = {}
            wf.ckList = {}
        wf.multix = xlist
        wf.multiy = ylist
        guessFreqs = initGuess
        for x in wf.multix:
            for y in wf.multiy:
                if initGuess is not None:
                    try:
                        pOpt, pCov, chiSq = wf.singleESRfit(x, y,max_peak = max_peak, epsy = epsy, initGuess=guessFreqs)
                        if pOpt is not None:
                            guessFreqs = pOpt[0::3][1:]
                        else:
                            guessFreqs = initGuess
                    except ValueError:
                        print("manual guess fails.. trying init guess")
                        try:
                            pOpt, pCov, chiSq = wf.singleESRfit(x, y,max_peak = max_peak, epsy = epsy, initGuess=initGuess)
                        except ValueError:
                            print("init guess fails.. trying auto guess")
                            try:
                                pOpt, pCov, chiSq = wf.singleESRfit(x, y,max_peak = max_peak, epsy = epsy)
                            except ValueError:
                                print("auto guess fails.. Popt set to be none")
                                

                    
                        
                else:
                    try:
                        pOpt, pCov, chiSq = wf.singleESRfit(x, y,epsy = epsy, max_peak = max_peak)
                    except ValueError:
                        print("auto guess fails.. Popt set to be none")
                        pOpt = None
                        pCov = None
                        chiSq = None

                wf.optList[(x, y)] = pOpt
                wf.covList[(x, y)] = pCov
                wf.sqList[(x, y)] = chiSq
                wf.ckList[(x, y)] = wf.FitCheck
            print("{}-th row has completed".format(x))

        wf.isMultfit = True
    
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
        self.ROIfitsaved = False
        self.isDEmap = False
        self.isPara = False
        self.isMp = True

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
        if self.isROIfit:
            for i in range(self.Nfile):
                tmp = [self.WFList[i].optList, self.WFList[i].sqList, self.imgShift[i]]
                filename = self.fileDir[i].split('.')[0] + '_fit.pkl'
                if picklepath is None:
                    picklepath = self.folderPath + 'pickle/'
                    if not os.path.exists(picklepath):
                        os.makedirs(picklepath)
                with open(picklepath + filename, 'wb') as f:
                    pickle.dump(tmp, f)
                    print("{} file has been dumped!".format(i))

            self.ROIfitsaved = True

    def loadFitResult(self, picklepath = None):
        for i in range(self.Nfile):
            filename = self.fileDir[i].split('.')[0] + '_fit.pkl'
            if picklepath is None:
                picklepath = self.folderPath + 'pickle/'
                if not os.path.exists(picklepath):
                    os.makedirs(picklepath)
            with open(picklepath + filename, 'rb') as f:
                tmpWF = self.WFList[i]
                [tmpWF.optList, tmpWF.sqList, self.imgShift[i]] = pickle.load(f)
                tmpWF.isMultfit = True
                print("{} file has been loaded!".format(i))
                if not self.isAlign:
                    tmpWF.shiftDat(dx = -self.imgShift[i][0], dy = -self.imgShift[i][1])

        self.ROIfitloaded = True



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

    def imageStack(self, Nslice = 3, plot = False, imageList = None):
        if imageList is None:
            fileLen = self.Nfile
            imageList = np.arange(self.Nfile)
        else:
            fileLen = len(imageList)

        for i in range(fileLen):
            tmpDat = self.WFList[imageList[i]].dat.copy()
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
        self.mgSize = 3

        if len(self.xr) > 1 and len(self.yr) > 1:
            self.roiShape = 'square'
        elif len(self.xr) == 1 and len(self.yr) == 1:
            self.roiShape = 'point'
        else:
            self.roiShape = 'line'

        if plot:
            image = self.imageStack()
            fig,ax=plt.subplots(1,1)
            ax.set_title("average image stack with roi")
            imgs=ax.imshow(image, interpolation=None)
            ax.add_patch(Rectangle((self.ylow, self.xlow), self.yhigh-self.ylow+self.mgSize, self.xhigh-self.xlow+self.mgSize, fill=None, alpha = 1))
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

    def imageAlign(self, nslice = 3, referN = 0, rr = 20, debug = False):
        if self.isROI:
            beforeAlignImg = self.imageStack(Nslice = nslice)
            tmpDat = self.WFList[referN].dat[:,:,nslice].copy()
            tmpDat = self.normImg(tmpDat)
            smallImg = tmpDat[self.rroi[0][0]:self.rroi[0][1], self.rroi[1][0]:self.rroi[1][1]].copy()
            shiftXY = []
            for i in range(self.Nfile):
                print(self.fileDir[i] + " file is processing...")
                tmpWF = self.WFList[i]
                tmp = tmpWF.dat[:,:, nslice].copy()
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

    def mpMultiESRfit(self,max_peak = 4, initGuess = None):
        if self.isMp:
            if self.isROI:
                print("fitting start! Current time: {}".format(time.time()))
                if self.Nfile < nprocs:
                    myPool = mp.Pool(processes=self.Nfile)
                    xr = self.xr
                    yr = self.yr
                    for i in range(self.Nfile):
                        tmpWF = self.WFList[i]
                        myPool.apply_async(WFmatrixESRfit, args=(tmpWF, xr, yr, max_peak, epslion_y, initGuess))

                    print("fitting complete! Current time: {}".format(time.time()))
                    self.isROIfit = True
            else:
                print("This function can only operate after assgining a roi!!")
        else:
            print("multiprocess is not allowed!")
                        

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
                fig, ax = plt.subplots(nrows=2, ncols= self.Nfile, figsize= (7*self.Nfile,6*2))
                for i in range(self.Nfile):
                    dmap = ax[0, i].imshow(self.roiDmap[:, :, i], vmax = 3, vmin = 2.8)
                    ax[0, i].set_title("roi D map")
                    divider = make_axes_locatable(ax[0, i])
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(dmap, cax=cax)

                    emap = ax[1, i].imshow(self.roiEmap[:, :, i], vmax = 0.2, vmin = 0)
                    ax[1, i].set_title("roi E map")
                    divider = make_axes_locatable(ax[1, i])
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(emap, cax=cax)

                plt.show()
                plt.close()
            else:
                fig, ax = plt.subplots(nrows=1, ncols= 2, figsize= (15,6))
                dmap = ax[0].imshow(self.roiDmap[:, :, refN], vmax = 3, vmin = 2.8)
                if withroi:
                    ax[0].add_patch(Rectangle((self.ylow, self.xlow), self.yhigh-self.ylow+self.mgSize, self.xhigh-self.xlow+self.mgSize, fill=None, alpha = 1))
                ax[0].title.set_text("D map (GHz)")
                divider = make_axes_locatable(ax[0])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(dmap, cax=cax)


                emap = ax[1].imshow(self.roiEmap[:, :, refN], vmax = 0.2, vmin = 0)
                if withroi:
                    ax[1].add_patch(Rectangle((self.ylow, self.xlow), self.yhigh-self.ylow+self.mgSize, self.xhigh-self.xlow+self.mgSize, fill=None, alpha = 1))
                ax[1].title.set_text("E map (GHz)")
                divider = make_axes_locatable(ax[1])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(emap, cax=cax)

                plt.show()
                plt.close()
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

            fig, ax = plt.subplots(nrows=1, ncols= 2, figsize= (15,6))
            ax[0].plot(self.ParaList, Emeans, '-', color = 'r')
            ax[0].set_ylim(0, Eymax)
            ax[0].title.set_text("E (GHz)")
            ax[0].errorbar(self.ParaList, Emeans, yerr = Estds, fmt ='o')

            ax[1].plot(self.ParaList, Dmeans, '-', color = 'r')
            ax[1].set_ylim(0, Dymax)
            ax[1].title.set_text("D (GHz)")
            ax[1].errorbar(self.ParaList, Dmeans, yerr = Dstds, fmt ='o')
            plt.show()
            plt.close()

        else:
            print("please generate DE map and input the parameter list")

    def lineroiDEvsParas(self, Espacing = 0.1, Dspacing = 0.1):
        if self.isDEmap and self.isPara and self.roiShape == 'line':
            fig, ax = plt.subplots(nrows=1, ncols= 2, figsize= (12,12))
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

                ax[0].plot(datax, dataE + Eoffset, '-.', color = 'k', markersize=2)
                ax[1].plot(datax, dataD + Doffset, '-.', color = 'k', markersize=2)
                Eoffset += Espan + Espacing
                Doffset += Dspan + Dspacing
                
            plt.show()
            plt.close()
        else:
            print("please generate DE map and input the parameter list also make sure the roi is a line")
