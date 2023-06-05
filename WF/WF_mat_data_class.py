# In[1]:
import multipeak_fit as mf
import numpy as np
# import matplotlib.pyplot as plt, cm
from matplotlib import pyplot as plt, cm
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.integrate import quad
from scipy.signal import find_peaks
from scipy.signal import correlate2d
import scipy.optimize as opt
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

    def binning(self):
        binsize=4
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
        self.norm()
        self.binned=True

    def norm(self):
        self.dat = mf.normalize_widefield(self.dat, numpoints= np.floor(self.npoints/2).astype(int))
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
        if not self.isNorm:
            self.norm()
        if not self.binned:
            self.binning()
        fig, ax = plt.subplots(nrows=1, ncols= 2, figsize= (12,6))
        ## plot the image
        # IMGplot = ax[0].imshow(self.originalDat[:,:,3].copy())
        IMGplot = ax[0].imshow(self.binOrigDat[:,:,3].copy())
        ax[0].add_patch(Rectangle((x - 1, y -1), 1, 1, fill=None, alpha = 1))
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

    def singleESRpeakfind(self, x = 0, y = 0, max_peak = 4, method = 'user'):
        if not self.isNorm:
            self.norm()
        if not self.binned:
            self.binning()
        yVals = self.pointESR(x, y)
        ymax = max(1 - yVals)
        # ymin = min(1 - yVals)
        if method == 'user':
            #for LuH-N
            peakPos, _ = find_peaks(1 - yVals, distance = 15,  height=(ymax/5, ymax), width = 2)

            #for LT
            # peakPos, _ = find_peaks(1 - yVals, distance = 59,  height=(ymax/4.05, ymax), width = 3)
            #for RT
            # peakPos, _ = find_peaks(1 - yVals, distance = 40,  height=(ymax/4.05, ymax), width = 4)
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
                if autofind:
                    self.peakPos = self.singleESRpeakfind(x, y)
                    # print(self.peakPos)
                else:
                    self.peakPos = np.fromstring(input('Enter frequency (for example: 2.71, 2.81, 2.91, 3.01):'),sep=',')
                initParas = mf.generate_pinit(self.fVals[self.peakPos], yVals[self.peakPos])

                pOpt, pCov= mf.fit_data(self.fVals, yVals, init_params= initParas, fit_function= mf.lor_fit, maxFev=900)
                # print(pOpt)
                self.pOptPrint=pOpt
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
                    # else:
                    #     self.myFavoriatePlot(x, y)
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

# %% EvsI and makeContourPlot
def EvsI(x, Delta, alpha):
    return np.sqrt(Delta**2+(alpha*x)**2)
def linFit(x, alpha, c):
    return alpha*x+c
def makeContourplot(rawdat):
    (xx,yy,zz) = rawdat.shape
    xlist = np.arange(0, xx, 1)
    ylist = np.arange(0, yy, 1)
    X, Y = np.meshgrid(xlist, ylist)
    Z = np.zeros((xx, yy), dtype=float)
    for x in range(xx):
        for y in range(yy):
            Z[x, y] = min(rawdat[x, y])
    return X, Y, Z

# %% class MultiFileWF
class MultiFileWF:
    def __init__(self,folderpath, Ilist, temp):
        self.temp=temp
        self.Ilist=np.array(Ilist, dtype=float)
        filenamearr=os.listdir(folderpath)
        filenamearr.sort
        self.filenamearr=filenamearr
        print(filenamearr)
        self.nfiles=int(len(self.Ilist)) #Number of files to process
        self.WFList=[]
        for filename in filenamearr[:len(self.Ilist)]:
            self.WFList.append(WFimage(folderpath + filename))
            print("mat file is loaded successfully!")
        print("WFList Size: " + str(len(self.WFList)))
        print("all mat files have been loaded successfully!")
        
        self.datList=[]
        self.fValsList=[]
        for WF in self.WFList:
            WF.norm()
            WF.binning()
            self.datList.append(WF.dat)
            self.fValsList.append(WF.fVals)

        self.um=False #If false, it doesn't convert pixels to micron.
        self.allMultFit=False
        self.refine=False
        self.poiFit=False
        self.EvsIfit=False
        self.refineCount=0
        # self.binned=False

    def test(self):
        for WF in self.WFList:
            testimg = WF.originalDat[:,:,3].copy()
            fig, ax = plt.subplots(nrows=1, ncols= 1, figsize= (6,6))
            ax.imshow(testimg)
            plt.show()

    def roi(self):
        self.xlow=0
        self.ylow=0
        totalxr=len(self.datList[:][0][0])
        totalyr=len(self.datList[0][:][0])

        self.xr=np.arange(self.xlow,totalxr,1)
        self.yr=np.arange(self.ylow,totalyr,1)


    def MF_POIFit(self, poiX, poiY): #Single Point
        self.poiX=poiX
        self.poiY=poiY
        self.POIoptListList = []
        self.POIfVals=[]
        self.POIyVals=[]
        self.POIfitcurve=[]
        self.POIparams=[]
        self.peakPosList = []
        for i in np.arange(len(self.WFList)):
            # print(i)
            WF=self.WFList[i]
            WF.norm()
            peakPos = WF.singleESRpeakfind(poiX, poiY)
            self.peakPosList.append(peakPos)
            pOpt, pCov, chiSq = WF.singleESRfit(poiX,poiY)
            print("chiSq is: "+ str(chiSq))
            # WF.myFavoriatePlot(poiX, poiY)
            # print("pOpt:", pOpt)
            if pOpt is not None:
                self.POIoptListList.append(pOpt)
                self.POIfVals.append(WF.fVals)
                self.POIyVals.append(WF.dat[poiX, poiY,:])
                self.POIfitcurve.append(mf.lor_fit(WF.fVals, *pOpt))
                # self.POIparams.append(np.zeros((int(np.floor(len(pOpt)/3)), 3)))
                # for j in np.arange(int(np.floor(len(pOpt)/3))):
                #     self.POIparams[i][j,:]= pOpt[1+3*j:4+3*j]
                # self.poiDs.append((pOpt[-4]+pOpt[-1])/2)
                # self.poiEs.append(abs(pOpt[-4]-pOpt[-1])/2) #Equivalent to singleESREs
            else:
                print("Unable to fit this point. Choose another POI!")
                self.POIoptListList.append(None)
                self.POIfVals.append(None)
                self.POIyVals.append(None)
        self.poiFit=True

    def MF_multiESRFit(self): #Map
        self.optListList = []
        self.NpeakList=[]
        for WF in self.WFList:
            WF.multiESRfit(self.xr, self.yr, backupOption=np.array([3.85, 4.25]))
            Npeak=WF.multiNpeakplot()
            self.NpeakList.append(Npeak)
            self.optListList.append(WF.optList)
        self.allMultFitandN=True

# %%
def poiDandE(MFWF, poiX, poiY): #Single Point
    if MFWF.poiFit==True:
        poiDs=[]
        poiEs=[]
        for i in np.arange(len(MFWF.WFList)):
            if MFWF.optListList[i][poiX, poiY] is not None:
                pOpt=MFWF.optListList[i][poiX, poiY]
                poiDs.append((pOpt[-4]+pOpt[-1])/2)
                poiEs.append(abs(pOpt[-4]-pOpt[-1])/2) #Equivalent to singleESREs
        # print("This is self.poiEs:" + str(self.poiEs))
            else: print("BAD POI! Didn't append POI D and POI E for file %i."%(i))
    else:
        print("Please run MF_POIFit() first!")
    return poiDs, poiEs

def poiProcessDandE(MFWF, poiX, poiY):
    if MFWF.allMultFitandN==True and MFWF.poiFit==True:
        Dmin=0
        Dmax=8
        Emin=0
        Emax=3
        # Dmin=3.5
        # Dmax=4.3
        # Emin=0.0
        # Emax=0.4
        poiDs, poiEs=poiDandE(MFWF, poiX, poiY)

        if MFWF.temp=="lt":
            EpOpt,EpCov= opt.curve_fit(linFit, MFWF.Ilist[:MFWF.nfiles], poiEs)
            poialpha=EpOpt[0].astype(float)
            poiDelta=EpOpt[1].astype(float)
            # poiBval=MFWF.poialpha/gamma #G/A
            poifittedE=linFit(MFWF.Ilist[:MFWF.nfiles], *EpOpt)
            return poiEs, poialpha, poiDelta, poifittedE
        else:
            EpOpt,EpCov= opt.curve_fit(EvsI, MFWF.Ilist[:MFWF.nfiles], poiEs)
            print("EpOpt: " + str(EpOpt))
            # else:
                # print('Bad POI! Choose another point!')
            poiDelta=EpOpt[0].astype(float)
            poialpha=EpOpt[1].astype(float)
            # poiBval=self.poialpha/gamma #G/A
            poifittedE=EvsI(MFWF.Ilist[:MFWF.nfiles], poiDelta, poialpha)
            return poiEs, poialpha, poiDelta, poifittedE
    else:
        print("Please run MF_multiESRFit() and MF_POIFit() first!")

def DandE(self): #Map
    if self.allMultFitandN==True:
        self.Dlist=[]
        self.Elist=[]
        for WF in self.WFList:
            if WF.DEmap!=None:
                Dmap, Emap= WF.DEmap()
            self.Dlist.append(Dmap)
            self.Elist.append(Emap)
    else:
        print("Please run MF_multiESRFit() first!")

def processDandE(self, poiX, poiY, alphalow, alphahigh):
    if self.allMultFitandN==True and self.poiFit==True:
        Dmin=0
        Dmax=8
        Emin=0
        Emax=3
        # Dmin=3.5
        # Dmax=4.3
        # Emin=0.0
        # Emax=0.4

        #Find the bad points
        self.badptmap=np.zeros((len(self.xr),len(self.yr)))
        for i in np.arange(self.nfiles):
            for j in self.xr:
                for k in self.yr:
                    if self.Dlist[i][j-self.xlow,k-self.ylow]<Dmin or self.Dlist[i][j-self.xlow,k-self.ylow]>Dmax or self.Elist[i][j-self.xlow,k-self.ylow]<Emin or self.Elist[i][j-self.xlow,k-self.ylow]>Emax:
                        self.badptmap[j-self.xlow,k-self.ylow]=1
        fig,ax=plt.subplots(1,1)
        ax.set_title("Bad Point Map (1 is Bad)")
        BPM=ax.imshow(self.badptmap, interpolation=None)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(BPM, cax=cax)


        ##Fit the Data for POI! This will be used as the guess parameters.
        # # if np.shape(self.poiEs)!=():
        # print("Ilist: "+ str(self.Ilist[:self.nfiles]))
        # print("poiEs: "+ str(self.poiEs))
        poiEs, poialpha, poiDelta, poifittedE=poiProcessDandE(MFWF, poiX, poiY)
        
        if self.temp=="lt":
            EpOpt,EpCov= opt.curve_fit(linFit, self.Ilist[:self.nfiles], poiEs)
            self.poialpha=EpOpt[0].astype(float)
            self.poiBval=self.poialpha/gamma #G/A
        else:
            EpOpt,EpCov= opt.curve_fit(EvsI, self.Ilist[:self.nfiles], poiEs)
            print("EpOpt: " + str(EpOpt))
            # else:
                # print('Bad POI! Choose another point!')
            self.poiDelta=EpOpt[0].astype(float)
            self.poialpha=EpOpt[1].astype(float)
            self.poiBval=self.poialpha/gamma #G/A

        if self.temp=="lt":
            self.poifittedE=linFit(self.Ilist[:self.nfiles], *EpOpt)
        else:
            self.poifittedE=EvsI(self.Ilist[:self.nfiles], self.poiDelta, self.poialpha)

        # if onept==False:
        self.ESpOptmap=np.zeros((len(self.xr),len(self.yr),2))
        self.Deltamap=np.zeros((len(self.xr),len(self.yr)))
        self.alphamap=np.ones((len(self.xr),len(self.yr)))
        self.Bmap=np.zeros((len(self.xr),len(self.yr)))
        for i in self.xr:
            for j in self.yr:
                if self.badptmap[i-self.xlow,j-self.ylow]!=1:
                    singlePointE=[]
                    for k in np.arange(len(self.Elist)):
                        singlePointE.append(self.Elist[k][i-self.xlow,j-self.ylow])
                    if self.temp=="lt":
                        try:
                            ESpOpt,pCov= opt.curve_fit(linFit, self.Ilist[:self.nfiles], singlePointE)
                        except opt.OptimizeWarning:
                            print('Pixel:[%i,%i] OptimizeWarning'%(i,j))
                            continue
                        except RuntimeError:
                            print('Pixel:[%i,%i] encountered RuntimeError'%(i,j))
                            continue
                        except ValueError:
                            print('Pixel:[%i,%i] encountered ValueError for EvsI fit'%(i,j))
                            continue
                        self.ESpOptmap[i-self.xlow,j-self.ylow,:]=ESpOpt
                        self.Deltamap[i-self.xlow,j-self.ylow]=ESpOpt[1] #Actually cmap
                        self.alphamap[i-self.xlow,j-self.ylow]=ESpOpt[0]
                        # self.Bmap[i-self.xlow,j-self.ylow]=ESpOpt[1]/gamma #G/A

                    else:
                        try:
                            ESpOpt,pCov= opt.curve_fit(EvsI, self.Ilist[:self.nfiles], singlePointE, p0=EpOpt)
                        except opt.OptimizeWarning:
                            print('Pixel:[%i,%i] OptimizeWarning'%(i,j))
                            continue
                        except RuntimeError:
                            print('Pixel:[%i,%i] encountered RuntimeError'%(i,j))
                            continue
                        except ValueError:
                            print('Pixel:[%i,%i] encountered ValueError for EvsI fit'%(i,j))
                            continue
                        self.ESpOptmap[i-self.xlow,j-self.ylow,:]=ESpOpt
                        self.Deltamap[i-self.xlow,j-self.ylow]=ESpOpt[0]
                        self.alphamap[i-self.xlow,j-self.ylow]=ESpOpt[1]
                        self.Bmap[i-self.xlow,j-self.ylow]=ESpOpt[1]/gamma #G/A
                else:
                    self.ESpOptmap[i-self.xlow,j-self.ylow,:]=[0,0]
                    self.Deltamap[i-self.xlow,j-self.ylow]=1
                    self.alphamap[i-self.xlow,j-self.ylow]=1
                    self.Bmap[i-self.xlow,j-self.ylow]=0
        
        plotthree(self, alphalow, alphahigh)
        manual=input("Would you like to manually fix points? y/n")
        if manual=='y':
            manualEntry(self)

        self.EvsIfit=True
    else:
        print("Please run MF_multiESRFit() and MF_POIFit() first!")

def avgDmap(self):
    if self.allMultFitandN==True and self.poiFit==True:
        self.avgD=np.zeros((len(self.xr),len(self.yr)))
        for j in self.xr:
            for k in self.yr:
                D=[]
                for i in np.arange(self.nfiles):
                    D.append(self.Dlist[i][j,k])
                if self.badptmap[j,k]!=1:
                    self.avgD[j,k]=np.mean(D)
                else:
                    self.avgD[j,k]=0

def manualEntry(self):
    manESpOptmap=np.zeros((len(self.xr),len(self.yr),2))
    choice=input("Would you like to (a) Choose which points to fix, or (b) Run through all failed points?")
    if choice=='a':
        ex=0
        while ex==0:
            loc=input("Input Point")
            try:
                locarr = loc.split(",")
                xloc=int(locarr[0])
                yloc=int(locarr[1])
            except ValueError:
                    print('Please Re-Input the Value')
                    continue
            except IndexError:
                    print('Please Re-Input the Value')
                    continue
            fig,ax=plt.subplots(1,3, figsize=(15, 3))
            k=0
            for WF in self.WFList:
                ax[k].plot(WF.fVals, WF.dat[xloc,yloc,:])
                ax[k].set_xticks(np.arange(round(min(WF.fVals),1), round(max(WF.fVals)+0.1,1), 0.1))
                k=k+1
            plt.show()

            inputElist=[]
            k=0
            for WF in self.WFList:
                try:
                    f1 = float(input("Enter Manual Peak Position 1 for File %i: "%(k)))
                    f2 = float(input("Enter Manual Peak Position 2 for File %i: "%(k)))
                    inputElist.append(abs(f1-f2)/2)
                    k=k+1
                except ValueError:
                    print('Please Re-Input the Value')
                    continue

            if self.temp=="lt":
                try:
                    ESpOpt,pCov= opt.curve_fit(linFit, self.Ilist[:self.nfiles], inputElist)
                    manESpOptmap[xloc, yloc]=ESpOpt
                    # self.Deltamap[xloc, yloc]=ESpOpt[1] #Actually cmap
                    # self.alphamap[xloc, yloc]=ESpOpt[0]
                except ValueError:
                    print('Something went wrong...')
                    continue
            else:
                try:
                    ESpOpt,pCov= opt.curve_fit(EvsI, self.Ilist[:self.nfiles], inputElist, p0=EpOpt)
                    manESpOptmap[xloc, yloc]=ESpOpt
                    # self.alphamap[xloc, yloc]=ESpOpt[1]
                    # self.Deltamap[xloc, yloc]=ESpOpt[0]
                except ValueError:
                    print('Something went wrong...')
                    continue
            
            self.alphamap[xloc, yloc]=manESpOptmap[xloc, yloc][0]
            self.Deltamap[xloc, yloc]=manESpOptmap[xloc, yloc][1]

            plotthree(self, alphalow, alphahigh)
            another=input("Fit another point? y/n")
            if another=='n':
                ex=1
    if choice=='b':
        for i in self.xr:
            for j in self.yr:
                if self.alphamap[i-self.xlow,j-self.ylow]<alphalow or self.alphamap[i-self.xlow,j-self.ylow]>alphahigh:
                    fig,ax=plt.subplots(1,3, figsize=(15, 3))
                    k=0
                    for WF in self.WFList:
                        ax[k].plot(WF.fVals, WF.dat[i-self.xlow,j-self.ylow,:])
                        ax[k].set_xticks(np.arange(round(min(WF.fVals),1), round(max(WF.fVals)+0.1,1), 0.1))
                        k=k+1
                    plt.show()
                    yn=input("Fix Point (%i,%i)? y/n" %(i-self.xlow,j-self.ylow))
                    if yn=='y':
                        inputElist=[]
                        k=0
                        for WF in self.WFList:
                            try:
                                f1 = float(input("Enter Manual Peak Position 1 for File %i: "%(k)))
                                f2 = float(input("Enter Manual Peak Position 2 for File %i: "%(k)))
                                inputElist.append(abs(f1-f2)/2)
                                k=k+1
                            except ValueError:
                                print('Please Re-Input the Value')
                                continue

                        if self.temp=="lt":
                            ESpOpt,pCov= opt.curve_fit(linFit, self.Ilist[:self.nfiles], inputElist)
                            self.Deltamap[i-self.xlow,j-self.ylow]=ESpOpt[1] #Actually cmap
                            self.alphamap[i-self.xlow,j-self.ylow]=ESpOpt[0]
                        else:
                            ESpOpt,pCov= opt.curve_fit(EvsI, self.Ilist[:self.nfiles], inputElist, p0=EpOpt)
                            self.alphamap[i-self.xlow,j-self.ylow]=ESpOpt[1]
                            self.Deltamap[i-self.xlow,j-self.ylow]=ESpOpt[0]
                        plotthree(self, alphalow, alphahigh)
                    elif yn=='e':
                        break
                    # print(val)


# %%

def pxToUm(MFWF):
    # UmPerPx=0.175 #No bins
    # UmPerPx=20/38 #3x3 bin
    UmPerPx=20/28 #4x4 bin
    fulltickpts=np.linspace(0, MFWF.WFList[0].X, 5)
    # label_list = round(np.multiply(fulltickpts, UmPerPx), 3)
    label_list = []
    for i in fulltickpts:
        label_list.append(round(np.multiply(i, UmPerPx), 3))
    return fulltickpts, label_list
    
def plotone(MFWF):
    cmap = cm.get_cmap('rainbow').copy()
    cmap.set_over('lightgray')
    cmap.set_under('lightgray')
    cmap.set_bad('white')

    fulltickpts, label_list=pxToUm(MFWF)
    # label_list=np.round(label_list,3)
    # label_list=[round(x) for x in label_listraw]
    roitickpts=fulltickpts

    #Masking
    for k in np.arange(MFWF.nfiles):
        for i in MFWF.xr:
            for j in MFWF.yr:
                if MFWF.badptmap[i,j]==1:
                        MFWF.Dlist[k][i,j]=np.nan
                        MFWF.Elist[k][i,j]=np.nan

    Dmin=3.5
    Dmax=4.3
    Emin=0.0
    Emax=0.4
    fig, ax = plt.subplots(MFWF.nfiles,5, figsize=(20, 3.5*MFWF.nfiles+1))
    fig.tight_layout(pad=2.5)
    params=[]
    for i in np.arange(MFWF.nfiles):
        ccmap = cm.get_cmap('viridis').copy()
        ax[i,0].imshow(MFWF.datList[i][:,:,3], vmin=0.9994, vmax=1.001,cmap=ccmap)
        ax[i,0].set_xticks(fulltickpts)
        ax[i,0].set_xticklabels(label_list)
        ax[i,0].set_yticks(fulltickpts)
        ax[i,0].set_yticklabels(label_list)
        ax[i,0].set_xlabel("um (Estimated)")
        ax[i,0].set_title(MFWF.filenamearr[i][10:-4])
        # rect = Rectangle((ylow,xlow),yrange,xrange,linewidth=1,edgecolor='r',facecolor='none')
        # ax[i,0].add_patch(rect)
        ax[i,0].plot(MFWF.poiY, MFWF.poiX, 'ro')

        ax[i,1].set_title("Single Point ESR")
        ax[i,1].set_xlabel("Frequency (GHz)")
        ax[i,1].set_ylabel("Contrast")
        singleESRplot = ax[i,1].plot(MFWF.fValsList[i], MFWF.POIyVals[i])
        # ax[i,1].plot(MFWF.fValsList[i], MFWF.POIyVals[i])
        ax[i,1].plot(MFWF.fValsList[i], mf.lor_fit(MFWF.fValsList[i], *MFWF.POIoptListList[i]), '--', linewidth=2.5)
        params.append(np.zeros((int(np.floor(len(MFWF.POIoptListList[i])/3)), 3)))
        for j in np.arange(int(np.floor(len(MFWF.POIoptListList[i])/3))):
            params[i][j,:]= MFWF.POIoptListList[i][1+3*j:4+3*j]

        for j in np.arange(int(np.floor(len(MFWF.POIoptListList[i])/3))):
            ESRtext=str(round(params[i][j,2],3))
            ax[i,1].plot(MFWF.fValsList[i],MFWF.POIoptListList[i][0]+mf.lorentzian(MFWF.fValsList[i],*params[i][j,:]), '-', label=ESRtext, linewidth=2.5)
        ax[i,1].legend(title="Center \nFrequencies", loc='lower right')

        ax[i,2].set_title("D")
        ax[i,2].set_xticks(roitickpts)
        ax[i,2].set_xticklabels(label_list)
        ax[i,2].set_yticks(roitickpts)
        ax[i,2].set_yticklabels(label_list)
        ax[i,2].set_xlabel("um (Estimated)")
        ax[i,2].plot(MFWF.poiY-MFWF.ylow, MFWF.poiX-MFWF.xlow, 'ro')
        cmap = cm.get_cmap('viridis').copy()
        cmap.set_under('white')
        cmap.set_over('red')
        Dpic=ax[i,2].imshow(MFWF.Dlist[i],vmin=Dmin, vmax=Dmax, cmap=cmap)
        divider = make_axes_locatable(ax[i,2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(Dpic, cax=cax)

        ax[i,3].set_title("E")
        ax[i,3].set_xticks(roitickpts)
        ax[i,3].set_xticklabels(label_list)
        ax[i,3].set_yticks(roitickpts)
        ax[i,3].set_yticklabels(label_list)
        ax[i,3].set_xlabel("um (Estimated)")
        ax[i,3].plot(MFWF.poiY-MFWF.ylow, MFWF.poiX-MFWF.xlow, 'ro')
        cmap = cm.get_cmap('viridis').copy()
        cmap.set_under('white')
        cmap.set_over('red')
        Epic=ax[i,3].imshow(MFWF.Elist[i], vmin=Emin, vmax=Emax, cmap=cmap)
        divider = make_axes_locatable(ax[i,3])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(Epic, cax=cax)

        XX, YY, ZZ = makeContourplot(MFWF.datList[i])
        Contourplot = ax[i,4].contourf(XX, YY, ZZ)
        ax[i,4].set_aspect('equal')
        ax[i,4].set_ylim(ax[i,4].get_ylim()[::-1])
        # rect = Rectangle((ylow,xlow),yrange,xrange,linewidth=1,edgecolor='r',facecolor='none')
        # ax[i,4].add_patch(rect)
        ax[i,4].set_xticks(fulltickpts)
        ax[i,4].set_xticklabels(label_list)
        ax[i,4].set_yticks(fulltickpts)
        ax[i,4].set_yticklabels(label_list)
        ax[i,4].set_xlabel("um (Estimated)")
        ax[i,4].plot(MFWF.poiY, MFWF.poiX, 'ro')
        divider = make_axes_locatable(ax[i,4])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(Contourplot, cax=cax)
def plottwo(MFWF, Ilist, poiX, poiY):
    poiEs, poialpha, poiDelta, poifittedE=poiProcessDandE(MFWF, poiX, poiY)

    gamma=2.8025e-3 #GHz/G
    poiBval=poialpha/gamma

    cmap = cm.get_cmap('rainbow').copy()
    cmap.set_over('lightgray')
    cmap.set_under('lightgray')
    cmap.set_bad('white')

    fulltickpts, label_list=pxToUm(MFWF)
    roitickpts=fulltickpts

    colorlist=['purple', 'mediumslateblue', 'blue','lightseagreen', 'green', 'lime', 'orange', 'tomato', 'red', 'hotpink', 'magenta']
    fig, ax = plt.subplots(1,2, figsize=(8.5, 3.5))
    # ax[0].set_title("D versus Current (A)")
    # ax[0].plot(Ilist[:len(MFWF.poiDs)], MFWF.poiDs, 'o')
    # # ax[0].set_ylim(3.8, 4.2)

    ax[0].set_title("E versus Current (A)")
    ax[0].plot(Ilist[:len(poiEs)], poiEs, 'o')
    text1=r'$E = \sqrt{(\Delta)^2 + (\alpha \bullet I)^2 }$'
    if MFWF.temp=="lt":
        text2=r'$\alpha =$' + str(round(poialpha,3)) + "\n" + str(round(poiBval,3))+ " G/A"
    else:
        text2=r'$\Delta =$' + str(round(poiDelta,3)) + "\n" + r'$\alpha =$' + str(round(poialpha,3)) + "\n" + str(round(poiBval,3))+ " G/A"

    ax[0].plot(Ilist[:MFWF.nfiles], poifittedE, c='mediumslateblue', linestyle="--", alpha=0.8)
    ax[0].text(0.35, 0.75, text1+"\n"+text2, fontsize=14, horizontalalignment='center', verticalalignment='center', transform=ax[1].transAxes)

    for i in np.arange(MFWF.nfiles):
        ax[1].set_title("ESR Plots Together")
        # for j in np.arange(nfiles):
        ax[1].plot(MFWF.fValsList[i], MFWF.POIyVals[i], label=str(MFWF.filenamearr[i]), c=colorlist[i])
        # ax[2].plot(esrXlist[j], esrFitList[j], label=str(filenamearr[j][:-4])+" Fit", c=colorlist[j], alpha=0.5, linestyle="--")
        ax[1].legend(loc='upper left', bbox_to_anchor=(1.05, 0.95))
    plt.show()
def plotthree(MFWF, alphalow, alphahigh):
    cmap = cm.get_cmap('rainbow').copy()
    cmap.set_over('lightgray')
    cmap.set_under('lightgray')
    cmap.set_bad('white')

    fulltickpts, label_list=pxToUm(MFWF)
    roitickpts=fulltickpts

    #Masking
    for i in MFWF.xr:
        for j in MFWF.yr:
            if MFWF.badptmap[i,j]==1 or MFWF.Deltamap[i,j]==1 or MFWF.alphamap[i,j]==1:
                MFWF.Deltamap[i,j]=np.nan
                MFWF.alphamap[i,j]=np.nan
    MFWF.Dlist=np.ma.masked_invalid(MFWF.Dlist)
    MFWF.Elist=np.ma.masked_invalid(MFWF.Elist)
    MFWF.Deltamap=np.ma.masked_invalid(MFWF.Deltamap)
    MFWF.alphamap=np.ma.masked_invalid(MFWF.alphamap)

    fig,ax=plt.subplots(1,2, figsize=(23, 10))
    ax[0].set_title("Delta Map", fontsize=25)
    ax[0].set_xticks(fulltickpts)
    ax[0].set_xticklabels(label_list)
    ax[0].set_yticks(fulltickpts)
    ax[0].set_yticklabels(label_list)
    ax[0].set_xlabel("um (Estimated)")
    DeltaMAP=ax[0].imshow(MFWF.Deltamap, vmin=0, cmap=cmap, interpolation=None, origin='upper')
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(DeltaMAP, cax=cax)

    ax[1].set_title("Alpha Map", fontsize=25)
    ax[1].set_xticks(fulltickpts)
    ax[1].set_xticklabels(label_list)
    ax[1].set_yticks(fulltickpts)
    ax[1].set_yticklabels(label_list)
    ax[1].set_xlabel("um (Estimated)")
    alphaMAP=ax[1].imshow(MFWF.alphamap, vmin=alphalow, vmax=alphahigh, cmap=cmap, interpolation=None, origin='upper')
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar=plt.colorbar(alphaMAP, cax=cax)

    # ax[1].set_title("B/H Map", fontsize=25)
    # ax[1].set_xticks(fulltickpts)
    # ax[1].set_xticklabels(label_list)
    # ax[1].set_yticks(fulltickpts)
    # ax[1].set_yticklabels(label_list)
    # ax[1].set_xlabel("um (Estimated)")
    # alphaMAP=ax[1].imshow(MFWF.alphamap/0.069, vmin=alphalow, vmax=alphahigh, cmap=cmap, interpolation=None, origin='upper')
    # divider = make_axes_locatable(ax[1])
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # cbar=plt.colorbar(alphaMAP, cax=cax)

    plt.show()


# # %%
# rtfolderpath='C:/Users/esthe/OneDrive/Desktop/VSCode/Plotting/Data/WF/RT CeH9 Data/'
# ltfolderpath='C:/Users/esthe/OneDrive/Desktop/VSCode/Plotting/Data/WF/LT CeH9 Data/'
# rtIlist=[3.0, 2.7, 2.4, 2.1, 1.8, 1.5, 1.2, 0.9, 0.6, 0.3, 0]
# ltIlist=[3.0,2.5,2.0]
# # ltIlist=[3.0,2.0]
# # Ilist = Ilist.astype(np.float)
# # %%
# temp="lt"
# MFWF=MultiFileWF(ltfolderpath, ltIlist, temp)

# # temp="rt"
# # MFWF=MultiFileWF(rtfolderpath, rtIlist, temp)

# # %%
# MFWF.roi()
# MFWF.MF_multiESRFit()
# # %%
# picklefolder='C:/Users/esthe/OneDrive/Desktop/VSCode/Plotting/Data/WF/LT CeH9 Pickle/'
# with open(picklefolder + 'LT_4bin_CenterNorm_linFit_fixedax.pkl', 'rb') as f:
#     MFWF = pickle.load(f)
# # plottwo(ltMFWFload, ltIlist)
# # plotthree(MFWF, 0.0, 0.072)
# # %%

# poiX=10
# poiY=5
# alphalow=0
# alphahigh=0.072
# MFWF.MF_POIFit(poiX,poiY)
# poiDandE(MFWF, poiX, poiY)
# DandE(MFWF)
# processDandE(MFWF, poiX, poiY, alphalow, alphahigh)
# # %% 
# # plotone(MFWF)
# # plottwo(MFWF, rtIlist, poiX, poiY)
# plotthree(MFWF, alphalow, alphahigh)
# # manualEntry(MFWF)
# # %%
# # %%
# for WF in MFWF.WFList:
#     WF.myFavoriatePlot(20, 11)
# # %%
# import pickle
# # # # Serialize the object to a file
# picklefolder='C:/Users/esthe/OneDrive/Desktop/VSCode/Plotting/Data/WF/LT CeH9 Pickle/'
# # with open(picklefolder + 'LT_4bin_CenterNorm_linFit_fixedax.pkl', 'wb') as f:
# #     pickle.dump(MFWF, f)

# # %%
# # import pickle
# # # # Serialize the object to a file
# # picklefolder='C:/Users/esthe/OneDrive/Desktop/VSCode/Plotting/Data/WF/LT CeH9 Pickle/'
# with open(picklefolder + 'LT_4bin_CenterNorm_linFit_fixedax.pkl', 'rb') as f:
#     ltMFWFload = pickle.load(f)
# # plottwo(ltMFWFload, ltIlist)
# plotthree(ltMFWFload, 0.0, 0.072)

# # with open(picklefolder + 'LT_3bin_CenterNorm_linFit_no2p5.pkl', 'rb') as f:
# #     ltMFWFload = pickle.load(f)
# # # plottwo(ltMFWFload, ltIlist)
# # plotthree(ltMFWFload, 0.0, 0.072)


# with open(picklefolder + 'RT_4bin_CenterNorm_fixedax.pkl', 'rb') as f:
#     rtMFWFload = pickle.load(f)
# # plottwo(rtMFWFload, rtIlist)
# plotthree(rtMFWFload, 0.055, 0.085)
# # %%
# # %%
# from matplotlib.ticker import FormatStrFormatter

# def makeLineCut(startpt, endpt):
#     startx=startpt[0]
#     starty=startpt[1]
#     endx=endpt[0]
#     endy=endpt[1]

#     if startx<endx:
#         xline=np.floor(np.arange(startx, endx, 1))
#     else:
#         xline=np.floor(np.arange(startx, endx, -1))
#     if starty<endy:
#         yline=np.floor(np.arange(starty, endy, 1))
#     else:
#         yline=np.floor(np.arange(starty, endy, -1))

#     print("original lengths: "+"x: "+ str(len(xline))+"y: "+ str(len(yline)))
#     linelength=max(len(xline),len(yline))
#     if len(xline) !=linelength:
#         xline=np.linspace(startx, endx, linelength)
#     if len(yline) !=linelength:
#         yline=np.linspace(starty, endy, linelength)
#     xline=np.round(xline[:linelength], 0).astype(int)
#     yline=np.round(yline[:linelength], 0).astype(int)

#     dist=np.sqrt((endx-startx)**2+(endy-starty)**2)
#     distvals=np.linspace(0,dist,len(xline))

#     # print(xline)
#     # print(yline)
#     # print(dist)

#     return xline, yline, distvals
#     # EvsIcomp(xline, yline)

# def boverhplot(startpt, endpt):
#     #Line cut processing
#     xline, yline, distvals=makeLineCut(startpt, endpt)
#     ##Getting H
#     RTalphaList=[]
#     for i in np.arange(rtMFWFload.WFList[0].X):
#         for j in np.arange(rtMFWFload.WFList[0].Y):
#             if rtMFWFload.alphamap[i,j]>0.04:
#                 RTalphaList.append(rtMFWFload.alphamap[i,j])
#     RTalphaList.sort()
#     fig,ax=plt.subplots(1,1)
#     ax.hist(RTalphaList, bins=20)
#     ax.set_title("RTalphaList Before Cut")
#     alphaListLow=np.floor(len(RTalphaList)*0.3).astype(int)
#     alphaListHigh=np.floor(len(RTalphaList)*0.8).astype(int)
#     fig,ax=plt.subplots(1,1)
#     rtH=np.mean(RTalphaList[alphaListLow:alphaListHigh])
#     ax.hist(RTalphaList[alphaListLow:alphaListHigh], bins=20)
#     ax.set_title("RTalphaList After Cut")
#     print(RTalphaList[alphaListLow:alphaListHigh])
#     print(rtH)

#     ##Getting B/H
#     alphaline=[]
#     poiEsline=[]
#     poifittedEsline=[]
#     boverhline=[]
#     # for i in np.arange(len(xline)):
#     #     poiEs, poialpha, poifittedE=poiProcessDandE(ltMFWFload, xline[i], yline[i])
#     #     poiEsline.append(poiEs)
#     #     poifittedEsline.append(poifittedE)
#     #     alphaline.append(poialpha)
#     #     boverhline.append(poialpha/rtH)
#     for i in np.arange(len(xline)):
#         poialpha=ltMFWFload.alphamap[yline[i],xline[i]]
#         alphaline.append(poialpha)
#         print(str(xline[i]) + ", " + str(yline[i]))
#         print("poialpha: "+str(poialpha))
#         boverhline.append(poialpha/rtH)

#     # #Masking
#     # for i in ltMFWFload.xr:
#     #     for j in ltMFWFload.yr:
#     #         if ltMFWFload.badptmap[i,j]==1 or ltMFWFload.Deltamap[i,j]==1 or ltMFWFload.alphamap[i,j]==1:
#     #             ltMFWFload.alphamap[i,j]=np.nan
#     # ltMFWFload.alphamap=np.ma.masked_invalid(ltMFWFload.alphamap)

#     #Plotting
#     fig,ax=plt.subplots(1,2, figsize=(23, 10))

#     cmap = cm.get_cmap('rainbow').copy()
#     # cmap.set_over('lightgray')
#     # cmap.set_under('lightgray')
#     # cmap.set_bad('white')

#     fulltickpts, label_list=pxToUm(ltMFWFload)
#     roitickpts=fulltickpts

#     ax[0].set_title("Alpha Map", fontsize=25)
#     ax[0].set_xticks(fulltickpts)
#     ax[0].set_xticklabels(label_list)
#     ax[0].set_yticks(fulltickpts)
#     ax[0].set_yticklabels(label_list)
#     ax[0].set_xlabel("um (Estimated)")
#     ax[0].scatter(xline, yline, c="white", marker="*", s=200)
#     alphaMAP=ax[0].imshow(ltMFWFload.alphamap, vmin=0.0, vmax=0.072, cmap=cmap, interpolation=None, origin='upper')
#     ax[0].set_aspect('equal')
#     divider = make_axes_locatable(ax[0])
#     cax = divider.append_axes("right", size="5%", pad=0.05)
#     cbar=plt.colorbar(alphaMAP, cax=cax)

#     ax[1].set_title("B/H Line Cut", fontsize=25)
#     ax[1].plot(distvals, boverhline,"o-")
#     ax[1].plot(distvals, np.ones(len(distvals)),"--", alpha=0.5)
#     ax[1].set_ylim(0.0, 1)
#     ax[1].set_ylabel("B/H")
#     ax[1].set_xlabel("Distance (um)")

#     # colorlist=['violet', 'mediumorchid','purple', 'mediumslateblue', 'blue', 'dodgerblue','cyan','lightseagreen', 'green', 'limegreen','lime', 'orange', 'tomato', 'red', 'hotpink', 'magenta']
#     # fig,ax=plt.subplots(1,1, figsize=(5,5))
#     # fig.suptitle("Individual E vs I")
#     # for i in np.arange(len(alphaline)):
#     #     ax.plot(ltMFWFload.Ilist, poiEsline[i], 'o', c=colorlist[i], label=round(distvals[i],2))
#     #     ax.plot(ltMFWFload.Ilist, poifittedEsline[i], c=colorlist[i], linestyle="--", alpha=0.5)
#     #     ax.legend(title="Distance", loc='upper left', bbox_to_anchor=(1.05, 0.95))
    

# # %%
# startpt=[15,23]
# endpt=[16, 23]
# # xline, yline, distvals=makeLineCut(startpt, endpt)
# xline, yline, distvals=makeLineCut(endpt, startpt)
# print(xline, yline)
# boverhplot(startpt, endpt)
# print("pointval: "+str(ltMFWFload.alphamap[23, 14]))

# # boverhplot(endpt, startpt)
# # %%
# # with open(picklefolder + 'LT_4bin_CenterNorm_linFit_fixedax.pkl', 'rb') as f:
# #     ltMFWFload1 = pickle.load(f)
# # # plottwo(ltMFWFload, ltIlist)
# # plotthree(ltMFWFload, 0.0, 0.072)


# # startpt=[20,15]
# # endpt=[20, 10]
# # xline, yline, distvals=makeLineCut(startpt, endpt)

# # with open(picklefolder + 'LT_3bin_CenterNorm.pkl', 'rb') as f:
# #     ltMFWFload2 = pickle.load(f)
# #     ltMFWFload2.temp="lt"
# # # plottwo(ltMFWFload, ltIlist)
# # plotthree(ltMFWFload, 0.0, 0.072)
# for i in np.arange(len(xline)):
#     ltMFWFload.MF_POIFit(xline[i],yline[i])
#     # poiDandE(lt2MFWFload,xline[i],yline[i])
#     plottwo(ltMFWFload, ltIlist, xline[i],yline[i])

# # for i in np.arange(len(xline)):
# #     ltMFWFload2.MF_POIFit(xline[i],yline[i])
# #     # poiDandE(lt2MFWFload,xline[i],yline[i])
# #     plottwo(ltMFWFload2, ltIlist, xline[i],yline[i])

# # %%
# ltMFWFload.WFList[0].myFavoriatePlot(15, 18)
# # %%

# # val = input("Enter your value: ")
# # print(val)
# # %%
folderpath='C:/Users/esthe/OneDrive/Desktop/VSCode/Plotting/Data/WF/RT NLuH Data/'
filename='wf_1p4W_n30dbm_2A'
# Ilist=[3.0, 2.7, 2.4, 2.1, 1.8, 1.5, 1.2, 0.9, 0.6, 0.3, 0]

WF1=WFimage(folderpath + filename)

img1= WF1.sliceImage()
plt.imshow(img1)
# %% 
WF1.myFavoriatePlot(20, 11)
# %%
