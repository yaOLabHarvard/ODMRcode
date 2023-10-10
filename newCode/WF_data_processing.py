import WF_mat_data_class as wf
import multipeak_fit as mf
from matplotlib import pyplot as plt, cm
from matplotlib.patches import Rectangle
import pickle
import numpy as np
import scipy.optimize as opt
from mpl_toolkits.axes_grid1 import make_axes_locatable
gamma=2.8025e-3 #GHz/G

# %% EvsI and makeContourPlot
def EvsI(x, Delta, alpha):
    return np.sqrt(Delta**2+(alpha*x)**2)
def linFit(x, c, alpha):
    return c+alpha*x
def linFitZero(x, alpha):
    return alpha*x
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

# %%
def poiDandE(MFWF, poiX, poiY): #Single Point
    if MFWF.poiFit==True:
        poiDs=[]
        poiEs=[]
        for i in np.arange(len(MFWF.WFList)):
            if MFWF.optListList[i][poiX, poiY] is not None:
                peakFreqs = MFWF.optListList[i][poiX, poiY][0::3][1:]
                Fmax = max(peakFreqs)
                Fmin = min(peakFreqs)
                poiDs.append((Fmin + Fmax)/2)
                poiEs.append(np.abs((Fmax - Fmin)/2))

                # pOpt=MFWF.optListList[i][poiX, poiY]
                # poiDs.append((pOpt[3]+pOpt[-1])/2) #First and last peak freqs
                # poiEs.append(abs(pOpt[3]-pOpt[-1])/2) #Equivalent to singleESREs
        # print("This is self.poiEs:" + str(self.poiEs))
            else: print("BAD POI! Didn't append POI D and POI E for file %i."%(i))
    else:
        print("Please run MF_POIFit() first!")
    return poiDs, poiEs

def poiProcessDandE(MFWF, poiX, poiY, method='EvsI'):
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

        if method=='linFit':
            EpOpt,EpCov= opt.curve_fit(linFit, MFWF.Ilist[:MFWF.nfiles], poiEs)
            poiDelta=EpOpt[0].astype(float)
            poialpha=EpOpt[1].astype(float)
            # poiBval=MFWF.poialpha/gamma #G/A
            poifittedE=linFit(MFWF.Ilist[:MFWF.nfiles], *EpOpt)
            return poiEs, poialpha, poiDelta, poifittedE, EpCov
        elif method=='EvsI':
            EpOpt,EpCov= opt.curve_fit(EvsI, MFWF.Ilist[:MFWF.nfiles], poiEs)
            print("EpOpt: " + str(EpOpt))
            # else:
                # print('Bad POI! Choose another point!')
            poiDelta=EpOpt[0].astype(float)
            poialpha=EpOpt[1].astype(float)
            # poiBval=self.poialpha/gamma #G/A
            poifittedE=EvsI(MFWF.Ilist[:MFWF.nfiles], poiDelta, poialpha)
            return poiEs, poialpha, poiDelta, poifittedE, EpCov
        elif method=='linFitZero':
            EpOpt,EpCov= opt.curve_fit(linFitZero, MFWF.Ilist[:MFWF.nfiles], poiEs)
            poiDelta=0
            poialpha=EpOpt[0].astype(float)
            poifittedE=linFitZero(MFWF.Ilist[:MFWF.nfiles], *EpOpt)
            return poiEs, poialpha, poiDelta, poifittedE, EpCov
        else:
            print("Please choose a valid method: EvsI, linFit, or linFitZero")

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

def processDandE(MFWF, poiX, poiY, alphalow, alphahigh, method='EvsI'):
    if MFWF.allMultFitandN==True and MFWF.poiFit==True:
        Dmin=0
        Dmax=8
        Emin=0
        Emax=3
        # Dmin=3.5
        # Dmax=4.3
        # Emin=0.0
        # Emax=0.4

        #Find the bad points
        MFWF.badptmap=np.zeros((len(MFWF.xr),len(MFWF.yr)))
        for i in np.arange(MFWF.nfiles):
            for j in MFWF.xr:
                for k in MFWF.yr:
                    if MFWF.Dlist[i][j-MFWF.xlow,k-MFWF.ylow]<Dmin or MFWF.Dlist[i][j-MFWF.xlow,k-MFWF.ylow]>Dmax or MFWF.Elist[i][j-MFWF.xlow,k-MFWF.ylow]<Emin or MFWF.Elist[i][j-MFWF.xlow,k-MFWF.ylow]>Emax:
                        # MFWF.badptmap[j-MFWF.xlow,k-MFWF.ylow]=1
                        MFWF.badptmap[j-MFWF.xlow,k-MFWF.ylow]=0 #If you don't want to use bpm
        fig,ax=plt.subplots(1,1)
        ax.set_title("Bad Point Map (1 is Bad)")
        BPM=ax.imshow(MFWF.badptmap, interpolation=None)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(BPM, cax=cax)


        ##Fit the Data for POI! This will be used as the guess parameters.
        # # if np.shape(self.poiEs)!=():
        # print("Ilist: "+ str(self.Ilist[:self.nfiles]))
        # print("poiEs: "+ str(self.poiEs))
        # poiEs, poialpha, poiDelta, poifittedE, EpCov=poiProcessDandE(MFWF, poiX, poiY) Commented out on 10/6/2023
        
        if method=='linFit':
            poiEs, poialpha, poiDelta, poifittedE, EpCov=poiProcessDandE(MFWF, poiX, poiY, method='linFit')
            EpOpt,EpCov= opt.curve_fit(linFit, MFWF.Ilist[:MFWF.nfiles], poiEs)
            MFWF.poiDelta=EpOpt[0].astype(float)
            MFWF.poialpha=EpOpt[1].astype(float)
            MFWF.poiBval=MFWF.poialpha/gamma #G/A
            MFWF.poifittedE=linFit(MFWF.Ilist[:MFWF.nfiles], *EpOpt)
        elif method=='EvsI':
            poiEs, poialpha, poiDelta, poifittedE, EpCov=poiProcessDandE(MFWF, poiX, poiY, method='EvsI')
            EpOpt,EpCov= opt.curve_fit(EvsI, MFWF.Ilist[:MFWF.nfiles], poiEs)
            print("EpOpt: " + str(EpOpt))
            # else:
                # print('Bad POI! Choose another point!')
            MFWF.poiDelta=EpOpt[0].astype(float)
            MFWF.poialpha=EpOpt[1].astype(float)
            MFWF.poiBval=MFWF.poialpha/gamma #G/A
            MFWF.poifittedE=EvsI(MFWF.Ilist[:MFWF.nfiles], MFWF.poiDelta, MFWF.poialpha)
        if method=='linFitZero':
            poiEs, poialpha, poiDelta, poifittedE, EpCov=poiProcessDandE(MFWF, poiX, poiY, method='linFitZero')
            EpOpt,EpCov= opt.curve_fit(linFitZero, MFWF.Ilist[:MFWF.nfiles], poiEs)
            MFWF.poiDelta=0
            MFWF.poialpha=EpOpt[0].astype(float)
            MFWF.poiBval=MFWF.poialpha/gamma #G/A
            MFWF.poifittedE=linFitZero(MFWF.Ilist[:MFWF.nfiles], *EpOpt)

        # if onept==False:
        MFWF.ESpOptmap=np.zeros((len(MFWF.xr),len(MFWF.yr),2))
        MFWF.Deltamap=np.zeros((len(MFWF.xr),len(MFWF.yr)))
        MFWF.alphamap=np.ones((len(MFWF.xr),len(MFWF.yr)))
        MFWF.Bmap=np.zeros((len(MFWF.xr),len(MFWF.yr)))
        for i in MFWF.xr:
            for j in MFWF.yr:
                if MFWF.badptmap[i-MFWF.xlow,j-MFWF.ylow]!=1:
                    singlePointE=[]
                    for k in np.arange(len(MFWF.Elist)):
                        singlePointE.append(MFWF.Elist[k][i-MFWF.xlow,j-MFWF.ylow])
                    if method=='linFit':
                        try:
                            ESpOpt,pCov= opt.curve_fit(linFit, MFWF.Ilist[:MFWF.nfiles], singlePointE)
                        except opt.OptimizeWarning:
                            print('Pixel:[%i,%i] OptimizeWarning'%(i,j))
                            continue
                        except RuntimeError:
                            print('Pixel:[%i,%i] encountered RuntimeError'%(i,j))
                            continue
                        except ValueError:
                            print('Pixel:[%i,%i] encountered ValueError for EvsI fit'%(i,j))
                            continue
                        MFWF.ESpOptmap[i-MFWF.xlow,j-MFWF.ylow,:]=ESpOpt
                        MFWF.Deltamap[i-MFWF.xlow,j-MFWF.ylow]=ESpOpt[0] #Actually cmap
                        MFWF.alphamap[i-MFWF.xlow,j-MFWF.ylow]=ESpOpt[1]
                        # MFWF.Bmap[i-MFWF.xlow,j-MFWF.ylow]=ESpOpt[1]/gamma #G/A
                    elif method=='EvsI':
                        try:
                            ESpOpt,pCov= opt.curve_fit(EvsI, MFWF.Ilist[:MFWF.nfiles], singlePointE, p0=EpOpt)
                        except opt.OptimizeWarning:
                            print('Pixel:[%i,%i] OptimizeWarning'%(i,j))
                            continue
                        except RuntimeError:
                            print('Pixel:[%i,%i] encountered RuntimeError'%(i,j))
                            continue
                        except ValueError:
                            print('Pixel:[%i,%i] encountered ValueError for EvsI fit'%(i,j))
                            continue
                        MFWF.ESpOptmap[i-MFWF.xlow,j-MFWF.ylow,:]=ESpOpt
                        MFWF.Deltamap[i-MFWF.xlow,j-MFWF.ylow]=ESpOpt[0]
                        MFWF.alphamap[i-MFWF.xlow,j-MFWF.ylow]=ESpOpt[1]
                        MFWF.Bmap[i-MFWF.xlow,j-MFWF.ylow]=ESpOpt[1]/gamma #G/A
                    if method=='linFitZero':
                        try:
                            ESpOpt,pCov= opt.curve_fit(linFitZero, MFWF.Ilist[:MFWF.nfiles], singlePointE)
                        except opt.OptimizeWarning:
                            print('Pixel:[%i,%i] OptimizeWarning'%(i,j))
                            continue
                        except RuntimeError:
                            print('Pixel:[%i,%i] encountered RuntimeError'%(i,j))
                            continue
                        except ValueError:
                            print('Pixel:[%i,%i] encountered ValueError for EvsI fit'%(i,j))
                            continue
                        MFWF.ESpOptmap[i-MFWF.xlow,j-MFWF.ylow,:]=ESpOpt
                        MFWF.Deltamap[i-MFWF.xlow,j-MFWF.ylow]=0
                        MFWF.alphamap[i-MFWF.xlow,j-MFWF.ylow]=ESpOpt[1]
                        # MFWF.Bmap[i-MFWF.xlow,j-MFWF.ylow]=ESpOpt[1]/gamma #G/A
                else:
                    MFWF.ESpOptmap[i-MFWF.xlow,j-MFWF.ylow,:]=[0,0]
                    MFWF.Deltamap[i-MFWF.xlow,j-MFWF.ylow]=1
                    MFWF.alphamap[i-MFWF.xlow,j-MFWF.ylow]=1
                    MFWF.Bmap[i-MFWF.xlow,j-MFWF.ylow]=0
        
        alphaDeltaPlots(MFWF, alphalow, alphahigh, H=0.07)
        manual=input("Would you like to manually fix points? y/n")
        if manual=='y':
            manualEntry(MFWF, alphalow, alphahigh, method)

        MFWF.EvsIfit=True
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

def manualEntry(self, alphalow, alphahigh, method="EvsI"):
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

            if method=='linFit':
                try:
                    ESpOpt,pCov= opt.curve_fit(linFit, self.Ilist[:self.nfiles], inputElist)
                    manESpOptmap[xloc, yloc]=ESpOpt
                except ValueError:
                    print('Something went wrong...')
                    continue
                self.alphamap[xloc, yloc]=manESpOptmap[xloc, yloc][0]
                self.Deltamap[xloc, yloc]=manESpOptmap[xloc, yloc][1]
            elif method=='EvsI':
                try:
                    # ESpOpt,pCov= opt.curve_fit(EvsI, self.Ilist[:self.nfiles], inputElist, p0=EpOpt)
                    ESpOpt,pCov= opt.curve_fit(EvsI, self.Ilist[:self.nfiles], inputElist)
                    manESpOptmap[xloc, yloc]=ESpOpt
                except ValueError:
                    print('Something went wrong...')
                    continue
                self.alphamap[xloc, yloc]=manESpOptmap[xloc, yloc][0]
                self.Deltamap[xloc, yloc]=manESpOptmap[xloc, yloc][1]
            elif method=='linFitZero':
                try:
                    ESpOpt,pCov= opt.curve_fit(linFitZero, self.Ilist[:self.nfiles], inputElist)
                    manESpOptmap[xloc, yloc]=ESpOpt
                except ValueError:
                    print('Something went wrong...')
                    continue
                self.alphamap[xloc, yloc]=manESpOptmap[xloc, yloc][0]
                self.Deltamap[xloc, yloc]=0
            

            alphaDeltaPlots(self, alphalow, alphahigh)
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
                            self.Deltamap[i-self.xlow,j-self.ylow]=ESpOpt[0] #Actually cmap
                            self.alphamap[i-self.xlow,j-self.ylow]=ESpOpt[1]
                        else:
                            # ESpOpt,pCov= opt.curve_fit(EvsI, self.Ilist[:self.nfiles], inputElist, p0=EpOpt)
                            ESpOpt,pCov= opt.curve_fit(EvsI, self.Ilist[:self.nfiles], inputElist)
                            self.alphamap[i-self.xlow,j-self.ylow]=ESpOpt[1]
                            self.Deltamap[i-self.xlow,j-self.ylow]=ESpOpt[0]
                        alphaDeltaPlots(self, alphalow, alphahigh)
                    elif yn=='e':
                        break
                    # print(val)


# %%

def pxToUm(MFWF):
    # UmPerPx=1 #No conversion
    # UmPerPx=0.175 #No bins
    # UmPerPx=20/38 #3x3 bin
    # UmPerPx=20/28 #4x4 bin
    UmPerPx=175/168 #3x3 bin for LuH-N
    fulltickpts=np.linspace(0, MFWF.WFList[0].X, 5)
    # label_list = round(np.multiply(fulltickpts, UmPerPx), 3)
    label_list = []
    for i in fulltickpts:
        label_list.append(round(np.multiply(i, UmPerPx), 3))
    return fulltickpts, label_list
    
def fullDataPlot(MFWF, immin=0, immax=0, Dmin=0, Dmax=0, Emin=0, Emax=0): #Previously plotone
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

    # Dmin=3.5
    # Dmax=4.3
    # Emin=0.0
    # Emax=0.4
    fig, ax = plt.subplots(MFWF.nfiles,5, figsize=(20, 3.5*MFWF.nfiles+1))
    fig.tight_layout(pad=2.5)
    params=[]
    for i in np.arange(MFWF.nfiles):
        ccmap = cm.get_cmap('viridis').copy()
        # ax[i,0].imshow(MFWF.datList[i][:,:,3], vmin=0.9994, vmax=1.001,cmap=ccmap)
        if immin==0 and immax==0:
            ax[i,0].imshow(MFWF.datList[i][:,:,3],cmap=ccmap)
        else:
            ax[i,0].imshow(MFWF.datList[i][:,:,3], vmin=immin, vmax=immax,cmap=ccmap)
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
        #For plotting all peak fits:
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
        if Dmin==0 and Dmax==0:
            Dpic=ax[i,2].imshow(MFWF.Dlist[i], cmap=cmap)
        else:
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
        if Emin==0 and Emax==0:
            Epic=ax[i,3].imshow(MFWF.Elist[i], cmap=cmap)
        else:
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

def EvsIPlot(MFWF, Ilist, poiX, poiY, method):
    poiEs, poialpha, poiDelta, poifittedE, EpCov=poiProcessDandE(MFWF, poiX, poiY)

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
    ax[0].set_xlabel("Current (A)")
    ax[0].set_ylabel("E Splitting (GHz)")
    ax[0].plot(Ilist[:len(poiEs)], poiEs, 'o')
    if method=='linFit':
        text1=r'$E = \alpha \bullet I + c$'
        text2=r'$\c =$' + str(round(poiDelta,3)) + "\n" + r'$\alpha =$' + str(round(poialpha,3)) + "\n" + str(round(poiBval,3))+ " G/A"
    elif method=='EvsI':
        text1=r'$E = \sqrt{(\Delta)^2 + (\alpha \bullet I)^2 }$'
        text2=r'$\Delta =$' + str(round(poiDelta,3)) + "\n" + r'$\alpha =$' + str(round(poialpha,3)) + "\n" + str(round(poiBval,3))+ " G/A"
    elif method=='linFitZero':
        text1=r'$E = \alpha \bullet I$'
        text2=r'$\alpha =$' + str(round(poialpha,3)) + "\n" + str(round(poiBval,3))+ " G/A"

    ax[0].plot(Ilist[:MFWF.nfiles], poifittedE, c='mediumslateblue', linestyle="--", alpha=0.8)
    ax[0].text(0.35, 0.75, text1+"\n"+text2, fontsize=14, horizontalalignment='center', verticalalignment='center', transform=ax[1].transAxes)
    print(text2)

    ax[1].set_title("ESR Plots Together")
    ax[1].set_xlabel("Frequency (GHz)")
    ax[1].set_ylabel("Contrast")
    for i in np.arange(MFWF.nfiles):
        # for j in np.arange(nfiles):
        ax[1].plot(MFWF.fValsList[i], MFWF.POIyVals[i], label=str(MFWF.filenamearr[i]), c=colorlist[i])
        # ax[2].plot(esrXlist[j], esrFitList[j], label=str(filenamearr[j][:-4])+" Fit", c=colorlist[j], alpha=0.5, linestyle="--")
    ax[1].legend(loc='upper left', bbox_to_anchor=(1.05, 0.95))
    plt.show()

def getH(MFWF, alphalow, alphahigh, hxl, hxh, hyl, hyh, histbins, gam=False):
    if gam==True:
        alphalow=alphalow/gamma
        alphahigh=alphahigh/gamma

    fulltickpts, label_list=pxToUm(MFWF)
    roitickpts=fulltickpts

    cmap = cm.get_cmap('rainbow').copy()
    cmap.set_over('lightgray')
    cmap.set_under('lightgray')
    cmap.set_bad('white')

    fig, ax=plt.subplots(1,2, figsize=(20,10))
    if gam==True:
        alphaMAP=ax[0].imshow(MFWF.alphamap/gamma, vmin=alphalow, vmax=alphahigh, cmap=cmap, interpolation=None, origin='upper')
        ax[0].set_title("Spatial Uniformity of B Values")
    else:
        alphaMAP=ax[0].imshow(MFWF.alphamap, vmin=alphalow, vmax=alphahigh, cmap=cmap, interpolation=None, origin='upper')
        ax[0].set_title("Spatial Uniformity of alpha Values")
    rect = Rectangle((hxl,hyl),hxh-hxl,hyh-hyl,linewidth=1,edgecolor='r',facecolor='none')
    ax[0].add_patch(rect)
    ax[0].set_xticks(fulltickpts)
    ax[0].set_xticklabels(label_list)
    ax[0].set_yticks(fulltickpts)
    ax[0].set_yticklabels(label_list)
    ax[0].set_xlabel("um (Estimated)")
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar=plt.colorbar(alphaMAP, cax=cax)

    # alphaMAPCut=ax[1].imshow(MFWF.alphamap[hxl:hxh, hyl:hyh], vmin=alphalow, vmax=alphahigh, cmap=cmap, interpolation=None, origin='upper')
    # print("Mean of chosen region: "+str(np.mean(MFWF.alphamap[hxl:hxh, hyl:hyh])))
    histList=[]
    for i in np.arange(len(MFWF.alphamap[hxl:hxh,:])):
        for j in np.arange(len(MFWF.alphamap[:,hxl:hxh])):
            if gam==True:
                histList.append(MFWF.alphamap[i,j]/gamma)
            else:
                histList.append(MFWF.alphamap[i,j])
    if gam==True:
        ax[1].hist(histList,bins=histbins, range=(alphalow, alphahigh))
        ax[1].set_title("Histogram of B Values")
        ax[1].set_xlabel("B Values")
    else:
        ax[1].hist(histList,bins=histbins, range=(alphalow, alphahigh))
        ax[1].set_title("Histogram of alpha Values")
        ax[1].set_xlabel("alpha Values")

def alphaDeltaPlots(MFWF, alphalow, alphahigh, H):
    cmap = cm.get_cmap('rainbow').copy()
    cmap.set_over('lightgray')
    cmap.set_under('lightgray')
    cmap.set_bad('white')

    

    fulltickpts, label_list=pxToUm(MFWF)
    roitickpts=fulltickpts

    #Masking
    for i in MFWF.xr:
        for j in MFWF.yr:
            if MFWF.badptmap[i-MFWF.xlow,j-MFWF.ylow]==1 or MFWF.Deltamap[i-MFWF.xlow,j-MFWF.ylow]==1 or MFWF.alphamap[i-MFWF.xlow,j-MFWF.ylow]==1:
                MFWF.Deltamap[i-MFWF.xlow,j-MFWF.ylow]=np.nan
                MFWF.alphamap[i-MFWF.xlow,j-MFWF.ylow]=np.nan
    MFWF.Dlist=np.ma.masked_invalid(MFWF.Dlist)
    MFWF.Elist=np.ma.masked_invalid(MFWF.Elist)
    MFWF.Deltamap=np.ma.masked_invalid(MFWF.Deltamap)
    MFWF.alphamap=np.ma.masked_invalid(MFWF.alphamap)

    fig,ax=plt.subplots(1,1, figsize=(18, 18))
    # fig,ax=plt.subplots(1,1, figsize=(23, 10))
    # ax[0].set_title("Delta Map", fontsize=25)
    # ax[0].set_xticks(fulltickpts)
    # ax[0].set_xticklabels(label_list)
    # ax[0].set_yticks(fulltickpts)
    # ax[0].set_yticklabels(label_list)
    # ax[0].set_xlabel("um (Estimated)")
    # DeltaMAP=ax[0].imshow(MFWF.Deltamap, vmin=0, cmap=cmap, interpolation=None, origin='upper')
    # divider = make_axes_locatable(ax[0])
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # plt.colorbar(DeltaMAP, cax=cax)

    # ax.set_title("Alpha Map", fontsize=25)
    # ax.set_xticks(fulltickpts)
    # ax.set_xticklabels(label_list)
    # ax.set_yticks(fulltickpts)
    # ax.set_yticklabels(label_list)
    # ax.set_xlabel("um (Estimated)")
    # alphaMAP=ax.imshow(MFWF.alphamap, vmin=alphalow, vmax=alphahigh, cmap=cmap, interpolation=None, origin='upper')
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # cbar=plt.colorbar(alphaMAP, cax=cax)

    ax.set_title("B/H Map", fontsize=25)
    ax.set_xticks(fulltickpts)
    ax.set_xticklabels(label_list)
    ax.set_yticks(fulltickpts)
    ax.set_yticklabels(label_list)
    ax.set_xlabel("um (Estimated)")
    alphaMAP=ax.imshow(MFWF.alphamap/H, vmin=alphalow, vmax=alphahigh, cmap=cmap, interpolation=None, origin='upper')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar=plt.colorbar(alphaMAP, cax=cax)

    plt.show()

def zfsTP(T, P):
    zfsT = 2.8771+ -4.625e-6*T+ 1.067e-7*T*T+ -9.325e-10*T*T*T+ 1.739e-12*T*T*T*T+ -1.838e-15*T*T*T*T*T
    zfsP = 1e-2*P
    return zfsT+zfsP

def DPlot(MFWF, Dmin=0, Dmax=0, T=300, P=0, filenum=0):
    # T= 250 #K
    # P= 0
    val= zfsTP(T, P)
    dfdp=0.01 #GHz/GPa
    print(val)

    cmap = cm.get_cmap('rainbow').copy()
    cmap.set_under('white')
    cmap.set_over('red')
    cmap.set_bad('white')

    fulltickpts, label_list=pxToUm(MFWF)
    # label_list=np.round(label_list,3)
    # label_list=[round(x) for x in label_listraw]
    roitickpts=fulltickpts

    fig, ax=plt.subplots(1,1, figsize=(6,6))
    ax.set_title("Calculated Pressure (GPa)")
    ax.set_xticks(roitickpts)
    ax.set_xticklabels(label_list)
    ax.set_yticks(roitickpts)
    ax.set_yticklabels(label_list)
    ax.set_xlabel("um (Estimated)")
    # ax.plot(MFWF.poiY-MFWF.ylow, MFWF.poiX-MFWF.xlow, 'ro')
    # cmap = cm.get_cmap('viridis').copy()
    # cmap.set_under('white')
    # cmap.set_over('red')
    if Dmin==0 and Dmax==0:
        Dpic=ax.imshow((np.array(MFWF.Dlist[filenum])-val)/dfdp, cmap=cmap)
    else:
        Dpic=ax.imshow((np.array(MFWF.Dlist[filenum])-val)/dfdp,vmin=Dmin, vmax=Dmax, cmap=cmap)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(Dpic, cax=cax)

def BxByBzplots(MFWF, Ilist, poiX, poiY):
    # for WF in MFWF.WFList:
    params=[]
    Bx=[]
    By=[]
    Bz=[]
    D=[]
    for i in np.arange(MFWF.nfiles):
        if Ilist[i]==0:
            pOpt, pCov, chiSq = MFWF.WFList[i].singleESRfit(poiX,poiY, max_peak = 1)
        else:
            pOpt, pCov, chiSq = MFWF.WFList[i].singleESRfit(poiX,poiY, max_peak = 4)
            # print("chiSq is: "+ str(chiSq))
            # WF.myFavoriatePlot(poiX, poiY)
            # print("pOpt:", pOpt)
        if pOpt is not None:
            POIoptListList=pOpt

        Fitfreqs=[]
        WF=MFWF.WFList[i]
        params.append(np.zeros((int(np.floor(len(POIoptListList)/3)), 3)))
        for j in np.arange(int(np.floor(len(POIoptListList)/3))):
            params[i][j,:]= POIoptListList[1+3*j:4+3*j]

        for j in np.arange(int(np.floor(len(POIoptListList)/3))):
            Fitfreqs.append(round(params[i][j,2],3))
        if len(Fitfreqs)!=4:
            while len(Fitfreqs)<4:
                Fitfreqs.append(Fitfreqs[0])
        print(Fitfreqs)
        result=WF.NVESRexactfit(Fitfreqs=Fitfreqs)
        Bx.append(result[0])
        By.append(result[1])
        Bz.append(result[2])
        D.append(result[3])

    #Bx:
    EpOpt,EpCov= opt.curve_fit(linFitZero, MFWF.Ilist[:MFWF.nfiles], Bx)
    Bxpoialpha=EpOpt[0].astype(float)
    BxpoifittedE=linFitZero(MFWF.Ilist[:MFWF.nfiles], *EpOpt)
    #By:
    EpOpt,EpCov= opt.curve_fit(linFitZero, MFWF.Ilist[:MFWF.nfiles], By)
    Bypoialpha=EpOpt[0].astype(float)
    BypoifittedE=linFitZero(MFWF.Ilist[:MFWF.nfiles], *EpOpt)
    #Bz:
    EpOpt,EpCov= opt.curve_fit(linFitZero, MFWF.Ilist[:MFWF.nfiles], Bz)
    Bzpoialpha=EpOpt[0].astype(float)
    BzpoifittedE=linFitZero(MFWF.Ilist[:MFWF.nfiles], *EpOpt)

    fig,ax=plt.subplots(1,3, figsize=(9,3))
    ax[0].set_title("Bx")
    ax[0].set_xlabel("Current (A)")
    ax[0].set_ylabel("Magnetic Field Strength (G)")
    ax[0].set_ylim(0, 50)
    ax[0].plot(Ilist, Bx, "o")
    ax[0].plot(Ilist, BxpoifittedE, "--")
    ax[0].text(0, 40, "slope: " + str(round(Bxpoialpha,3)), fontsize=14, horizontalalignment='left', verticalalignment='center')

    ax[1].set_title("By")
    ax[1].set_xlabel("Current (A)")
    ax[1].set_ylabel("Magnetic Field Strength (G)")
    ax[1].set_ylim(0, 50)
    ax[1].plot(Ilist, By, "o")
    ax[1].plot(Ilist, BypoifittedE, "--")
    ax[1].text(0, 40, "slope: " + str(round(Bypoialpha,3)), fontsize=14, horizontalalignment='left', verticalalignment='center')

    ax[2].set_title("Bz")
    ax[2].set_xlabel("Current (A)")
    ax[2].set_ylabel("Magnetic Field Strength (G)")
    ax[2].set_ylim(0, 100)
    ax[2].plot(Ilist, Bz, "o")
    ax[2].plot(Ilist, BzpoifittedE, "--")
    ax[2].text(0, 80, "slope: " + str(round(Bzpoialpha,3)), fontsize=14, horizontalalignment='left', verticalalignment='center')
    fig.show()

def EvsIDiffDatasetCompare(MFWFList, IlistList, tempList, poiX, poiY,method):
    fig, ax = plt.subplots(1,2, figsize=(8.5, 3.5))
    for val in np.arange(len(MFWFList)):
        # fig, ax = plt.subplots(1,2, figsize=(8.5, 3.5))
        MFWF1=MFWFList[val]
        Ilist1=IlistList[val]

        poiEs1, poialpha1, poiDelta1, poifittedE1, EpCov1=poiProcessDandE(MFWF1, poiX, poiY)
    #   poiEs2, poialpha2, poiDelta2, poifittedE2, EpCov2=poiProcessDandE(MFWF2, poiX2, poiY2)

        print("EpCov1: "+str(np.sqrt(EpCov1)))
        # print("EpCov2: "+str(np.sqrt(EpCov2)))

        gamma=2.8025e-3 #GHz/G
        poiBval1=poialpha1/gamma
        # poiBval2=poialpha2/gamma

        cmap = cm.get_cmap('rainbow').copy()
        cmap.set_over('lightgray')
        cmap.set_under('lightgray')
        cmap.set_bad('white')

        fulltickpts1, label_list=pxToUm(MFWF1)
        roitickpts=fulltickpts1

        # colorlist=cm.rainbow(np.linspace(0,1,len(MFWFList)))
        colorlist=cm.cool(np.linspace(0,1,len(MFWFList)))
        # colorlist=['blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue']
        esrcolorlist=['purple', 'mediumslateblue', 'blue','lightseagreen', 'green', 'lime', 'orange', 'tomato', 'red', 'hotpink', 'magenta']
        # ax[0].set_title("D versus Current (A)")
        # ax[0].plot(Ilist[:len(MFWF.poiDs)], MFWF.poiDs, 'o')
        # # ax[0].set_ylim(3.8, 4.2)

        if method=='linFit':
            # text1=r'$E = \alpha \bullet I + c$'
            text21=r'$\c =$' + str(round(poiDelta1,3)) + "\n" + r'$\alpha =$' + str(round(poialpha1,3)) + "\n" + str(round(poiBval1,3))+ " G/A"
            # text22=r'$\c =$' + str(round(poiDelta2,3)) + "\n" + r'$\alpha =$' + str(round(poialpha2,3)) + "\n" + str(round(poiBval2,3))+ " G/A"
        elif method=='EvsI':
            # text1=r'$E = \sqrt{(\Delta)^2 + (\alpha \bullet I)^2 }$'
            text21=r'$\Delta =$' + str(round(poiDelta1,3)) + "\n" + r'$\alpha =$' + str(round(poialpha1,3)) + "\n" + str(round(poiBval1,3))+ " G/A"
            # text22=r'$\Delta =$' + str(round(poiDelta2,3)) + "\n" + r'$\alpha =$' + str(round(poialpha2,3)) + "\n" + str(round(poiBval2,3))+ " G/A"
        elif method=='linFitZero':
            # text1=r'$E = \alpha \bullet I$'
            text21=r'$\alpha =$' + str(round(poialpha1,3)) + "\n" + str(round(poiBval1,3))+ " G/A"
            # text22=r'$\alpha =$' + str(round(poialpha2,3)) + "\n" + str(round(poiBval2,3))+ " G/A"

        ax[0].set_title("E versus Current (A)")
        ax[0].plot(Ilist1[:len(poiEs1)], poiEs1, 'o', color=colorlist[val], alpha=0.8, label=tempList[val])
        # ax[0].plot(Ilist2[:len(poiEs2)], poiEs2, 'o', color="red", alpha=0.5)
        ax[0].plot(Ilist1[:MFWF1.nfiles], poifittedE1, c=colorlist[val], linestyle="--", alpha=0.4)
        # ax[0].plot(Ilist2[:MFWF2.nfiles], poifittedE2, c='red', linestyle="--", alpha=0.4)
        # ax[0].text(0.35, 0.75, text1+"\n"+text2, fontsize=14, horizontalalignment='center', verticalalignment='center', transform=ax[1].transAxes)
        # ax[0].set_ylim(0,0.35)
        # ax[0].set_xlim(0,4.8)
        ax[0].legend(loc='upper left', bbox_to_anchor=(0.05, 0.95))
        print(text21)
        # print(text22)
        ax[1].set_title("ESR Plots Together")
        for i in np.arange(MFWF1.nfiles):
            # for j in np.arange(nfiles):
            ax[1].plot(MFWF1.fValsList[i], MFWF1.WFList[i].dat[poiX,poiY,:]+i*(0.01), label=str(MFWF1.filenamearr[i]), c=colorlist[val])
            # ax[1].plot(MFWF1.fValsList[i], MFWF1.WFList[i].dat[poiX,poiY,:]+i*(0.01), label=str(MFWF1.filenamearr[i]), c=esrcolorlist[i])
            # ax[2].plot(esrXlist[j], esrFitList[j], label=str(filenamearr[j][:-4])+" Fit", c=colorlist[j], alpha=0.5, linestyle="--")
        # for i in np.arange(MFWF2.nfiles):
        #     # for j in np.arange(nfiles):
        #     ax[1].plot(MFWF2.fValsList[i], MFWF2.POIyVals[i], label=str(MFWF2.filenamearr[i]), c=colorlist[i])
        #     # ax[2].plot(esrXlist[j], esrFitList[j], label=str(filenamearr[j][:-4])+" Fit", c=colorlist[j], alpha=0.5, linestyle="--")

        ax[1].legend(loc='upper left', bbox_to_anchor=(1.05, 0.95))
    plt.show()

# %%

def makeLineCut(startpt, endpt):
    startx=startpt[0]
    starty=startpt[1]
    endx=endpt[0]
    endy=endpt[1]

    if startx<endx:
        xline=np.floor(np.arange(startx, endx, 1))
    else:
        xline=np.floor(np.arange(startx, endx, -1))
    if starty<endy:
        yline=np.floor(np.arange(starty, endy, 1))
    else:
        yline=np.floor(np.arange(starty, endy, -1))

    print("original lengths: "+"x: "+ str(len(xline))+"y: "+ str(len(yline)))
    linelength=max(len(xline),len(yline))
    if len(xline) !=linelength:
        xline=np.linspace(startx, endx, linelength)
    if len(yline) !=linelength:
        yline=np.linspace(starty, endy, linelength)
    xline=np.round(xline[:linelength], 0).astype(int)
    yline=np.round(yline[:linelength], 0).astype(int)

    dist=np.sqrt((endx-startx)**2+(endy-starty)**2)
    distvals=np.linspace(0,dist,len(xline))

    # print(xline)
    # print(yline)
    # print(dist)

    return xline, yline, distvals
    # EvsIcomp(xline, yline)

def esrlinecut(MFWF, mfcount, offset, startpt, endpt, step=1):
    xline, yline, distvals=makeLineCut(startpt, endpt)
    colorlist=cm.cool(np.linspace(0,1,len(xline)))
    xline=xline[::step]
    yline=yline[::step]

    fig, ax=plt.subplots(1,1, figsize=(8,8))
    for i in np.arange(len(xline)):

        yVals=MFWF.WFList[mfcount].dat[xline[i], yline[i],:]+i*offset
        ax.plot(MFWF.WFList[mfcount].fVals, yVals, color=colorlist[i])
        if i==len(xline)-1:
            peakPos=MFWF.WFList[mfcount].singleESRpeakfind(xline[i], yline[i], method='pro', max_peak = 4)
            for j in np.arange(len(peakPos)):
                peak=MFWF.WFList[mfcount].fVals[peakPos][j]
                print(peak)
                ax.axvline(x = peak, color = 'lightgrey', linestyle='--')

def boverhplot(MFWFList, startpt, endpt, alphalow, alphahigh, bhlow, bhhigh, H=None):
    #Line cut processing
    xline, yline, distvals=makeLineCut(startpt, endpt)
    ##Getting H
    if H==None:
        getH()
        # RTalphaList=[]
        # for i in np.arange(rtMFWFload.WFList[0].X):
        #     for j in np.arange(rtMFWFload.WFList[0].Y):
        #         if rtMFWFload.alphamap[i,j]>0.04:
        #             RTalphaList.append(rtMFWFload.alphamap[i,j])
        # RTalphaList.sort()
        # fig,ax=plt.subplots(1,1)
        # ax.hist(RTalphaList, bins=20)
        # ax.set_title("RTalphaList Before Cut")
        # alphaListLow=np.floor(len(RTalphaList)*0.3).astype(int)
        # alphaListHigh=np.floor(len(RTalphaList)*0.8).astype(int)
        # fig,ax=plt.subplots(1,1)
        # rtH=np.mean(RTalphaList[alphaListLow:alphaListHigh])
        # ax.hist(RTalphaList[alphaListLow:alphaListHigh], bins=20)
        # ax.set_title("RTalphaList After Cut")
        # print(RTalphaList[alphaListLow:alphaListHigh])
        # print(rtH)
    else:
        rtH=H
    for mfcount in np.arange(len(MFWFList)):
        ##Getting B/H
        alphaline=[]
        poiEsline=[]
        poifittedEsline=[]
        boverhline=[]
        # for i in np.arange(len(xline)):
        #     poiEs, poialpha, poifittedE=poiProcessDandE(ltMFWFload, xline[i], yline[i])
        #     poiEsline.append(poiEs)
        #     poifittedEsline.append(poifittedE)
        #     alphaline.append(poialpha)
        #     boverhline.append(poialpha/rtH)
        for i in np.arange(len(xline)):
            poialpha=MFWFList[mfcount].alphamap[yline[i],xline[i]]
            alphaline.append(poialpha)
            # print(str(xline[i]) + ", " + str(yline[i]))
            # print("poialpha: "+str(poialpha))
            boverhline.append(poialpha/rtH)

        # #Masking
        # for i in ltMFWFload.xr:
        #     for j in ltMFWFload.yr:
        #         if ltMFWFload.badptmap[i,j]==1 or ltMFWFload.Deltamap[i,j]==1 or ltMFWFload.alphamap[i,j]==1:
        #             ltMFWFload.alphamap[i,j]=np.nan
        # ltMFWFload.alphamap=np.ma.masked_invalid(ltMFWFload.alphamap)

        #Plotting
        fig,ax=plt.subplots(1,2, figsize=(23, 10))

        cmap = cm.get_cmap('rainbow').copy()
        # cmap.set_over('lightgray')
        # cmap.set_under('lightgray')
        # cmap.set_bad('white')

        fulltickpts, label_list=pxToUm(MFWFList[mfcount])
        roitickpts=fulltickpts

        ax[0].set_title("beta Map", fontsize=25)
        ax[0].set_xticks(fulltickpts)
        ax[0].set_xticklabels(label_list)
        ax[0].set_yticks(fulltickpts)
        ax[0].set_yticklabels(label_list)
        ax[0].set_xlabel("um (Estimated)")
        ax[0].scatter(xline, yline, c="white", marker="*", s=200)
        alphaMAP=ax[0].imshow(MFWFList[mfcount].alphamap/gamma, vmin=alphalow, vmax=alphahigh, cmap=cmap, interpolation=None, origin='upper')
        ax[0].set_aspect('equal')
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar=plt.colorbar(alphaMAP, cax=cax)

        ax[1].set_title("B/H Line Cut", fontsize=25)
        ax[1].plot(distvals, boverhline,"o-")
        ax[1].plot(distvals, np.ones(len(distvals)),"--", alpha=0.5)
        ax[1].set_ylim(bhlow, bhhigh)
        ax[1].set_ylabel("B/H")
        ax[1].set_xlabel("Distance (um)")

        # colorlist=['violet', 'mediumorchid','purple', 'mediumslateblue', 'blue', 'dodgerblue','cyan','lightseagreen', 'green', 'limegreen','lime', 'orange', 'tomato', 'red', 'hotpink', 'magenta']
        # fig,ax=plt.subplots(1,1, figsize=(5,5))
        # fig.suptitle("Individual E vs I")
        # for i in np.arange(len(alphaline)):
        #     ax.plot(ltMFWFload.Ilist, poiEsline[i], 'o', c=colorlist[i], label=round(distvals[i],2))
        #     ax.plot(ltMFWFload.Ilist, poifittedEsline[i], c=colorlist[i], linestyle="--", alpha=0.5)
        #     ax.legend(title="Distance", loc='upper left', bbox_to_anchor=(1.05, 0.95))

