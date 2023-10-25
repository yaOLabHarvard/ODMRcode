# Things to improve in future.
# 1. readESRfile returns a zero vector for errVals, have to change that
from os.path import isfile
from os.path import join
from os import listdir
import os
import matplotlib.pyplot as plt
import scipy.optimize as opt
import numpy as np
import pandas as pd
#from colorama import Fore
import matplotlib.ticker as ticker
from matplotlib import gridspec
from matplotlib.patches import Rectangle
from scipy.io import loadmat
# import plotly.graph_objects as go


data_path=r"D:\Data"  # the r in front is convert a normal string to a raw string
zfs= lambda T: (2.8771+ -4.625e-6*T+ 1.067e-7*T*T+ -9.325e-10*T*T*T+ 1.739e-12*T*T*T*T+ -1.838e-15*T*T*T*T*T)
#																											HELPER FUNCTIONS

# gaussian fit functions
def gaussian(xVals,A,sigma,x0):
    '''
    Returns a gaussian for the fit function
    https://terpconnect.umd.edu/~toh/spectrum/CurveFitting.html
    '''
    return A*np.exp(-(xVals-x0)**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))

def gauss_fit(xVals,bckgnd,*args):
    '''
    Fitting function for arbitrary number of gaussian peaks
    bckgnd - background value
    *args contains the parameters for the fits in sets of three. In each set:
    - Amplitude (A) corresponds to the first number
    - width (sigma) corresponds to the second number
    - center (x0) corresponds to the thrid number
    '''
    yVals= np.zeros_like(xVals)
    for i in range(0, len(args), 3):
        A= args[i]
        sigma= args[i+1]
        x0= args[i+2]
        yVals+= gaussian(xVals,A,sigma,x0)
    yVals=yVals+bckgnd
    return yVals

def gaussian_c(xVals,C,FWHM,x0):
    sigma= FWHM/2.355
    return C*np.exp(-(xVals-x0)**2/(2*sigma**2))

def gauss_fit_c(xVals,bckgnd,*args):
    '''
    Fitting function for arbitrary number of gaussian peaks
    bckgnd - background value
    *args contains the parameters for the fits in sets of three. In each set:
    - Amplitude (A) corresponds to the first number
    - width (sigma) corresponds to the second number
    - center (x0) corresponds to the thrid number
    '''
    yVals= np.zeros_like(xVals)
    for i in range(0, len(args), 3):
        A= args[i]
        FWHM= args[i+1]
        x0= args[i+2]
        yVals+= gaussian_c(xVals,A,FWHM,x0)
    yVals=yVals+bckgnd
    return yVals

def gauss_fit_bkg(xVals,slope,intercept,*args):
    '''
    Fitting function for arbitrary number of gaussian peaks with linear background
    slope - slope of background value
    intercept - intercept of background value
    *args contains the parameters for the fits in sets of three. In each set:
    - Amplitude (A) corresponds to the first number
    - width (sigma) corresponds to the second number
    - center (x0) corresponds to the thrid number
    '''
    yVals= np.zeros_like(xVals)
    for i in range(0, len(args), 3):
        A= args[i]
        sigma= args[i+1]
        x0= args[i+2]
        yVals+= gaussian(xVals,A,sigma,x0)
    yVals=yVals+slope*xVals+intercept
    return yVals

# lorentzian fit functions
def lorentzian(xVals,A,gamma,x0):
    '''
    Returns a gaussian for the fit function
    '''
    return A/np.pi*((gamma/2.)/((xVals-x0)**2+(gamma/2.)**2))

def lor_fit(xVals,bckgnd,*args):
    '''
    Fitting function for arbitrary number of lorentzian peaks
    bckgnd - background value
    *args contains the parameters for the fits in sets of three. In each set:
    - Amplitude (A) corresponds to the first number
    - width (gamma) corresponds to the second number
    - center (x0) corresponds to the thrid number
    '''
    yVals= np.zeros_like(xVals)
    for i in range(0, len(args), 3):
        A= args[i]
        gamma= args[i+1]
        x0= args[i+2]
        yVals+= lorentzian(xVals,A,gamma,x0)
    yVals= yVals+bckgnd
    return yVals

def lor_fit_bkg(xVals,slope,intercept,*args):
    '''
    Fitting function for arbitrary number of lorentzian peaks
    bckgnd - background value
    *args contains the parameters for the fits in sets of three. In each set:
    - Amplitude (A) corresponds to the first number
    - width (gamma) corresponds to the second number
    - center (x0) corresponds to the thrid number
    '''
    yVals= np.zeros_like(xVals)
    for i in range(0, len(args), 3):
        A= args[i]
        gamma= args[i+1]
        x0= args[i+2]
        yVals+= lorentzian(xVals,A,gamma,x0)
    yVals= yVals+slope*xVals+intercept
    return yVals

##########################################	EXTRACT FILENAMES

def extractScanFiles(date, makePlots= False):
    '''
    scan_files= extractScanFiles(date)

    Returns the list of scan files collected on a particular date
    date is a string containing the corresponding to a particular date in data_path
    '''
    folderpath= join(data_path,date)
    scan_files= [join(folderpath,f) for f in listdir(folderpath) if (isfile(join(folderpath,f)) and 'Image' in f and '.txt' in f)]
    if makePlots:
        for f in scan_files:
            plotScanFile(f)
    return scan_files

def extractESRFiles(date, makePlots= False):
    '''
    ESR_files= extractESRFiles(date)
    Returns the list of ESR files collected on a particular date
    date is a string containing the corresponding to a particular date in data_path
    '''
    folderpath= join(data_path,date)
    ESR_files= [join(folderpath,f) for f in listdir(folderpath) if (isfile(join(folderpath,f)) and 'ESR' in f and '.txt' in f and not('BackUpData' in f or 'fit_params' in f or 'Log' in f or 'full' in f))]
    if makePlots:
        for f in ESR_files:
            plt.figure()
            plotFile(f, plotErrors= False)
    plt.close()
    return ESR_files

def extractODMRFiles(date, excludeRounds= True, makePlots= False):
    '''
    ODMR_files= extractODMRFiles(date,excludeRounds= True)

    Returns the list of ODMR files collected on a particular date
    date is a string containing the corresponding to a particular date in data_path
    '''
    folderpath= join(data_path,date)
    if excludeRounds:
        ODMR_files= [join(folderpath,f) for f in listdir(folderpath) if (isfile(join(folderpath,f)) and 'ODMR' in f
                    and '.txt' in f and not('BackUpData' in f or 'fit_params' in f or 'Round' in f or 'Log' in f))]
    else:
        ODMR_files= [join(folderpath,f) for f in listdir(folderpath) if (isfile(join(folderpath,f)) and 'ODMR' in f
                        and '.txt' in f and not('BackUpData' in f or 'fit_params' in f or 'Log' in f))]
    if makePlots:
        for f in ODMR_files:
            plt.figure()
            plotFile(f,plotErrors=False)
    plt.close()
    return ODMR_files

def extractTrackFiles(date):
    '''
    track_files= extractTrackFiles(date,excludeRounds= True)

    Returns the list of Track files collected on a particular date
    date is a string containing the corresponding to a particular date in data_path
    '''
    folderpath= join(data_path)
    track_files= [join(folderpath,f) for f in listdir(folderpath) if (isfile(join(folderpath,f))
                                                                      and 'TrackLog' in f and '.txt' in f)]
    return track_files

def extractPLFiles(date):
    '''
    pl_files= extractPLFiles(date,excludeRounds= True)

    Returns the list of PL collected on a particular date
    date is a string containing the corresponding to a particular date in data_path
    '''
    folderpath= join(data_path,'spectrometer')
    if(float(date.replace('-',''))<20190725):
        folderpath= join(folderpath,date)
        filenames= [join(folderpath,f) for f in listdir(join(folderpath)) if (isfile(join(folderpath,f)) and ('.csv' in f) and not('.npz' in f))]
    else:
        filenames= [join(folderpath,f) for f in listdir(join(folderpath)) if (isfile(join(folderpath,f)) and ('.csv' in f) and (date in f) and not('.npz' in f))]
    
    return filenames

def extractRabiFiles(date,makePlots=False):
    '''
    rabi_files= extractRabiFiles(date,excludeRounds= True)

    Returns the list of Rabi files collected on a particular date
    date is a string containing the corresponding to a particular date in data_path
    '''
    folderpath= join(data_path,date)
    rabi_files= [join(folderpath,f) for f in listdir(folderpath) if (isfile(join(folderpath,f)) and 'Rabi' in f
                    and '.txt' in f and not('BackUpData' in f or 'fit_params' in f or 'Round' in f or 'Log' in f))]
    if makePlots:
        for f in rabi_files:
            plt.figure()
            plotFile(f,plotErrors=False)
    plt.close()
    return rabi_files


######################################		FILE READING AND PLOTTING FUNCTIONS

def readSpectrometerFile(filename):
    '''
    Wavelength,Intensity= readSpectrometerFile(filename)

    Reads a csv file generated by LightField
    '''
    dat= pd.read_csv(filename,header= 0)
    return dat['Wavelength'].values, dat['Intensity'].values

def readScanFile(filename):
    '''
    scan,xLims,yLims= readScanFile(filename)

    Reads a image file saved by ImageNVC and returns a scan as a matrix
    '''
    datFile= open(filename,'r')
    datFileIterator= iter(datFile)
    scan = []
    for line in datFileIterator:
        line= line.replace('\n','').strip()
        if 'VxRange' in line:
            arr= line.split('[um]: ')[-1].split(' NVx')[0]
            xLims= [float(x) for x in arr[1:-1].split(' ')]
        elif 'VyRange' in line:
            arr= line.split('[um]: ')[-1].split(' NVy')[0]
            yLims= [float(y) for y in arr[1:-1].split(' ')]
        elif 'VzRange' in line:
            continue
        elif 'DTRange' in line:
            continue
        elif 'Size' in line:
            size= line.split(':')[-1].strip()[1:-1].split(' ')
            x= int(size[0])
            y= int(size[1])
        else:
            try:
                scan.append(float(line))
            except ValueError:
                break
    scan = np.reshape(scan, (x,y))
    return scan,xLims,yLims

def readESRFile(filename):
    '''
    freqVals,sigVals,errVals= readESRFile(filename)

    Have to improve this: Can get raw data from BackUpData file to better estimate the errors etc

    Reads and parses an ESR file and returns the freqency and the signal values
    as numpy arrays
    '''
    datFrame= pd.read_csv(filename,sep=' ',skiprows= [0,1,2], names= ['frequency','signal'], comment= 'X').dropna().astype(float)
    freqVals= datFrame['frequency'].values/1e9
    sigVals= datFrame['signal'].values
    errVals= np.sqrt(sigVals)
    return freqVals, sigVals, errVals

def readRawRabiFile(filename):
    '''
    tauVals,sigVals,refVals,ref2Vals= readRawRabiFile(filename)

    Reads and parses a Rabi file and returns the tau, signal values and reference values
    as numpy arrays
    '''
    datFrame= pd.read_csv(filename,header= None,skiprows=[0,1,2],sep= ' ',comment= 'X',names= ['time','ref','signal']).dropna(axis=0).astype(float)
    tauVals= datFrame['time'].values/1e9
    sigVals= datFrame['signal'].values
    refVals= datFrame['ref'].values
    return tauVals,sigVals,refVals

def readRabiFile(filename):
    '''
	tauVals,yVals,errVals= readESRFile(filename)

    Have to improve this: Can get raw data from BackUpData file to better estimate the errors etc

    Reads and parses an ESR file and returns the freqency and the signal values
    as numpy arrays
    '''
    datFrame= pd.read_csv(filename,header= None,skiprows=[0,1,2],sep= ' ',comment= 'X',names= ['time','ref','signal']).dropna(axis=0).astype(float)
    tauVals= datFrame['time'].values/1e9
    sigVals= datFrame['signal'].values
    refVals= datFrame['ref'].values
    yVals= sigVals/refVals
    errVals= yVals*np.sqrt(1/sigVals+1/refVals)
    return tauVals, yVals, errVals

def readRawODMRFile(filename):
    '''
    freqVals,sigVals,refVals= readRawODMRFile(filename)

    Reads an ODMR file and returns the raw data (counts) for both signal and reference
    '''
    datFrame= pd.read_csv(filename,header= None,skiprows=[0,1,2],sep= ' ',comment= 'X',names= ['frequency','ref','signal']).dropna(axis=0).astype(float)
    freqVals= datFrame['frequency'].values/1e9
    sigVals= datFrame['signal'].values
    refVals= datFrame['ref'].values
    return freqVals,sigVals,refVals

def readODMRFile(filename):
    '''
    freqVals,yVals,errVals= readODMRFile(filename)

    Reads an ODMR file and returns the frequency and processed signal values and error values
    TO DO IN FUTURE:
    read the pi pulse length and return the rabi frequency, return the MW power
    '''
    datFrame= pd.read_csv(filename,header= None,skiprows=[0,1,2],sep= ' ',comment= 'X',names= ['frequency','ref','signal']).dropna(axis=0).astype(float)
    freqVals= datFrame['frequency'].values/1e9
    sigVals= datFrame['signal'].values
    refVals= datFrame['ref'].values
    yVals= sigVals/refVals
    errVals= yVals*np.sqrt(1/sigVals+1/refVals)
    return freqVals,yVals,errVals

def readFile(filename):
    '''
    xVals,yVals,errVals= readFile(filename)

    Calls the appropriate read file function (ESR/ODMR)
    '''
    if 'ESR' in filename:
        xVals,yVals,errVals= readESRFile(filename)
    elif 'ODMR' in filename:
        xVals,yVals,errVals= readODMRFile(filename)
    elif 'Rabi' in filename:
        xVals,yVals,errVals= readRabiFile(filename)
    else:
        print('Not a valid filename passed')
        return None
    return xVals,yVals,errVals

def plotFile(filename,plotErrors= True):
    '''
    freqVals,yVals,errVals= plot_file(filename)

    Calls either readESRFile or readODMRFile and plots the data
    ESR data is normalized
    Returns the output of the called function
    '''
    plt.figure(figsize= (8,5))
    if 'ESR' in filename:
        xVals,yVals,errVals= readESRFile(filename)
        yVals/= np.max(yVals)
        plt.xlabel('Frequency (GHz)',fontsize=15)
        plt.ylabel('Contrast',fontsize=15)
    elif 'ODMR' in filename:
        xVals,yVals,errVals= readODMRFile(filename)
        plt.xlabel('Frequency (GHz)',fontsize=15)
        plt.ylabel('Contrast',fontsize=15)
    elif 'Rabi' in filename:
        xVals,yVals,errVals= readRabiFile(filename)
        xVals= xVals*1e9 # to convert the number into nanoseconds
        plt.xlabel('Time (ns)',fontsize=15)
        plt.ylabel('Contrast',fontsize=15)
    else:
        print('Not a valid filename passed')
        return
    if plotErrors:
        plt.errorbar(xVals,yVals,yerr=errVals)
    else:
        plt.plot(xVals,yVals)
    ax= plt.gca()
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
    plt.xticks(fontsize=15)
    plt.yticks(fontsize= 15)
    plt.title(filename)
    plt.show()
    return xVals,yVals,errVals

def plotScanFile(filename):
    '''
    scan= plotScanFile(filename)
    '''
    scan,xLims,yLims= readScanFile(filename)
    plt.figure(figsize=(7,7))
    if xLims and yLims:
        plt.imshow(scan,aspect= 'equal', extent= xLims+yLims)
    else:
        plt.imshow(scan,aspect= 'equal')
    plt.title(filename)
    plt.xlabel('$\mu m$',fontsize= 15)
    plt.ylabel('$\mu m$',fontsize= 15)
    plt.colorbar()
    plt.show()
    plt.close()
    return scan

def getFitVals(filename,fit_function=lor_fit):
    '''
    xVals,yVals= getFitVals(filename)
    '''
    xVals,yVals,errVals= readFile(filename)
    pOpt,pCov,rVal= readFitFile(filename.replace('.txt','_fit_params.npz'))
    pltX= np.linspace(np.min(xVals),np.max(xVals),len(xVals)*100)
    pltFit= fit_function(pltX,*pOpt)
    return pltX,pltFit

def plotRawODMRFile(filename):
    '''
    freqVals,sigVals,refVals= plotRawODMRFile(filename)

    Plots the signal and reference counts for the ODMR file and returns the frequency, signal and refernce counts
    '''
    freqVals,sigVals,refVals= readRawODMRFile(filename)
    plt.figure()
    plt.errorbar(freqVals,sigVals,yerr=1/np.sqrt(sigVals),label='signal')
    plt.errorbar(freqVals,refVals,yerr=1/np.sqrt(refVals),label='reference')
    plt.title(filename)
    plt.xlabel('Frequency')
    plt.ylabel('Counts')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    return freqVals,sigVals,refVals

def readTrackFile(filename):
    '''
    xVals,yVals,zVals,countVals= readTrackFile(filename)

    Returns a pandas DataFrame with columns xVals,yVals,zVals,countVals
    containing the x,y,z positions and counts respectively for everytrack during the
    process of the measurement that the track files corresponds to
    '''
    if not('TrackLog' in filename):
        print('Not a valid track file')
        return
    dat= np.loadtxt(filename)
    xTrackIndices= np.arange(0,len(dat),4,dtype=int)
    return dat[xTrackIndices],dat[xTrackIndices+1],dat[xTrackIndices+2],dat[xTrackIndices+3]

def plotTrackFile(filename):
    '''
    No return values

    Plots the track file
    '''
    xVals,yVals,zVals,countVals= readTrackFile(filename)
    numTracks= len(xVals)
    indexVals= np.arange(numTracks)
    fig,ax= plt.subplots(2,2,figsize=(15,10))
    ax[0,0].plot(xVals,'ro')
    ax[0,0].set_title('X Tracking Values')
    ax[0,1].plot(yVals,'bo')
    ax[0,1].set_title('Y Tracking Values')
    ax[1,0].plot(zVals,'go')
    ax[1,0].set_title('Z Tracking Values')
    ax[1,1].plot(countVals,'ko')
    ax[1,1].set_title('Count Tracking Values')
    fig.suptitle(filename)
    plt.show()

def readFitFile(filename):
    '''
    pOpt,pCov,rVal= readFitFile(filename)
    Filename has to be a .npz file corresponding to the data file for which the fits are demanded
    Function to parse the params data file and return
    pOpt - Optimal parameters for the fit
    pCov - Covariance matrix
    rVal - r-squared value of the fit
    '''
    if '.txt' in filename:
        filename= filename.replace('.txt','_fit_params.npz')
    paramsFile= np.load(filename)
    return paramsFile['pOpt'],paramsFile['pCov'],paramsFile['rVal']

def combineFiles(filenames):
    '''
    xVals,yVals,errVals= combineFiles(filenames)

    Combines data from different files
    '''
    for i,f in enumerate(filenames):
        if i==0:
            xVals,yVals,errVals= readFile(f)
            if not(errVals is None):
                errVals= errVals**2
            continue
        x,y,e= readFile(f)
        xVals+= x
        yVals+= y
        if not(e is None):
            errVals+= e**2
    xVals/= len(filenames)
    yVals/= len(filenames)
    errVals/= len(filenames)
    errVals= np.sqrt(errVals)
    return xVals,yVals,errVals

#################################################			FITTING FUNCTION

def make_plotly(filename):
	'''
	make_plotly(filename)

	Makes interactive plotly fidget during fitting process
	'''
	xVals,yVals,errVals= readFile(filename)
	yVals/= np.max(yVals)
	f= go.FigureWidget([go.Scatter(x= xVals, y= yVals)])
	f.layout.xaxis.title= "Frequency [GHz]"
	f.layout.yaxis.title= "Contrast"
	f.layout.title= filename
	f.show()
	return xVals, yVals, errVals

def fit_file(filename,fit_function=lor_fit, saveData= True, spectrometer_file= False):
    '''
    pOpt,pCov,rVal= fit_ODMR_file(filename,fit_function=lor_fit)

    You have to be inside the circle to know how to use this
    '''
    # input message and result printing tempate
    input_messege= 'Enter initial parameters separated by comma (leave blank and press enter if you want to skip this fit):\nBackground,A_peak1,sigma_peak1,x0_peak1,... \n'
    template_bkg= 'Data Filename: %s\nR-squared value: %s\nCenter Frequencies: %s\nShifts (MHz): %s\nSplittings (MHz): %s\nLinewdiths (MHz): %s\nPercentage Contrast: %s\nBackground Slope:%s\nBackground Intercept:%s'
    template= 'Data Filename: %s\nR-squared value: %s\nCenter Frequencies: %s\nShifts (MHz): %s\nSplittings (MHz): %s\nLinewdiths (MHz): %s\nPercentage Contrast: %s\nBackground:%s'

    # getting init params from user and making fit
    if spectrometer_file:
        xVals,yVals= readSpectrometerFile(filename)
        errVals= None
        plt.figure()
        plt.plot(xVals,yVals)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity')
        plt.title(filename)
        plt.show()
    else:
        xVals, yVals, errVals= make_plotly(filename)
    init_params= np.fromstring(input(input_messege),sep=',')
    print(init_params)
    if len(init_params)==0:
        print('Fitting aborted')
        return None, None, None
    elif fit_function is lor_fit:
        print('Fitting to Lorentzian')
        lin_bkg= False
    elif fit_function is gauss_fit:
        print('Fitting to Gaussian')
        lin_bkg= False
    elif fit_function is gauss_fit_bkg:
        print('Fitting to Gaussian with linear background')
        lin_bkg= True
    elif fit_function is lor_fit_bkg:
        print('Fitting to Lorentzian with linear background')
        lin_bkg= True

    #init_params= compute_init_params(init_params, fit_function)

    pOpt,pCov= opt.curve_fit(fit_function,xVals,yVals,p0=init_params)
    fitVals= fit_function(xVals,*pOpt)

    # plotting data and fits
    fig,ax= plt.subplots(nrows=2,ncols=1,figsize= (8,8))
    #plt.tight_layout()
    pltX= np.linspace(np.min(xVals),np.max(xVals),len(xVals)*5)
    pltFit= fit_function(pltX,*pOpt)
    ax[0].errorbar(xVals,yVals, label= 'data')
    ax[0].plot(pltX,pltFit,label = 'fit')
    ax[0].legend(loc=3)
    ax[0].set_ylabel('Intensity')
    residuals= yVals-fit_function(xVals,*pOpt)
    ax[1].plot(xVals,residuals)
    ax[1].set_xlabel('Frequency (GHz)')
    ax[1].set_ylabel('Data - Fit')
    ax[0].set_title(filename)
    plotFilename= filename.replace('.txt','.png').replace('.csv','.png')
    fig.savefig(plotFilename)
    print('Fits plotted in {0}'.format(plotFilename))

    # saving the fits
    rVal= compute_rVal(xVals,yVals,fitVals)
    if saveData:
        saveFilename= filename.replace('.txt','_fit_params.npz').replace('.csv','_fit_params.npz')
        np.savez_compressed(saveFilename,pOpt=pOpt,pCov=pCov,rVal=rVal)
        print('Fits saved to {0}'.format(saveFilename))
        fitReturns= [] # used in the next step
    else:
        saveFilename= None
        fitReturns= [pOpt, pCov, rVal] # used in the next step

    # if the fit is a standard gauss fit or lorentzian fit then the following data is printed
    if fit_function is lor_fit or fit_function is gauss_fit:
        resFreq,shift,split,linewidth,rVal,amplitude,bckgnd= computeShifts(saveFilename,*fitReturns)
        resFreqErr,linewidthErr,amplitudeErr,bckgndErr= getFittingErrors(saveFilename,*fitReturns)

        if fit_function is lor_fit:
            contrast= 1e2*(np.abs(amplitude)*2/(np.pi*linewidth/1e3))/bckgnd
            contrastErr= contrast*np.sqrt((amplitudeErr/amplitude)**2+(linewidthErr/linewidth)**2+(bckgndErr/bckgnd)**2)
        elif fit_function is gauss_fit:
            contrast= 1e2*np.abs(amplitude)/np.abs(linewidth/1e3*np.sqrt(2*np.pi))/bckgnd
            contrastErr= contrast*np.sqrt((amplitudeErr/amplitude)**2+(linewidthErr/linewidth)**2+(bckgndErr/bckgnd)**2)
            linewidth= 2.355*linewidth # FWHM for gaussians https://en.wikipedia.org/wiki/Full_width_at_half_maximum
        datSet= saveFilename.split('\\')[-1]
        print(Fore.RED)
        print('Center frequency: %1.4f GHz'%np.mean(resFreq))
        print('Fitting function:%s'%fit_function)
        print(template%(datSet,rVal,str(resFreq),shift,split,str(linewidth),str(contrast),str(bckgnd)))
        print('Contrast Error: %s'%(contrastErr))
        print(Fore.RESET)
    elif fit_function is lor_fit_bkg or fit_function is gauss_fit_bkg:
        resFreq,shift,split,linewidth,rVal,amplitude,slope,intercept= computeShifts_lin_bkg(saveFilename,*fitReturns)
        if fit_function is lor_fit_bkg:
            contrast= 1e2*(np.abs(amplitude)*2/(np.pi*linewidth/1e3))/(slope*resFreq+intercept)
        elif fit_function is gauss_fit_bkg:
            contrast= 1e2*np.abs(amplitude)/np.abs(linewidth/1e3*np.sqrt(2*np.pi))/(slope*resFreq+intercept)
            linewidth= 2.355*linewidth # FWHM for gaussians https://en.wikipedia.org/wiki/Full_width_at_half_maximum
        datSet= saveFilename.split('\\')[-1]
        print(Fore.RED)
        print('Fitting function:%s'%fit_function)
        print(template%(datSet,rVal,str(resFreq),shift,split,str(linewidth),str(contrast),str(slope),str(intercept)))
        #print('Contrast Error: %s'%(contrastErr))
        print(Fore.RESET)



    return pOpt,pCov,rVal

def computeShifts(filename, *args):
    '''
    resFreq,shift,split,linewidth,rVal,amplitude,bckgnd= computeShifts(filename)

    Filename has to be a .npz file corresponding to the data for which the fit is demanded
    Given a filename this function retrieves the params file and from the optimal parameters
    it computes the number of peaks, shiftings and splitting

    Linewidth, shifting and splitting are returned in MHz
    resFreq is returned in GHz
    Linewith corresponds to FWHM for lorentzian and sigma for gaussians
    '''
    if filename:
        pOpt,pCov,rVal= np.array(readFitFile(filename))
    else:
        pOpt= args[0]
        pCov= args[1]
        rVal= args[2]

    numPeaks= (len(pOpt)-1)/3
    bckgnd= pOpt[0]
    amplitudeIndices= np.arange(0,3*numPeaks,3,dtype='int')+1
    linewidthIndices= amplitudeIndices+1
    freqIndices= linewidthIndices+1

    linewidth= 1e3*pOpt[linewidthIndices]
    resFreq= pOpt[freqIndices]
    amplitude= pOpt[amplitudeIndices]

    if numPeaks==1:
        print('1 peak')
        split= 'NaN \t'
        shift= '%1.3f '%(1e3*(resFreq[0]-2.87))
    else:
        print('%i peaks'%(len(resFreq)))
        split= ['%1.3f'%(val*1e3) for val in np.diff(resFreq)]
        shift= '%1.3f'%((np.mean(resFreq)-2.87)*1e3)

    return resFreq,shift,split,linewidth,'%1.3f'%(rVal),amplitude,bckgnd

def computeShifts_lin_bkg(filename, *args):
    '''
    resFreq,shift,split,linewidth,rVal,amplitude,slope,intercept= computeShifts_lin_bkg(filename)

    Filename has to be a .npz file corresponding to the data for which the fit is demanded
    Given a filename this function retrieves the params file and from the optimal parameters
    it computes the number of peaks, shiftings and splitting

    Linewidth, shifting and splitting are returned in MHz
    resFreq is returned in GHz
    Linewith corresponds to FWHM for lorentzian and sigma for gaussians
    '''
    if filename:
        pOpt,pCov,rVal= np.array(readFitFile(filename))
    else:
        pOpt= args[0]
        pCov= args[1]
        rVal= args[2]

    numPeaks= (len(pOpt)-2)/3
    slope= pOpt[0]
    intercept= pOpt[1]
    amplitudeIndices= np.arange(0,3*numPeaks,3,dtype='int')+2
    linewidthIndices= amplitudeIndices+1
    freqIndices= linewidthIndices+1

    linewidth= 1e3*pOpt[linewidthIndices]
    resFreq= pOpt[freqIndices]
    amplitude= pOpt[amplitudeIndices]

    if numPeaks==1:
        print('1 peak')
        split= 'NaN \t'
        shift= '%1.3f '%(1e3*(resFreq[0]-2.87))
    else:
        print('%i peaks'%(len(resFreq)))
        split= ['%1.3f'%(val*1e3) for val in np.diff(resFreq)]
        shift= '%1.3f'%((np.mean(resFreq)-2.87)*1e3)

    return resFreq,shift,split,linewidth,'%1.3f'%(rVal),amplitude,slope,intercept

def getFittingErrors(filename, *args):
    '''
    resFreqErr,linewidthErr,amplitudeErr,bckgndErr= getFittingErrors(filename)

    Returns the errors for the fit parameters
    The fileanmae has to be a .npz file corresponding to the data for which the fit is demanded
    Given a filename this function retrieves the parasm filea and from the covariance matrix determines
    the parameter errors
    '''
    if filename:
        pOpt,pCov,rVal= np.array(readFitFile(filename))
    else:
        pOpt= args[0]
        pCov= args[1]
        rVal= args[2]

    pErr= np.sqrt(np.diag(pCov))
    numPeaks= (len(pErr)-1)/3;
    bckgndErr= pErr[0]
    amplitudeIndices= np.arange(0,3*numPeaks,3,dtype='int')+1
    linewidthIndices= amplitudeIndices+1
    freqIndices= linewidthIndices+1

    linewidthErr= 1e3*pErr[linewidthIndices]
    resFreqErr= pErr[freqIndices]
    amplitudeErr= pErr[amplitudeIndices]

    return resFreqErr,linewidthErr,amplitudeErr,bckgndErr

def compute_rVal(xVals,yVals,fitVals):
    '''
    rVal= fit_stats(xVals,yVals,fitVals)
    rVal= fit_stats(xVals,yVals,fitVals)

    R squared value computed according to wikipedia
    https://en.wikipedia.org/wiki/Coefficient_of_determination
    '''
    SSres= np.sum((yVals-fitVals)**2)
    SStot= np.sum((yVals-np.mean(yVals))**2)
    rVal= 1. - SSres/SStot
    return rVal

def compute_init_params(params,fit_function= lor_fit):
    '''
    Returns contrast computed from amplitudes of normalized functions
    '''
    if not(fit_function is lor_fit) or not(fit_function is gauss_fit):
        return params
    num_peaks= len(params)-1
    bckgndErr= params[0]
    contrastIndices= np.arange(0,3*numPeaks,3,dtype='int')+1
    linewidthIndices= amplitudeIndices+1
    freqIndices= linewidthIndices+1

    contrast= params[contrastIndices]
    linewidth= params[linewidthIndices]
    resFreq= params[freqIndices]

    if fit_function is lor_fit:
        amplitude= (np.abs(contrast)*np.pi*linewidth/2)
    elif fit_function is gauss_fit:
        linewidth= linewidth/2.355
        amplitude= np.abs(contrast)*np.abs(linewidth*np.sqrt(2*np.pi))
    params[contrastIndices]= amplitude
    params[linewidthIndices]= linewidth
    return params

def normalizeESR(yVals):
	'''
	yVals= normalizeESR(yVals)
	Normalizes ESR plot by averaging the 10 maximum values
	DO NOT USE FOR PLOT WITH WEIRD HEATING PEAKS
	'''
	yVals= yVals/np.mean(np.sort(yVals)[-10:])
	#yVals= yVals/np.trapz(1-yVals)
	return yVals

def rescaleESR(yVals):
	'''
	yVals= rescaleESR(yVals)

	Rescales ESR plot make 'Area under curve'==1
	'''
	a= np.mean(np.sort(yVals)[-10:])
	b= np.mean(np.sort(yVals)[:11])
	yVals= (yVals-b)/(a-b)
	return yVals

################################# WIDEFIELD FUNCTIONS

def read_matfile(filename, normalize= True, avg= True, Nx= 4, Ny= 4, Nf= 4, norm_points= 10, normal= True):
    gWide= loadmat(filename)
    gWide= gWide['gWide'][0,0]
    if normal: # usually normal; for some data files these indices don't work (not sure why)
        xFrom= np.squeeze(gWide[1])
        xTo= np.squeeze(gWide[2])
        fVals= np.squeeze(gWide[4])*1e-9
        dat= np.swapaxes(np.squeeze(gWide[-1]),0,1)
    else:
        xFrom= np.squeeze(gWide[4])
        xTo= np.squeeze(gWide[5])
        fVals= np.squeeze(gWide[7])*1e-9
        dat= np.swapaxes(np.squeeze(gWide[-1]),0,1)

    if avg:
        for axis,N in zip([0,1,2],[Nx, Ny, Nf]):
            dat= running_mean_nd(dat, N, axis)
        fVals= running_mean(fVals, Nf)

    if normalize:
        dat= normalize_widefield(dat, norm_points)
    print(dat.shape)
    X,Y,npoints= dat.shape
    return fVals, dat, xFrom, xTo, X, Y, npoints

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def running_mean_nd(x, N, axis= 0):
    len_axis= x.shape[axis]
    cumsum = np.cumsum(np.insert(x, 0, 0,axis= axis),axis= axis)
    result= (np.take(cumsum,np.arange(N,len_axis+1),axis= axis) - np.take(cumsum,np.arange(0,len_axis+1-N),axis= axis))/float(N)
    return result

def plot_row(img,freqVals,row,width_ratios=[1,5],normalized= False):
    if not(normalized):
        img_n= normalize_widefield(img)
    else:
        img_n= img
    x,y,f= img.shape
    ylims= [y,0]

    fig = plt.figure(figsize=(18,5))
    gs = gridspec.GridSpec(1, 2, width_ratios= width_ratios)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    img_slice= np.copy(np.squeeze(img[:,:,100]))
    img_slice[row,:]= np.nan
    ax0.imshow(img_slice)
    ax0.tick_params(axis= 'both', labelsize= 10)
    ax0.set_xlabel('Pixel', fontsize= 10)
    ax0.set_ylabel('Pixel', fontsize= 10)
    ax0.set_title('Representative Image',fontsize=12);

    wtrfl= np.squeeze(img_n[row,:,:])
    ax1.imshow(wtrfl,extent= np.append(freqVals[[0,-1]]*1e3,ylims))
    ax1.tick_params(labelsize= 10)
    ax1.set_xlabel('Frequency [MHz]',fontsize= 10)
    ax1.set_ylabel('$\mu m$',fontsize= 10)
    ax1.set_title('Row %i Waterfall'%row,fontsize= 12)

def plot_col(img,freqVals,col,width_ratios= [1,4],normalized= False):
    if not(normalized):
        img_n= normalize_widefield(img)
    else:
        img_n= img
    x,y,f= img.shape
    xlims= [x,0]

    fig = plt.figure(figsize=(18,5))
    gs = gridspec.GridSpec(1, 2, width_ratios=width_ratios)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    img_slice= np.copy(np.squeeze(img[:,:,20]))
    img_slice[:,col]= np.nan
    ax0.imshow(img_slice)
    ax0.tick_params(axis= 'both', labelsize= 10)
    ax0.set_xlabel('Pixel', fontsize= 10)
    ax0.set_ylabel('Pixel', fontsize= 10)

    wtrfl= np.squeeze(img_n[:,col,:])
    ax1.imshow(wtrfl,extent= np.append(freqVals[[0,-1]]*1e3,xlims))
    ax1.tick_params(labelsize= 10)
    ax1.set_xlabel('Frequency [MHz]',fontsize= 10)
    ax1.set_ylabel('$\mu m$',fontsize= 10)
    ax1.set_title('Col %i Waterfall'%col,fontsize= 12)

def normalize_widefield(img_raw, numpoints= 10, from_back= True):
    """
    img_n= normalize_widefield(img_raw)

    Returns widefield dataset with normalized contrast values based on the last 10 y data points
    """
    x,y,f= img_raw.shape
    for i in range(x):
        for j in range(y):
            if from_back:
                img_raw[i,j,:]= img_raw[i,j,:]/np.nanmean(img_raw[i,j,-numpoints:])
            else:
                img_raw[i,j,:]= img_raw[i,j,:]/np.nanmean(img_raw[i,j,:numpoints])
    return img_raw

def widefield_apodization(img_raw, freq, center_freq = 2.870, filter_para = 200):
    """
    img_n= widefield_apodization(img_raw)

    Returns widefield dataset with exp(-f/sigma) apodization
    """
    x,y,f= img_raw.shape
    apod = np.exp(-np.abs(freq - center_freq)/filter_para)
    for i in range(x):
        for j in range(y):
                for k in range(f):
                    img_raw[i,j,k]= img_raw[i,j,k]*apod[k]
    return img_raw

def fit_data(xVals,yVals,init_params= None,fit_function= gauss_fit, sort_frequencies= False, maxFev = 0):
    '''
    pOpt,pCov,rVal= fit_data(xVals,yVals,fit_function=lor_fit/gauss_fit)

    Pretty straightforward fitting procedure
    '''
    # input message and result printing tempate
    input_messege= 'Enter initial parameters separated by comma (leave blank and press enter if you want to skip this fit):\nBackground,A_peak1,sigma_peak1,x0_peak1,... \n'

    if not(np.any(init_params)):
        init_params= np.fromstring(input(input_messege),sep=',')
    if len(init_params)==0:
        print('Fitting aborted')
        return
    try:
        pOpt,pCov= opt.curve_fit(fit_function,xVals,yVals,p0=init_params, maxfev = maxFev)
    except RuntimeError:
        raise ValueError("Cannot find a good fit! Aborted")
        pOpt = None
        pCov = None

    if sort_frequencies:
        b= pOpt[0]
        cVals= pOpt[1::3]
        lVals= pOpt[2::3]
        fVals= pOpt[3::3]
        ind= np.argsort(fVals)
        pOpt[1::3]= cVals[ind]
        pOpt[2::3]= lVals[ind]
        pOpt[3::3]= fVals[ind]
    return pOpt,pCov

def generate_seed_parameters_manually(dat, fVals, rowVals, colVals):
    X, Y, npoints= dat.shape
    seed_pOpts= np.empty((len(rowVals),len(colVals)),dtype= object)

    for i,r in enumerate(rowVals):
        try:
            row= int((rowVals[i+1]+r)*0.5)
        except:
            row= int((r+X)*0.5)
        
        for j,c in enumerate(colVals):
            try:
                col= int(0.5*(c+colVals[j+1]))
            except:
                col= int(0.5*(c+Y))

            fig, ax= plt.subplots(nrows= 1, ncols= 2, figsize= (10,4))
            img= np.squeeze(np.copy(dat[:,:,10]))
            img[row-2:row+2,col-2:col+2]= np.nan
            ax[0].imshow(img)
            ax[1].plot(fVals, dat[row,col,:])

            plt.show()
            plt.close('all')
            seed= np.fromstring(input('Enter frequency:'),sep=',')
            if len(seed)==2:
                seed= np.append(seed, [False, False])
            if len(seed)==0:
                continue
            seed_pOpts[i,j]= generate_pinit(seed[:2],seed[2:])
    return seed_pOpts

def generate_seed_parameters_auto(dat, fVals, rowVals, colVals):
    X, Y, npoints= dat.shape
    seed_pOpts= np.empty((len(rowVals),len(colVals)),dtype= object)
    i0 = int(np.min(rowVals))
    i1 = int(np.max(rowVals))
    j0 = int(np.min(colVals))
    j1 = int(np.max(colVals))


    fig, ax= plt.subplots(nrows= 1, ncols= 2, figsize= (10,4))
    img = np.squeeze(np.copy(dat[:,:,40]))
    ax[0].imshow(img)
    ax[0].add_patch(Rectangle((i0, j0), i1 - i0 + 1, j1 - j0 + 1, fill=None, alpha = 1))
    ax[1].plot(fVals, dat[(i0+i1)//2, (j0+j1)//2, :])

    plt.show()
    plt.close('all')
    seed= np.fromstring(input('Enter frequency (for example: 2.71, 2.81, 2.91, 3.01):'),sep=',')

    peak_guess = []
    for i in range(len(seed)):
        peak_guess.append(False)
    for i in range(len(rowVals)):
        for j in range(len(colVals)):
            seed_pOpts[i,j]= generate_pinit(seed, peak_guess)
    return seed_pOpts


def getFitInfo(datfit, numpeaks, param):
    param= param.lower()
    ind= {'frequency':3, 'linewidth': 2, 'contrast': 1}[param]
    def helper(dat):
        if dat is None:
            if numpeaks==1:
                return np.nan
            else:
                return tuple(np.ones(numpeaks)*np.nan)
        pOpt= dat['pOpt']
        dat= pOpt[ind::3]
        ##print(dat)
        if numpeaks==1:
            return dat[0]
        else:
            return tuple(dat)
    vhelper= np.vectorize(helper)
    return vhelper(datfit)

def plot_compare(fVals, dat, datfit, fit_function, row, col, fig= None):
    if not(fig):
        plt.figure()
    else:
        plt.figure(fig.number)
    plt.plot(fVals, dat[row,col,:])
    plt.plot(fVals, fit_function(fVals, *datfit[row,col]['pOpt']))
    print(datfit[row,col]['pOpt'])
    plt.show()


def generate_bounds(numpeaks=2):
    lb= np.zeros(3*numpeaks+1)
    ub= np.zeros(3*numpeaks+1)
    lb[0]= 0; ub[0]= np.inf        # background
    lb[1::3]= -np.inf; ub[1::3]= 0 # contrast
    lb[2::3]= 0; ub[2::3]= np.inf  # linewidth
    lb[3::3]= 0; ub[3::3]= np.inf  # frequency
    return lb, ub

def generate_pinit(freqVals= None, peakHeights= None):
    ## 0 -- baesline; 1 -- intensity A; 2 -- width gamma; 3 -- offset x0
    p_init= np.zeros(3*len(freqVals)+1)
    gamma = 0.025
    p_init[0]= 1
    p_init[1::3]= np.pi*gamma/2*(np.array(peakHeights, dtype= float)-1)
    p_init[2::3]= gamma
    p_init[3::3]= freqVals
    ##print(p_init)
    return p_init

##################################### B field fitting functions

Sx= np.array([[0,1,0],[1,0,1],[0,1,0]])/np.sqrt(2)
Sy= np.array([[0,1,0],[-1,0,1],[0,-1,0]])/(np.sqrt(2)*1j)
Sz= np.array([[1,0,0],[0,0,0],[0,0,-1]])

#### helper functions for fitting
def fit_function_BT(xVals,Bx,By,Bz, T= 300):
    '''
    yVals= fit_function_BT(xVals,Bx,By,Bz,T)
    '''
    background= 1
    ampVals= np.ones(7)*-0.001
    linewidthVals= np.ones(7)*0.01
    freqVals= computeFreqVals(Bx,By,Bz,T)
    yFit= np.ones(xVals.shape)*background
    for a,x0,l in zip(ampVals,freqVals,linewidthVals):
        yFit+= gaussian(xVals,a,l,x0)
    return yFit

def lsolver(freqs):
    return lambda x: NLfieldsolver(x,freqs)


def NLfieldsolver(x,freqs):
    ## x0 -- bx; x1 -- by; x2 -- bz; x3 -- D
    freqVals= DFreqVals(x[0], x[1], x[2], x[3])
    freqVals = np.sort(freqVals)
    return freqVals - freqs

def DFreqVals(Bx,By,Bz, D):
    B0= [[Bx],[By],[Bz]]
    freqVals= []
    for i in range(1,5):
        R= nv_transformation(i).dot(nv_transformation(1).transpose())
        B = R.dot(B0)
        H= D*Sz**2 + 2.8025e-3*(Sx*B[0]+Sy*B[1]+Sz*B[2])

        w,v= np.linalg.eig(H)
        w= np.real(w)
        w.sort()

        r1= w[-1]-w[0]
        r2= w[-2]-w[0]

        freqVals= freqVals+[r1, r2]
    return freqVals

def computeFreqVals(Bx,By,Bz,T= 300):
    '''
    freqVals= computeFreqVals(B,T)

    Returns the resonance freqeuncies given the B field in the lab frame
    and the Temperature of the measurement
    B is in the frame of the 111 NV
    '''
    B= [[Bx],[By],[Bz]]
    freqVals= []
    for i in range(1,5):
        R= nv_transformation(i).dot(nv_transformation(1).transpose())
        H= NV_H(T,*R.dot(B))

        w,v= np.linalg.eig(H)
        w= np.real(w)
        w.sort()

        r1= w[-1]-w[0]
        r2= w[-2]-w[0]

        freqVals= freqVals+[r1, r2]
    return freqVals

def NV_H(T,Bx,By,Bz):
    '''
    H= NV_H(T,B)
    Returns a matrix that is the spin Hamiltonian of the NV ground state

    Defines the NV Hamiltonian in given the magentic field in the NV frame
    and the temperature
    '''
    # zero field term and temperature terms
#     d= np.array([2.8771, -4.625e-6, 1.067e-7, -9.325e-10, 1.739e-12, -1.838e-15])
#     H_zf= np.sum(d*T**(np.arange(len(d))))*Sz**2
    H_zf= (2.8771+ -4.625e-6*T+ 1.067e-7*T*T+ -9.325e-10*T*T*T+ 1.739e-12*T*T*T*T+ -1.838e-15*T*T*T*T*T)*Sz**2

    # magnetic field terms
    H_b= 2.8025e-3*(Sx*Bx+Sy*By+Sz*Bz)

    H= H_zf+H_b
    return H

def fit_function_BTP(xVals,Bx,By,Bz,T= 300,P= 0):
    '''
    yVals= fit_function_BTP(xVals,Bx,By,Bz,T, P)

    Implementing different linewidths for each peak
    '''
    background= 1
    ampVals= np.ones(7)*-0.001
    FWHMVals= np.ones(7)*0.01
    freqVals= computeFreqVals_pressure(Bx,By,Bz,T,P)
    yFit= np.ones(xVals.shape)*background
    for a,x0,l in zip(ampVals,freqVals,FWHMVals):
        yFit+= gaussian_c(xVals,a,l,x0)
    return yFit

def computeFreqVals_pressure(Bx,By,Bz, T= 300, P= 0):
    '''
    freqVals= computeFreqVals(B,T)

    Returns the resonance freqeuncies given the B field in the lab frame
    and the Temperature of the measurement

    B is in the frame of the 111 NV
    '''
    B= [[Bx],[By],[Bz]]
    freqVals= []
    for i in range(1,5):
        R= nv_transformation(i).dot(nv_transformation(1).transpose())
        H= NV_H_pressure(T,P,*R.dot(B))

        w,v= np.linalg.eig(H)
        w= np.real(w)
        w.sort()

        r1= w[-1]-w[0]
        r2= w[-2]-w[0]

        freqVals= freqVals+[r1, r2]
    return freqVals

def NV_H_pressure(T,P,Bx,By,Bz):
    '''
    H= NV_H_pressure(T,P,Bx,By,Bz)
    Returns a matrix that is the spin Hamiltonian of the NV ground state

    Defines the NV Hamiltonian in given the magentic field in the NV frame
    and temperature and a hydrostatic shift from temperature
    '''
    # zero field term and temperature terms
    #d= np.array([2.8771, -4.625e-6, 1.067e-7, -9.325e-10, 1.739e-12, -1.838e-15]) # the zeroth term includes the pressure shift
    #H_zf= np.sum(d*T**(np.arange(len(d))))*Sz**2+ 10e-3*P*Sz**2 ---> this gives a wrap around error
    H_zf= (2.87771 -4.625e-6*T+ 1.067e-7*T*T -9.325e-10*T*T*T+ 1.739e-12*T*T*T*T -1.838e-15*T*T*T*T*T)*Sz**2
    H_p= 10e-3*P*Sz**2
    # magnetic field terms
    H_b= 2.8e-3*(Sx*Bx+Sy*By+Sz*Bz)

    H= H_zf+H_b+H_p
    return H

def NV_Hes_pressure(T,P,Bx,By,Bz):
    '''
    H= NV_Hes_pressure(T,P,Bx,By,Bz)
    Returns a matrix that is the spin Hamiltonian of the NV ground state

    Defines the NV Hamiltonian in given the magentic field in the NV frame
    and temperature and a hydrostatic shift from temperature
    '''
    # zero field term and temperature terms
    #d= np.array([2.8771, -4.625e-6, 1.067e-7, -9.325e-10, 1.739e-12, -1.838e-15]) # the zeroth term includes the pressure shift
    #H_zf= np.sum(d*T**(np.arange(len(d))))*Sz**2+ 10e-3*P*Sz**2 ---> this gives a wrap around error
    H_zf= 1.43
    H_p= 10e-3*P*Sz**2
    # magnetic field terms
    H_b= 2.8e-3*(Sx*Bx+Sy*By+Sz*Bz)

    H= H_zf+H_b+H_p
    return H

#### helper functions: Rotation matrices
def rotx(theta): # same as Matlab
    '''
    Rotation around X axis by an angle theta
    '''
    return np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])

def roty(theta): # same as Matlab
    '''
    Rotation around Y axis by an angle theta
    '''
    return np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])

def rotz(theta): # same as Matlab
    '''
    Rotation around Z axis by an angle theta
    '''
    return np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

#### helper functions: Computing transformations for the different NV groups with respect to the lab frame
def nv_transformation(nv_group): # tested by function test_nv_transform
    '''
    Returns that transformation matrix to move from the crystal frame to the NV frame
    Group 1:  1  1  1
    Group 2: -1 -1  1
    Group 3:  1 -1 -1
    Group 4: -1  1 -1
    '''
    theta0= np.arccos(-1/3.)
    if nv_group==1:
        R= np.dot(rotz(np.pi),np.dot(roty(-theta0/2),rotz(-np.pi/4)))
    elif nv_group==2:
        R= np.dot(roty(theta0/2),rotz(-np.pi/4))
    elif nv_group==3:
        R= np.dot(roty(theta0/2-np.pi),rotz(np.pi/4))
    elif nv_group==4:
        R= np.dot(rotz(np.pi),np.dot(roty(np.pi-theta0/2),rotz(np.pi/4)))
    else:
        raise ValueError('Bad Input')
    return R

def test_nv_transform():
    '''
    Tests the transformation matrix for the NV groups
    '''
    dirs= np.array([[1, -1, 1, -1], [1, -1, -1, 1], [1, 1, -1, -1]])/np.sqrt(3);
    for i in range(1,5):
        R= nv_transformation(i);
        print('Testing NV group: {0}'.format(i));
        print(R)
        np.set_printoptions(precision=2);
        print(np.dot(R,dirs));
