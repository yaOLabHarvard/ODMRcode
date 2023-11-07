# %%
import WF_mat_data_class as wf
import numpy as np

# labtfolderpath='F:/NMR/NMR/py_projects/WF/ODMRcode/newCode/data_T/'
labbfolderpath='F:/NMR/NMR/py_projects/WF/ODMRcode/newCode/data_B/'
labqfolderpath='F:/NMR/NMR/py_projects/WF/ODMRcode/newCode/data_Q/'
labfolderpath='F:/NMR/NMR/py_projects/WF/ODMRcode/newCode/data_B/igor/'
testpath = 'F:/NMR/NMR/py_projects/WF/ODMRcode/newCode/test/'
txtpath = 'F:/NMR/NMR/py_projects/WF/ODMRcode/newCode/esr_igor/'
fileName = 'zfc-base-b5A'



#%%
######################################
## load multiple files
MFWF=wf.multiWFImage(labbfolderpath)
MFWF.setFileParameters(parameters=[0,0.1,0.25,0.5,1,2,7,5,-1])
##MFWF.test()
######################################

#%%
######################################
## load single file
testWF=wf.WFimage(labbfolderpath + fileName)
testWF.norm()
######################################

# %%
######################################
## dump and load fitting and correlation results
MFWF.dumpFitResult()
# %%
MFWF.loadFitResult(refreshChecks = True)
######################################

#%%
######################################
## do manual image correlation
MFWF.manualAlign(nslice = 3, referN = 0)
######################################


# %%
######################################
## pick a roi and do auto image correlations
MFWF.roi(xlow=43, ylow=30, xrange=20, yrange=20, plot=True)
MFWF.imageAlign(nslice = 3, referN = 0, rr=5, debug = True)
######################################


# %%
######################################
## pick a roi and do multi esr for all images
MFWF.roi(xlow=20, ylow=82, xrange=100, yrange=1, plot=True)
MFWF.roiMultiESRfit()
######################################


# %%
######################################
## pick a roi and do multi esr for a single image

MFWF.roi(xlow=20, ylow=20, xrange=100, yrange=100, plot=True)
testWF = MFWF.WFList[5]
testWF.multiESRfit(MFWF.xr, MFWF.yr, max_peak = 6)
######################################

# %%
######################################
## pick a roi and do auto image correlations
MFWF.roi(xlow=43, ylow=30, xrange=20, yrange=20, plot=True)
MFWF.imageAlign(nslice = 3, referN = 0, rr=5, debug = True)
######################################


# %%
######################################
## create a seed region in the selected roi and plot seeds
testWF = MFWF.WFList[5]
MFWF.roi(xlow=20, ylow=20, xrange=100, yrange=100, plot=True)
testWF.randomSeedGen(MFWF.xyArray, pointRatio = 0.001, plot=True)
testWF.multiESRfitManualCorrection(isResume = False, seedScan = True)
######################################

#%%

# %%
######################################
## manual correction with given error thorshold
testWF = MFWF.WFList[5]
currentE = 1e-5
MFWF.roi(xlow=20, ylow=82, xrange=100, yrange=1, plot=True)
#MFWF.roi(xlow=55, ylow=82, xrange=20, yrange=1, plot=True)
testWF.fitErrordetection(MFWF.xyArray, epschi = currentE)
testWF.multiESRfitManualCorrection(isResume = False)
######################################

# %%
######################################
## auto correlation for rest of the points in roi using the guess given below 
## 2.728,2.773,2.876,2.901,2.991,3.016,3.07,3.076
## 2.875,2.91,2.94,2.96, 2.97
##This block tests auto correction
guessfound = [2.91,2.93]
testWF.multiESRfitAutoCorrection(guessfound, forced = False, isResume = True)
######################################
#%%
######################################
## resume the manual correlation
##testWF.fitErrordetection(MFWF.xr, MFWF.yr, epschi = currentE)
testWF.multiESRfitManualCorrection(isResume = True)
######################################

#%%
######################################
## test for single pt esr by mfp or waterfall plots
testWF = MFWF.WFList[5]
px=31
py=66
testWF.myFavoriatePlot(px, py, maxPeak = 4)

# ycut = 82
# linecut = [[10, ycut],[130, ycut]]
# ##testWF.waterfallPlot(lineCut = linecut, stepSize = 4,  spacing = 0.005, plotTrace = True,plotFit=True)
# testWF.waterfallMap(lineCut = linecut, stepSize =1, plotTrace = False, localmin = False, flipped = False)
##testWF.DElineplot(lineCut = linecut , stepSize =1, plotTrace = True, plotD = False, plotE = True)
##testWF.DEwidthplot(lineCut = linecut , stepSize =1, plotTrace = True)
######################################


# %%
######################################
## create multiple de maps and plot them
MFWF.generateroiDEmap()
MFWF.plotroiDEmap(withroi=True)
######################################

# %%
######################################
# generate DE map from fitted roi region and plot them versus parameters
MFWF.roi(xlow=20, ylow=82, xrange=100, yrange=1, plot=True)
MFWF.generateroiDEmap()
# ## point
# MFWF.roi(xlow=70, ylow=70, xrange=1, yrange=1, plot=True)
# MFWF.roiDEvsParas()
## line
MFWF.roi(xlow=20, ylow=82, xrange=100, yrange=1, plot=True)
MFWF.roiDEvsParas(Eymax = 0.5)
MFWF.lineroiDEvsParas(Espacing = 0.1)
# ## square
# MFWF.roi(xlow=70, ylow=70, xrange=5, yrange=5, plot=True)
# MFWF.roiDEvsParas()
######################################

# %%
######################################
# save D and E results
labfolderpath='F:/NMR/NMR/py_projects/WF/ODMRcode/newCode/data_B/igor/'
MFWF.roi(xlow=20, ylow=82, xrange=100, yrange=1, plot=True)
for i in range(MFWF.Nfile):
    dataE = MFWF.roiEmap[MFWF.rroi[0][0]:MFWF.rroi[0][1],MFWF.rroi[1][0]:MFWF.rroi[1][1], i].flatten()
    dataD = MFWF.roiDmap[MFWF.rroi[0][0]:MFWF.rroi[0][1],MFWF.rroi[1][0]:MFWF.rroi[1][1], i].flatten()
    filename = MFWF.fileDir[i].split('.')[0] + '_fit.txt'
    output = np.array([dataE,dataD]).transpose()
    np.savetxt(labfolderpath+filename, output)
######################################


# %%
from scipy.spatial import cKDTree
import itertools
from matplotlib import pyplot as plt

def fitCheckQ(mwf, xyarray):
    correctArray = []
    print(xyarray)
    for [x, y] in xyarray:
        if mwf.ckList[(x, y)]<2:
            correctArray.append([x, y])
        
    return correctArray

def multiESRSeedfit(mwf, xlist, ylist, iter = 5, dist = 3, epschi = 5e-5, debug = True):
    for i in xlist:
        for j in ylist:
            mwf.ckList[(i,j)] = 0
        
        if mwf.isSeeded and mwf.isManualFit:
            bathArray = np.array(list(itertools.product(xlist, ylist)))
            theTree = cKDTree(bathArray)
            ##currentLevel = []
            for it in range(iter):
                currentLevel = []
                for [x, y] in mwf.seedList:
                    theGroup = bathArray[theTree.query_ball_point([x, y], dist)]
                    theGroup = fitCheckQ(mwf, theGroup)
                    currentLevel.append(theGroup)
                    
                    print("x{} y{}".format(x, y))
                    theGuess = mwf.optList[(x, y)][3::3]
                    print("guess {}".format(theGuess))
                    for [xx, yy] in theGroup:

                        
                        try:
                            pOpt, pCov, chiSq = mwf.singleESRfit(xx, yy, initGuess=theGuess)
                            mwf.FitCheck = 2    
                        except ValueError:
                            print("The fitting fails.. please correct!")
                            pOpt = None
                            pCov = None
                            chiSq = 1
                            mwf.FitCheck = -2
                                                                          
                        mwf.optList[(xx, yy)] = pOpt
                        mwf.covList[(xx, yy)] = pCov
                        mwf.sqList[(xx, yy)] = chiSq
                        mwf.ckList[(xx, yy)] = mwf.FitCheck
                    print("{} {} has completed".format(x, y))
                
                currentLevel = [item for sublist in currentLevel for item in sublist]
                mwf.seedList = np.array(currentLevel)
                if debug:
                    fig = plt.figure(num = 1, clear = True, figsize= (6,6))
                    ax = fig.add_subplot(1, 1, 1)
                    IMGplot = ax.imshow(mwf.dat[:,:,3].copy())
                    sx,sy = mwf.seedList.transpose()
                    ax.scatter(sx, sy, color = 'r')
                
                mwf.fitErrordetection(mwf.seedList, epschi = 1e-3)
                print(mwf.errorIndex)
                for [x,y] in mwf.errorIndex:
                    print(mwf.optList[x, y])
                mwf.multiESRfitManualCorrection(isResume = False, seedScan = False)
multiESRSeedfit(testWF, MFWF.xr, MFWF.yr, iter = 5, dist = 1, epschi = 5e-5, debug = True)
# %%
bathArray = np.array(list(itertools.product(np.arange(1,10), np.arange(1,10))))
theTree = cKDTree(bathArray)
print(bathArray[theTree.query_ball_point([5, 5], 2)])
# %%
print(testWF.seedList)
print(testWF.optList[46,91])
theGuess = testWF.optList[46,91][3::3]
pOpt, pCov, chiSq = testWF.singleESRfit(46, 90, initGuess=theGuess)
testWF.myFavoriatePlot(46, 90, fitParas = [pOpt, pCov, chiSq])
# %%
