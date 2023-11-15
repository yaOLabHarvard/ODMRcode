# %%
import WF_mat_data_class as wf
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
# labtfolderpath='F:/NMR/NMR/py_projects/WF/ODMRcode/newCode/data_T/'
labbfolderpath='F:/NMR/NMR/py_projects/WF/ODMRcode/newCode/data_B/'
labqfolderpath='F:/NMR/NMR/py_projects/WF/ODMRcode/newCode/data_Q/'
labfolderpath='F:/NMR/NMR/py_projects/WF/ODMRcode/newCode/data_B/igor/'
testpath = 'F:/NMR/NMR/py_projects/WF/ODMRcode/newCode/test/'
txtpath = 'F:/NMR/NMR/py_projects/WF/ODMRcode/newCode/esr_igor/'
bssco3Tfolderpath='F:/NMR/NMR/py_projects/WF/ODMRcode/newCode/data_bscco3_T/'
bssco3bfolderpath='F:/NMR/NMR/py_projects/WF/ODMRcode/newCode/data_bscco3_B/'
bssco320gpabfolderpath='F:/NMR/NMR/py_projects/WF/ODMRcode/newCode/data_bscco3_20GPa_B/'
fileName = 'bscco3_100K-1p6a'



#%%
######################################
## load multiple files
MFWF=wf.multiWFImage(bssco320gpabfolderpath)
MFWF.setFileParameters(parameters=[0, 0.4,0.8,1.2])
## parameters=[100, 103, 113, 128, 149, 158, 160, 160, 61, 70, 80, 90, 50]
MFWF.test()
######################################

#%%
######################################
## load single file
testWF=wf.WFimage(bssco3Tfolderpath + fileName)
testWF.norm()
######################################


#%%
######################################
## add an file into the MFWF
MFWF.addFile('bscco3_50K_1p6a')
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
MFWF.roi(xlow=60, ylow=20, xrange=20, yrange=20, plot=True)
MFWF.imageAlign(nslice = 3, referN = 0, rr=5, debug = True)
######################################


# %%
######################################
## pick a roi and do multi esr for all images
MFWF.roi(xlow=20, ylow=15, xrange=120, yrange=120, plot=True)
MFWF.roiMultiESRfit(max_peak = 8)
######################################


# %%
######################################
## pick a roi and do multi esr for a single image

MFWF.roi(xlow=20, ylow=15, xrange=120, yrange=120, plot=True)
testWF = MFWF.WFList[12]
testWF.multiESRfit(MFWF.xr, MFWF.yr, max_peak = 8)
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
MFWF.roi(xlow=40, ylow=40, xrange=100, yrange=100, plot=True)
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
testWF = MFWF.WFList[0]
px=120
py=80
testWF.myFavoriatePlot(px, py, maxPeak = 6)

# ycut = 90
# linecut = [[20, ycut],[140, ycut]]
# linecut = [[40,50],[140,80]]
# testWF.waterfallPlot(lineCut = linecut, stepSize = 5,  spacing = 0.005, plotTrace = True,plotFit=False)
# testWF.waterfallMap(lineCut = linecut, stepSize =1, plotTrace = False, localmin = False, flipped = False)
# testWF.DElineplot(lineCut = linecut , stepSize =1, plotTrace = True, plotD = False, plotE = True)
# testWF.DEwidthplot(lineCut = linecut , stepSize =1, plotTrace = True)
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
# MFWF.roi(xlow=29, ylow=130, xrange=120, yrange=1, plot=False)
# MFWF.generateroiDEmap()
## point
MFWF.roi(xlow=115, ylow=75, xrange=1, yrange=1, plot=True)
MFWF.generateroiDEmap()
MFWF.roiDEvsParas(Eymax=0.15, Dymax=3)
# ## line
# MFWF.roi(xlow=20, ylow=82, xrange=100, yrange=1, plot=True)
# MFWF.roiDEvsParas(Eymax = 0.5)
# MFWF.lineroiDEvsParas(Espacing = 0.1)
# ## square
# MFWF.roi(xlow=70, ylow=70, xrange=5, yrange=5, plot=True)
# MFWF.roiDEvsParas()
######################################

# %%
######################################
# save D and E line results
labfolderpath='F:/NMR/NMR/py_projects/WF/ODMRcode/newCode/test/igor/'
MFWF.roi(xlow=115, ylow=75, xrange=100, yrange=1, plot=True)
for i in range(MFWF.Nfile):
    dataE = MFWF.roiEmap[MFWF.rroi[0][0]:MFWF.rroi[0][1],MFWF.rroi[1][0]:MFWF.rroi[1][1], i].flatten()
    dataD = MFWF.roiDmap[MFWF.rroi[0][0]:MFWF.rroi[0][1],MFWF.rroi[1][0]:MFWF.rroi[1][1], i].flatten()
    filename = MFWF.fileDir[i].split('.')[0] + '_fit.txt'
    output = np.array([dataE,dataD]).transpose()
    np.savetxt(labfolderpath+filename, output)
######################################

# %%
######################################
# save D and E point results
labfolderpath='F:/NMR/NMR/py_projects/WF/ODMRcode/newCode/test/igor/'
filename = 'x138y120.txt'
MFWF.roi(xlow=138, ylow=120, xrange=1, yrange=1, plot=True)
MFWF.generateroiDEmap()
exportData = np.zeros((MFWF.Nfile, 2))
for i in range(MFWF.Nfile):
    dataE = MFWF.roiEmap[MFWF.rroi[0][0]:MFWF.rroi[0][1],MFWF.rroi[1][0]:MFWF.rroi[1][1], i].flatten()[0]
    dataD = MFWF.roiDmap[MFWF.rroi[0][0]:MFWF.rroi[0][1],MFWF.rroi[1][0]:MFWF.rroi[1][1], i].flatten()[0]
    exportData[i] = [dataE, dataD]

np.savetxt(labfolderpath+filename, exportData)
######################################
# %%
from scipy.spatial import cKDTree
import itertools
from matplotlib import pyplot as plt

def fitCheckQ(mwf, xyarray):
    correctArray = []
    ##print(xyarray)
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
            print("current iter: {}".format(it))
            print("current seed list {}".format(mwf.seedList))
            for [x, y] in mwf.seedList:
                theGroup = bathArray[theTree.query_ball_point([x, y], dist)]
                theGroup = fitCheckQ(mwf, theGroup)
                ##print("the current group {}".format(theGroup))
                currentLevel.append(theGroup)
                
                for [xx, yy] in theGroup:
                    if mwf.optList[(x, y)] is None:
                        continue
                    else:
                        theGuess = mwf.optList[(x, y)][3::3]
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
            mwf.seedList = np.unique(np.array(currentLevel), axis = 0)
            if debug:
                fig = plt.figure(num = 1, clear = True, figsize= (6,6))
                ax = fig.add_subplot(1, 1, 1)
                IMGplot = ax.imshow(mwf.dat[:,:,3].copy())
                sx,sy = mwf.seedList.transpose()
                ax.scatter(sy, sx, color = 'r')
                plt.show()
                
            mwf.isMultfit = True
            mwf.fitErrordetection(mwf.seedList, epschi = epschi)
                # print(mwf.errorIndex)
                # for [x,y] in mwf.errorIndex:
                #     print(mwf.optList[x, y])
            mwf.multiESRfitManualCorrection(isResume = False, seedScan = False)


testWF = MFWF.WFList[5] 
MFWF.roi(xlow=20, ylow=20, xrange=100, yrange=100, plot=True)
testWF.randomSeedGen(MFWF.xyArray, pointRatio = 2e-3, plot=True)
testWF.multiESRfitManualCorrection(isResume = False, seedScan = True)
multiESRSeedfit(testWF, MFWF.xr, MFWF.yr, iter = 5, dist = 3, epschi = 1e-2, debug = True)
# %%
bathArray = np.array(list(itertools.product(np.arange(1,10), np.arange(1,10))))
theTree = cKDTree(bathArray)
print(bathArray[theTree.query_ball_point([5, 5], 2)])
# %%
# %%
from matplotlib import pyplot as plt
figpath = "F:/NMR/NMR/py_projects/WF/ODMRcode/newCode/test/"
ii = 12
testWF = MFWF.WFList[8]
MFWF.roi(xlow=20, ylow=15, xrange=120, yrange=120, plot=True)
testWF.multix = MFWF.xr
testWF.multiy = MFWF.yr
DD,EE=testWF.DEmap(plot=False)
fig, ax = plt.subplots(nrows=1, ncols= 2, figsize= (15,6))
img1 = ax[0].imshow(DD, vmax = 3, vmin = 2.8)
ax[0].title.set_text("D map (GHz)")
divider = make_axes_locatable(ax[0])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(img1, cax=cax)


img2 = ax[1].imshow(EE, vmax = 0.2, vmin = 0)
ax[1].title.set_text("E map (GHz)")
divider = make_axes_locatable(ax[1])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(img2, cax=cax)

plt.savefig(figpath+"data" + str(ii) + ".png")
plt.show()
plt.close()
# %%
MFWF.roi(xlow=20, ylow=20, xrange=100, yrange=100, plot=True)
for i in MFWF.xr:
        for j in MFWF.yr:
            testWF.ckList[(i,j)] = 0
bathArray = np.array(list(itertools.product(MFWF.xr, MFWF.yr)))
theTree = cKDTree(bathArray)
currentLevel = []
sl = [[46,91]]
for [x, y] in sl:
    theGroup = bathArray[theTree.query_ball_point([x, y], 2)]
    theGroup = fitCheckQ(testWF, theGroup)
    print(theGroup)
    currentLevel.append(theGroup)

    print("x{} y{}".format(x, y))
    theGuess = testWF.optList[(x, y)][3::3]
    print("guess {}".format(theGuess))
    for [xx, yy] in theGroup:

                        
        try:
            pOpt, pCov, chiSq = testWF.singleESRfit(xx, yy, initGuess=theGuess)
            testWF.FitCheck = 2    
        except ValueError:
            print("The fitting fails.. please correct!")
            pOpt = None
            pCov = None
            chiSq = 1
            testWF.FitCheck = -2
                                                                          
        testWF.optList[(xx, yy)] = pOpt
        testWF.covList[(xx, yy)] = pCov
        testWF.sqList[(xx, yy)] = chiSq
        testWF.ckList[(xx, yy)] = testWF.FitCheck
        print("{} {} has completed".format(x, y))
                
currentLevel = [item for sublist in currentLevel for item in sublist]
print(currentLevel)
testWF.fitErrordetection(currentLevel, epschi = 1e-4)
print(testWF.errorIndex)
testWF.multiESRfitManualCorrection(isResume = False, seedScan = False)
# %%
MFWF.imgShift = np.zeros((MFWF.Nfile, 2))
# %%
MFWF.setFileParameters(parameters=[100, 103, 113, 128, 149, 158, 160, 160, 61, 70, 80, 90, 50])
# %%
