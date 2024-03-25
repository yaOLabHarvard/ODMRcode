# %%
import WF_mat_data_class as wf
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import pyplot as plt
%matplotlib qt
# labtfolderpath='F:/NMR/NMR/py_projects/WF/ODMRcode/newCode/data_T/'
labbfolderpath='F:/NMR/NMR/py_projects/WF/ODMRcode/newCode/data_B/'
labqfolderpath='F:/NMR/NMR/py_projects/WF/ODMRcode/newCode/data_Q/'
labfolderpath='F:/NMR/NMR/py_projects/WF/ODMRcode/newCode/data_B/igor/'
testpath = 'F:/NMR/NMR/py_projects/WF/ODMRcode/newCode/test/'
txtpath = 'F:/NMR/NMR/py_projects/WF/ODMRcode/newCode/esr_igor/'
# bssco3Tfolderpath='F:/NMR/NMR/py_projects/WF/ODMRcode/newCode/data_bscco3_T/'
# bssco3bfolderpath='F:/NMR/NMR/py_projects/WF/ODMRcode/newCode/data_bscco3_B/'
# bssco320gpabfolderpath='F:/NMR/NMR/py_projects/WF/ODMRcode/newCode/data_bscco3_20GPa_B/'
# bssco320gpatfolderpath='F:/NMR/NMR/py_projects/WF/ODMRcode/newCode/data_bscco3_20GPa_T/'
# bscco415gpatfolderpath='F:/NMR/NMR/py_projects/WF/ODMRcode/newCode/bscco4/15gpa/vsT/'
# bscco415gpabfolderpath='F:/NMR/NMR/py_projects/WF/ODMRcode/newCode/bscco4/15gpa/vsB_40dbm/'
# bscco415gpabzfolderpath='F:/NMR/NMR/py_projects/WF/ODMRcode/newCode/bscco4/15gpa/110K_vsBz/'
# bscco415gpabxfolderpath='F:/NMR/NMR/py_projects/WF/ODMRcode/newCode/bscco4/15gpa/vsBx/'
# bscco420gpapfolderpath='F:/NMR/NMR/py_projects/WF/ODMRcode/newCode/bscco4/20gpa/rtstrain/'
# bscco420gpatfolderpath='F:/NMR/NMR/py_projects/WF/ODMRcode/newCode/bscco4/20gpa/50K/'
ni327s1rtxcalipath='F:/NMR/NMR/py_projects/WF/ODMRcode/newCode/ni327_s1/rt/xcali/'
ni327s1rtzcalipath='F:/NMR/NMR/py_projects/WF/ODMRcode/newCode/ni327_s1/rt/zcali/'
ni327s19p7gpa103Kpath='F:/NMR/NMR/py_projects/WF/ODMRcode/newCode/ni327_s1/9p7/103K/'
ni327s114p2gpavsTpath='F:/NMR/NMR/py_projects/WF/ODMRcode/newCode/ni327_s1/14p2/vsT/'
ni327s116p5gpavsTpath='F:/NMR/NMR/py_projects/WF/ODMRcode/newCode/ni327_s1/16p5/vsT/'
ni327s116p5gpavsBpath='F:/NMR/NMR/py_projects/WF/ODMRcode/newCode/ni327_s1/16p5/vsB-19K_new/'
fileName = '327_s1_16p5gpa_19K_mwn36_4p5A_new'
plt.style.use('norm2')


#%%
######################################
## load multiple files
MFWF=wf.multiWFImage(ni327s116p5gpavsBpath)
MFWF.setFileParameters(parameters=[0,1,2,3,4,5,6,7,8])
## parameters=[0, 0.4,0.8,1.2]
## parameters=[100, 103, 113, 128, 149, 158, 160, 160, 61, 70, 80, 90, 50]
MFWF.test()
######################################

#%%
######################################
## load single file
testWF=wf.WFimage(ni327s116p5gpavsBpath + fileName)
testWF.norm()
######################################

#%%
######################################
## check the data
testWF.myFavoriatePlotMousepick()
######################################




#%%
######################################
## add an file into the MFWF
MFWF.addFile('327_s1_16p5gpa_101K_mwn34_3A')
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
MFWF.manualAlign(nslice = 3, referN = 1)
######################################


# %%
######################################
## pick a roi and do auto image correlations
MFWF.roi(xlow=120, ylow=35, xrange=30, yrange=30, plot=True)
MFWF.imageAlign(nslice = 3, referN = 4, rr=5, debug = False)
######################################

# %%
######################################
## check a single file out of MFWF
testWF = MFWF.WFList[0]
testWF.norm()
testWF.myFavoriatePlot(x=108,y=97)
######################################

# %%
######################################
## check a series of plots at the same point using MFWF
MFWF.myFavoriatePlot(x=108,y=97)
######################################
# %%
######################################
## pick a roi and do multi esr for all images
MFWF.roi(xlow=10, ylow=10, xrange=155, yrange=155, plot=True)
MFWF.roiMultiESRfit(max_peak = 5)
######################################

# %%
######################################
## pick a linecut and do multi esr for all images
MFWF.roi(xlow=82, ylow=73, xrange=36, yrange=13, plot=True, lineCut=True)
MFWF.roiMultiESRfit(max_peak = 6, lineFit = True)
######################################


# %%
######################################
## pick a roi and do multi esr for a single image

MFWF.roi(xlow=10, ylow=10, xrange=155, yrange=155, plot=True)
testWF = MFWF.WFList[3]
testWF.multiESRfit(MFWF.xr, MFWF.yr, max_peak = 4)
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
testWF = MFWF.WFList[0]
MFWF.roi(xlow=10, ylow=10, xrange=155, yrange=155, plot=True)
testWF.randomSeedGen(MFWF.xyArray, pointRatio = 0.001, plot=True)
testWF.multiESRfitManualCorrection(isResume = False, seedScan = True)
######################################

#%%
print(MFWF.xyArray)
# %%
######################################
## manual correction with given error thorshold
testWF = MFWF.WFList[0]
currentE = 0
MFWF.roi(xlow=10, ylow=10, xrange=155, yrange=155, plot=True, lineCut=True)
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
currentE = 0
guessfound = [2.825,3.024,3.15,3.22]
MFWF.roi(xlow=10, ylow=10, xrange=155, yrange=155, plot=True, lineCut=True)
testWF.fitErrordetection(MFWF.xyArray, epschi = currentE)
testWF.multiESRfitAutoCorrection(guessfound, forced = True, isResume = False)
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
testWF = MFWF.WFList[4]
# px=80
# py=90
# testWF.myFavoriatePlot(px, py, maxPeak = 6)

ycut = 80
linecut = [[0, ycut],[175, ycut]]
# xcut = 100
# linecut = [[xcut, 0],[xcut, 173]]
# linecut = [[82,73],[117,85]]
# testWF.waterfallPlot(lineCut = linecut, stepSize = 10,  spacing = 0.005, plotTrace = True,plotFit=False)
testWF.waterfallMap(lineCut = linecut, stepSize =1, plotTrace = True, localmin = False, flipped = False)
# testWF.DElineplot(lineCut = linecut , stepSize =1, plotTrace = True, plotD = False, plotE = True)
# testWF.DEwidthplot(lineCut = linecut , stepSize =1, plotTrace = True)
######################################



# %%
######################################
## create multiple de maps and plot them
MFWF.roi(xlow=10, ylow=10, xrange=155, yrange=155, plot=True)
MFWF.generateroiDEmap()
MFWF.plotroiDEmap(withroi=True)
######################################

# %%
######################################
# generate DE map from fitted roi region and plot them versus parameters
# MFWF.roi(xlow=29, ylow=130, xrange=120, yrange=1, plot=False)
# MFWF.generateroiDEmap()
## point
# MFWF.roi(xlow=115, ylow=75, xrange=1, yrange=1, plot=True)
# MFWF.generateroiDEmap()
# MFWF.roiDEvsParas(Eymax=0.15, Dymax=3)
# ## line
MFWF.roi(xlow=40, ylow=90, xrange=20, yrange=1, plot=True)
MFWF.roiDEvsParas(Eymax = 0.5)
MFWF.lineroiDEvsParas(Espacing = 0.1)
# ## square
# MFWF.roi(xlow=70, ylow=70, xrange=5, yrange=5, plot=True)
# MFWF.roiDEvsParas()
######################################

# %%
######################################
## B/H plot for single image
figpath = "F:/NMR/NMR/py_projects/WF/ODMRcode/newCode/test/"
MFWF.roi(xlow=10, ylow=10, xrange=155, yrange=155, plot=True)
testWF = MFWF.WFList[4]
testWF.multix = MFWF.xr
testWF.multiy = MFWF.yr
# currentT = MFWF.ParaList[8]
DD,EE=testWF.DEmap(plot=False)
ref = EE[150, 150]
fig, ax = plt.subplots(nrows=1, ncols= 1, figsize= (6,6))
img = ax.imshow(EE/ref, vmax = 1.2, vmin = 0.8)
# img = ax.imshow(DD, vmax = 4, vmin = 2)
ax.title.set_text("B/H map")
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(img, cax=cax)
plt.savefig(figpath+"BHmap_16p5gpa_"+str(101)+"K.png")
plt.show()
plt.close()
######################################
# %%
######################################
## sigma perp and para imaging for a single image
figpath = "F:/NMR/NMR/py_projects/WF/ODMRcode/newCode/test/"
testWF = MFWF.WFList[-1]
# currentT = MFWF.ParaList[8]
testWF.DEmap(plot=False, iscustom = True)
DD1=testWF.Dmap.copy()
testWF.DEmap(plot=False, iscustom = True)
DD2=testWF.Dmap.copy()
TheinverseMatrix = np.array([[0.050609, 0.0174183], [-0.0501975, 0.118225]])
spara = np.zeros((testWF.X, testWF.Y), dtype=float)
sperp = np.zeros((testWF.X,testWF.Y), dtype=float)
for x in range(testWF.X):
    for y in range(testWF.Y):
        [spara[x,y], sperp[x,y]] = np.matmul(TheinverseMatrix, np.array([1000*(DD1[x,y]-2.877), 1000*(DD2[x,y]-2.877)]))
fig, ax = plt.subplots(nrows=1, ncols= 2, figsize= (15,6))
img1 = ax[0].imshow(spara, vmax = 18, vmin = 5)
ax[0].title.set_text("sigma parallel (GPa)")
divider = make_axes_locatable(ax[0])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(img1, cax=cax)

img2 = ax[1].imshow(sperp, vmax = 18, vmin = 5)
ax[1].title.set_text("sigma perpendicular (GPa)")
divider = make_axes_locatable(ax[1])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(img2, cax=cax)

# plt.savefig(figpath+"BHmap"+str(currentT)+"K.png")
plt.show()
plt.close()
######################################

# %%
######################################
## DE plot for one of MFWF
figpath = "F:/NMR/NMR/py_projects/WF/ODMRcode/newCode/test/"
ii = 3
testWF = MFWF.WFList[ii]
MFWF.roi(xlow=10, ylow=10, xrange=155, yrange=155, plot=True)
testWF.multix = MFWF.xr
testWF.multiy = MFWF.yr
DD,EE=testWF.DEmap(plot=False)
fig, ax = plt.subplots(nrows=1, ncols= 2, figsize= (15,6))
img1 = ax[0].imshow(DD, vmax = 3.1, vmin = 2.9)
ax[0].title.set_text("D map (GHz)")
divider = make_axes_locatable(ax[0])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(img1, cax=cax)


img2 = ax[1].imshow(EE, vmax = 0.16, vmin = 0.1)
ax[1].title.set_text("E map (GHz)")
divider = make_axes_locatable(ax[1])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(img2, cax=cax)

plt.savefig(figpath+"data" + str(ii) + ".png")
plt.show()
plt.close()
######################################
# %%
######################################
# save D and E line results
labfolderpath='F:/NMR/NMR/py_projects/WF/ODMRcode/newCode/test/igor/'
MFWF.roi(xlow=82, ylow=73, xrange=36, yrange=13, plot=True, lineCut=True)
for i in range(MFWF.Nfile):
    tmpWF = MFWF.WFList[i]
    # dataE = MFWF.roiEmap[MFWF.rroi[0][0]:MFWF.rroi[0][1],MFWF.rroi[1][0]:MFWF.rroi[1][1], i].flatten()
    # dataD = MFWF.roiDmap[MFWF.rroi[0][0]:MFWF.rroi[0][1],MFWF.rroi[1][0]:MFWF.rroi[1][1], i].flatten()
    linenn = len(MFWF.xyArray)
    dataD = []
    dataE = []
    for [x,y] in MFWF.xyArray:
        DD,EE = tmpWF.DandE(x,y)
        dataD.append(DD)
        dataE.append(EE)

filename = MFWF.fileDir[i].split('.')[0] + '_hp_fit_2.txt'
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
######################################
# save raw esr data with series of paras
xx = 55
yy = 100
labfolderpath='F:/NMR/NMR/py_projects/WF/ODMRcode/newCode/test/igor/'
filename = 'ni327_s1_16p5gpa_vsB_-x{}y{}.txt'.format(xx,yy)

with open(labfolderpath+filename, 'a+') as file:

    for i in range(MFWF.Nfile):
        file.write("parameter {}\n".format(MFWF.ParaList[i]))
        tmpWF = MFWF.WFList[i]
        file.write("Freq(GHz)   Signal\n")
        fs = tmpWF.fVals
        ss = tmpWF.pointESR(xx,yy)
        for j in range(len(fs)):
            file.write("{}    {}\n".format(fs[j], ss[j]))

        file.write("####################################\n")
file.close()
######################################

 # %%
######################################
# create splitting vs B plot for a specific linecut
##testWF = MFWF.WFList[0]
ycut = 90
linecut = [[20, ycut],[140, ycut]]
MFWF.roi(xlow=20, ylow=90, xrange=120, yrange=1, plot=True)
MFWF.roiMultiESRfit(max_peak = 8)
##wf.lineCutGen(lineCut = linecut, stepSize =10)
######################################


# %% under construction
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
# %%

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
