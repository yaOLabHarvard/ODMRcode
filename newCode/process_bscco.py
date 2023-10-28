# %%
import WF_mat_data_class as wf
import WF_data_processing as pr
from matplotlib import pyplot as plt, cm
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
# rtfolderpath='C:/Users/esthe/OneDrive/Desktop/VSCode/Plotting/Data/WF/RT CeH9 Data/'
# homefolderpath='D:/work/py/attodry_lightfield/ODMRcode/newCode/data/'
# labtfolderpath='F:/NMR/NMR/py_projects/WF/ODMRcode/newCode/data_T/'
labbfolderpath='F:/NMR/NMR/py_projects/WF/ODMRcode/newCode/data_B/'
# figpath='D:/work/py/attodry_lightfield/ODMRcode/newCode/picture/'
# picklepath='D:/work/py/attodry_lightfield/ODMRcode/newCode/pickle/'
testpath = 'F:/NMR/NMR/py_projects/WF/ODMRcode/newCode/test/'
fileName = 'zfc-150K-2A'


#%%
## load multiple files
MFWF=wf.multiWFImage(labbfolderpath)
MFWF.setFileParameters(parameters=[0,0.1,0.25,0.5,1,2,7,5,10])
MFWF.test()
#%%
## do manual image correlation
MFWF.manualAlign(nslice = 3, referN = 0)

# %%
## pick a roi and do image correlations
MFWF.roi(xlow=43, ylow=40, xrange=20, yrange=20, plot=False)
MFWF.imageAlign(nslice = 2, referN = 0, rr=5, debug = False)

# %%
## pick a roi and do multi esr for all images
MFWF.roi(xlow=20, ylow=82, xrange=100, yrange=1, plot=True)
MFWF.roiMultiESRfit()

# %%
## create multiple de maps and plot them
MFWF.generateroiDEmap()
MFWF.plotroiDEmap(withroi=True)


# %%
# This block tests manual correction
testWF = MFWF.WFList[3]
# testWF.covList = {}
currentE = 1e-7
MFWF.roi(xlow=20, ylow=82, xrange=100, yrange=1, plot=True)
#MFWF.roi(xlow=55, ylow=82, xrange=20, yrange=1, plot=True)
# for x in MFWF.xr:
#     for y in MFWF.yr:
#         testWF.covList[(x,y)] = 0
testWF.fitErrordetection(MFWF.xr, MFWF.yr, epschi = currentE)
testWF.multiESRfitManualCorrection(isResume = False)
# %%
# This block tests auto correction
guessfound = [2.87,2.92,2.94]
testWF.multiESRfitAutoCorrection(guessfound, forced = False, isResume = False)
#%%
testWF.fitErrordetection(MFWF.xr, MFWF.yr, epschi = currentE)
testWF.multiESRfitManualCorrection(isResume = True)
#%%
## this block will test the single WFimage files
# testWF = wf.WFimage(ltfolderpath+fileName)
testWF = MFWF.WFList[3]
testWF.norm()
xr = np.arange(20,120)
ycut = 110
yr= np.arange(ycut, ycut+1)
testWF.multiESRfit(xr, yr)

# %%
# %%
# DD,EE=testWF.DEmap(plot=False)
# fig, ax = plt.subplots(nrows=1, ncols= 2, figsize= (15,6))
# img1 = ax[0].imshow(DD, vmax = 3, vmin = 2.8)
# ax[0].title.set_text("D map (GHz)")
# divider = make_axes_locatable(ax[0])
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(img1, cax=cax)


# img2 = ax[1].imshow(EE, vmax = 0.2, vmin = 0)
# ax[1].title.set_text("E map (GHz)")
# divider = make_axes_locatable(ax[1])
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(img2, cax=cax)

# plt.savefig(figpath+"DEmap"+str(currentT)+"K.png")
# plt.show()
# plt.close()
# %%
## test for single pt esr
# px=110
# py=90
# testWF.myFavoriatePlot(px, py, maxPeak = 4)
ycut = 110
linecut = [[20, ycut],[120, ycut]]
testWF.waterfallPlot(lineCut = linecut, stepSize = 4,  spacing = 0.005, plotTrace = True,plotFit=True, plot = False)
# %%
## create linecuts for single image
linecut = [[70, 10],[70, 140]]
testWF.waterfallPlot(lineCut = linecut, stepSize = 5,  spacing = 0.01, plotTrace = True,plotFit=True, plot = False)
plt.savefig(figpath+"vcut"+str(currentB)+"G.png")
linecut = [[10, 80],[140, 80]]
testWF.waterfallPlot(lineCut = linecut, stepSize = 5,  spacing = 0.01, plotTrace = True,plotFit=True, plot = False)
plt.savefig(figpath+"hcut"+str(currentB)+"G.png")
# %%
## B/H plot for single image
ref = EE[120, 30]
fig, ax = plt.subplots(nrows=1, ncols= 1, figsize= (6,6))
img = ax.imshow(EE/ref, vmax = 1.5, vmin = 0)
ax.title.set_text("B/H map")
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(img, cax=cax)
plt.savefig(figpath+"BHmap"+str(currentT)+"K.png")
plt.show()
plt.close()
# %%
## D and E line plot for single image
linecut = [[70, 10],[70, 140]]
testWF.DElineplot(lineCut = linecut , stepSize =1, plotTrace = True, plotD = False, plotE = True)
linecut = [[10, 80],[140, 80]]
testWF.DElineplot(lineCut = linecut , stepSize =1, plotTrace = True, plotD = False, plotE = True)

# %%
## outer peak width plot for single image
linecut = [[70, 10],[70, 140]]
testWF.DEwidthplot(lineCut = linecut , stepSize =1, plotTrace = True)
linecut = [[10, 80],[140, 80]]
testWF.DEwidthplot(lineCut = linecut , stepSize =1, plotTrace = True)

# %%
## splitting average over a roi versus all parameters for multiple images
Emeans = []
Estds = []
MFWF.roi(xlow=93, ylow=83, xrange=5, yrange=5, plot=True)
for tmpWF in MFWF.WFList:
    DD,EE=tmpWF.DEmap(plot=False)
    tmpEs = EE[MFWF.rroi[0][0]:MFWF.rroi[0][1],MFWF.rroi[1][0]:MFWF.rroi[1][1]]
    tmpMean = np.mean(tmpEs)
    Emeans.append(tmpMean)
    tmpstd = np.std(tmpEs)
    Estds.append(tmpstd)

fig, ax = plt.subplots(nrows=1, ncols= 1, figsize= (6,6))
ax.plot(MFWF.ParaList, Emeans, '-', color = 'r')
ax.set_ylim(0, 0.2)
ax.errorbar(MFWF.ParaList, Emeans, yerr = Estds, fmt ='o')
plt.show()
plt.close()
# %%
# test for averaged roi versus parameter
MFWF.roi(xlow=20, ylow=82, xrange=100, yrange=1, plot=True)
MFWF.generateroiDEmap()
# ## point
# MFWF.roi(xlow=70, ylow=70, xrange=1, yrange=1, plot=True)
# MFWF.roiDEvsParas()
## line
MFWF.roi(xlow=20, ylow=82, xrange=100, yrange=1, plot=True)
MFWF.roiDEvsParas()
MFWF.lineroiDEvsParas()
# ## square
# MFWF.roi(xlow=70, ylow=70, xrange=5, yrange=5, plot=True)
# MFWF.roiDEvsParas()
# %%
MFWF.dumpFitResult()
# %%
MFWF.loadFitResult()

# %%
for i in range(MFWF.Nfile):
    dataE = MFWF.roiEmap[MFWF.rroi[0][0]:MFWF.rroi[0][1],MFWF.rroi[1][0]:MFWF.rroi[1][1], i].flatten()
    dataD = MFWF.roiDmap[MFWF.rroi[0][0]:MFWF.rroi[0][1],MFWF.rroi[1][0]:MFWF.rroi[1][1], i].flatten()
    filename = MFWF.fileDir[i].split('.')[0] + '_fit.txt'
    output = np.array([dataE,dataD]).transpose()
    np.savetxt(homefolderpath+filename, output)
# %%
MFWF.imgShift = np.insert(MFWF.imgShift, 0, 0, axis = 0)


# %%
