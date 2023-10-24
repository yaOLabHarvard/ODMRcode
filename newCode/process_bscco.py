# %%
import WF_mat_data_class as wf
import WF_data_processing as pr
from matplotlib import pyplot as plt, cm
import pickle
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
# rtfolderpath='C:/Users/esthe/OneDrive/Desktop/VSCode/Plotting/Data/WF/RT CeH9 Data/'
homefolderpath='D:/work/py/attodry_lightfield/ODMRcode/newCode/data/'
# labtfolderpath='F:/NMR/NMR/py_projects/WF/ODMRcode/newCode/data_T/'
# labbfolderpath='F:/NMR/NMR/py_projects/WF/ODMRcode/newCode/data_B/'
# figpath='D:/work/py/attodry_lightfield/ODMRcode/newCode/picture/'
# picklepath='D:/work/py/attodry_lightfield/ODMRcode/newCode/pickle/'
testpath = 'F:/NMR/NMR/py_projects/WF/ODMRcode/newCode/test/'
fileName = 'zfc-150K-2A'


#%%
## load multiple files
MFWF=wf.multiWFImage(homefolderpath)
MFWF.setFileParameters(parameters=[20,150,160,170,180,190,200,210,220,230,240,250,270])
# MFWF.test()
#%%
## pickle save and load
# with open(picklepath + 'bscco_rt_only_0Aand0p5A.pkl', 'rb') as f:
#     MFWF = pickle.load(f)
# # %%
# with open(picklepath + 'bscco_rt_only_0Aand0p5A.pkl', 'wb') as f:
#     pickle.dump(MFWF, f)

# %%
## pick a roi and do image correlations
MFWF.roi(xlow=43, ylow=40, xrange=20, yrange=20, plot=False)
MFWF.imageAlign(nslice = 2, referN = 0, rr=5, debug = False)

# %%
## pick a roi and do multi esr for all images
MFWF.roi(xlow=60, ylow=60, xrange=20, yrange=20, plot=True)
MFWF.roiMultiESRfit()

# %%
## create multiple de maps and plot them
MFWF.generateroiDEmap()
MFWF.plotroiDEmap(withroi=True)


# %%
# This block tests fitting edition
testWF = MFWF.WFList[0]
MFWF.roi(xlow=50, ylow=50, xrange=50, yrange=50, plot=True)
testWF.multiESRfitManualCorrection(MFWF.xr, MFWF.yr, isResume = False)

#%%
## this block will test the single WFimage files
# testWF = wf.WFimage(ltfolderpath+fileName)
testWF = MFWF.WFList[0]
testWF.norm()

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
px=63
py=105
testWF.myFavoriatePlot(px, py, maxPeak = 8)
# linecut = [[20, 78],[130, 78]]
# testWF.waterfallPlot(lineCut = linecut, stepSize = 2,  spacing = 0.01, plotTrace = True,plotFit=True, plot = False)
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
# test for averaged roi versus parameters
MFWF.generateroiDEmap()
## point
MFWF.roi(xlow=70, ylow=70, xrange=1, yrange=1, plot=True)
MFWF.roiDEvsParas()
## line
MFWF.roi(xlow=60, ylow=70, xrange=20, yrange=1, plot=True)
MFWF.roiDEvsParas()
MFWF.lineroiDEvsParas()
## square
MFWF.roi(xlow=70, ylow=70, xrange=5, yrange=5, plot=True)
MFWF.roiDEvsParas()
# %%
