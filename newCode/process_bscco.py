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
labqfolderpath='F:/NMR/NMR/py_projects/WF/ODMRcode/newCode/data_Q/'
labfolderpath='F:/NMR/NMR/py_projects/WF/ODMRcode/newCode/data_B/igor/'
# figpath='D:/work/py/attodry_lightfield/ODMRcode/newCode/picture/'
# picklepath='D:/work/py/attodry_lightfield/ODMRcode/newCode/pickle/'
testpath = 'F:/NMR/NMR/py_projects/WF/ODMRcode/newCode/test/'
txtpath = 'F:/NMR/NMR/py_projects/WF/ODMRcode/newCode/esr_igor/'
fileName = 'zfc-150K-2A'


#%%
## load multiple files
MFWF=wf.multiWFImage(labqfolderpath)
MFWF.setFileParameters(parameters=[150,210,270,0,50])
##MFWF.test()

# %%
MFWF.dumpFitResult()
# %%
MFWF.loadFitResult(refreshChecks = True)

#%%
## do manual image correlation
MFWF.manualAlign(nslice = 3, referN = 0)

#%%
print(MFWF.imgShift)
# %%
## pick a roi and do image correlations
MFWF.roi(xlow=43, ylow=30, xrange=20, yrange=20, plot=True)
MFWF.imageAlign(nslice = 3, referN = 0, rr=5, debug = True)

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
testWF = MFWF.WFList[4]
currentE = 0
MFWF.roi(xlow=20, ylow=82, xrange=100, yrange=1, plot=True)
#MFWF.roi(xlow=55, ylow=82, xrange=20, yrange=1, plot=True)
testWF.fitErrordetection(MFWF.xr, MFWF.yr, epschi = currentE)
testWF.multiESRfitManualCorrection(isResume = False)
# %%
## 2.728,2.773,2.876,2.901,2.991,3.016,3.07,3.076
## 2.875,2.91,2.94,2.96, 2.97
##This block tests auto correction
guessfound = [2.91,2.93]
testWF.multiESRfitAutoCorrection(guessfound, forced = False, isResume = True)
#%%
##testWF.fitErrordetection(MFWF.xr, MFWF.yr, epschi = currentE)
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
testWF = MFWF.WFList[4]
# px=15
# py=82
# testWF.myFavoriatePlot(px, py, maxPeak = 4)
##75,125,160,200
## 0.    0.1   0.25  0.5   1.    2.    7.    5.   -1. 
ycut = 82
linecut = [[10, ycut],[130, ycut]]
##testWF.waterfallPlot(lineCut = linecut, stepSize = 4,  spacing = 0.005, plotTrace = True,plotFit=True)
testWF.waterfallMap(lineCut = linecut, stepSize =1, plotTrace = False, localmin = False, flipped = False)
# %%
# %%
## create linecuts for single image
linecut = [[70, 10],[70, 140]]
testWF.waterfallPlot(lineCut = linecut, stepSize = 5,  spacing = 0.01, plotTrace = True,plotFit=True)
plt.savefig(figpath+"vcut"+str(currentB)+"G.png")
linecut = [[10, 80],[140, 80]]
testWF.waterfallPlot(lineCut = linecut, stepSize = 5,  spacing = 0.01, plotTrace = True,plotFit=True)
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
MFWF.roiDEvsParas(Eymax = 0.5)
MFWF.lineroiDEvsParas(Espacing = 0.1)
# ## square
# MFWF.roi(xlow=70, ylow=70, xrange=5, yrange=5, plot=True)
# MFWF.roiDEvsParas()

# %%
MFWF.roi(xlow=20, ylow=82, xrange=100, yrange=1, plot=True)
for i in range(MFWF.Nfile):
    dataE = MFWF.roiEmap[MFWF.rroi[0][0]:MFWF.rroi[0][1],MFWF.rroi[1][0]:MFWF.rroi[1][1], i].flatten()
    dataD = MFWF.roiDmap[MFWF.rroi[0][0]:MFWF.rroi[0][1],MFWF.rroi[1][0]:MFWF.rroi[1][1], i].flatten()
    filename = MFWF.fileDir[i].split('.')[0] + '_fit.txt'
    output = np.array([dataE,dataD]).transpose()
    np.savetxt(labfolderpath+filename, output)


# # %%
# import os
# import pickle
# picklepath = None
# for i in range(MFWF.Nfile):
#         tmpd =  {}
#         for x in MFWF.xr:
#             for y in MFWF.yr:
#                 tmpd[(x,y)] = 0
#         tmp = [MFWF.WFList[i].optList, MFWF.WFList[i].sqList, tmpd, MFWF.imgShift[i]]
#         filename = MFWF.fileDir[i].split('.')[0] + '_fit.pkl'
#         if picklepath is None:
#             picklepath = MFWF.folderPath + 'pickle/'
#             if not os.path.exists(picklepath):
#                 os.makedirs(picklepath)
#         with open(picklepath + filename, 'wb') as f:
#             pickle.dump(tmp, f)
#             print("{} file has been dumped!".format(i))

# %%
haha = np.array([1,2,3,4,5,6,7,8,9,10])
nums = np.array([1])
dlList =np.zeros(len(nums)*3, dtype = int)
print(nums)
for i in range(len(nums)):
    print(nums)
    flag = int(3*nums[i]+1)
    dlList[i:i+3]=[flag, flag+1, flag+2]
haha = np.delete(haha, dlList)
print(haha)
# %%
aa= np.array([1,2,3,4,5,6,7,8,9,10])
nums = np.fromstring(input("input the peak number you wish to delete (for example: 0, 1, 2):"), sep=',')
nums = np.array(nums)
dlList =np.zeros(len(nums)*3, dtype = int)
for i in range(len(nums)):

    flag = 3*nums[i]+1
    dlList[3*i:3*i+3]=[flag, flag+1, flag+2]
    print(dlList)

aa = np.delete(aa, dlList)
# %%
# import scipy
# testdata = MFWF.WFList[5].pointESR(120, 30)
# print(scipy.signal.argrelmin(testdata, order = 15))
# plt.plot(testdata)
# plt.show()
# %%
nlist= ['0', '0p1', '0p25','0p5','1','2','7','5','0qu']
for i in range(9):
    esry = MFWF.WFList[i].pointESR(100, 82)
    esrf = MFWF.WFList[i].fVals
    plt.plot(esrf, esry)
    plt.show()
    fname = 'x100y82_'+nlist[i]+ '.txt'
    data = np.array([esry, esrf]).transpose()
    np.savetxt(txtpath+fname, data)
# %%
MFWF.test()
# %%
