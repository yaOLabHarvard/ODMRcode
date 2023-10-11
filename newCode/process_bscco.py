# %%
import WF_mat_data_class as wf
import WF_data_processing as pr
from matplotlib import pyplot as plt, cm
import pickle
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
# rtfolderpath='C:/Users/esthe/OneDrive/Desktop/VSCode/Plotting/Data/WF/RT CeH9 Data/'
ltfolderpath='F:/NMR/NMR/py_projects/WF/ODMRcode/newCode/data/'
fileName = 'bnop-150K-zfc-250mw-n30db-magb-2A.mat'

## for multiple files
#%%
MFWF=wf.multiWFImage(ltfolderpath)
# MFWF.test()
# # %%
# picklefolder='F:/NMR/NMR/py_projects/WF/ODMRcode/forEsther/data/'
# # with open(picklefolder + 'bscco_rt_only_0Aand0p5A.pkl', 'rb') as f:
# #     MFWF = pickle.load(f)
# with open(picklefolder + 'bscco_rt_only_0Aand0p5A.pkl', 'wb') as f:
#     pickle.dump(MFWF, f)

# %%
MFWF.roi(xlow=40, ylow=30, xrange=30, yrange=30, plot=True)
MFWF.imageAlign(nslice = 3, referN = 0, rr=10, plot = True)


# %%
MFWF.roi(xlow=60, ylow=60, xrange=20, yrange=20, plot=True)
MFWF.roiMultiESRfit()
MFWF.roiDEmap(plot=True)
# %%
# %%
MFWF.WFList[1].MWintmap(plot = True)
# %%

# for single files
#%%
testWF = MFWF.WFList[1]
testWF.norm()

# %%

##initguess = [2.75, 2.9, 3.0, 3.08]
xlist=np.arange(10,140,1)
ylist=np.arange(10,140,1)
testWF.multiESRfit(xlist, ylist, max_peak = 6)
# %%
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

plt.show()
plt.close()
# %%
px=97
py=25
testWF.myFavoriatePlot(px, py)
# %%
linecut = [[70, 10],[70, 140]]
testWF.waterfallPlot(lineCut = linecut, stepSize = 5,  spacing = 0.01, plotTrace = True,plotFit=True, plot = True)
linecut = [[10, 80],[140, 80]]
testWF.waterfallPlot(lineCut = linecut, stepSize = 5,  spacing = 0.01, plotTrace = True,plotFit=True, plot = True)
# %%
ref = EE[120, 30]
fig, ax = plt.subplots(nrows=1, ncols= 1, figsize= (6,6))
img = ax.imshow(EE/ref, vmax = 1.5, vmin = 0)
ax.title.set_text("B/H map")
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(img, cax=cax)

plt.show()
plt.close()
# %%
