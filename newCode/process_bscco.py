# %%
import WF_mat_data_class as wf
import WF_data_processing as pr
from matplotlib import pyplot as plt, cm
import pickle
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
# rtfolderpath='C:/Users/esthe/OneDrive/Desktop/VSCode/Plotting/Data/WF/RT CeH9 Data/'
ltfolderpath='D:/work/py/attodry_lightfield/ODMRcode/newCode/data/'
fileName = 'bnop-150K-cali-250mw-n30db-magb-2A.mat'
# %%
# MFWF=wf.multiWFImage(ltfolderpath)
# # MFWF.test()
# # # %%
# # picklefolder='F:/NMR/NMR/py_projects/WF/ODMRcode/forEsther/data/'
# # # with open(picklefolder + 'bscco_rt_only_0Aand0p5A.pkl', 'rb') as f:
# # #     MFWF = pickle.load(f)
# # with open(picklefolder + 'bscco_rt_only_0Aand0p5A.pkl', 'wb') as f:
# #     pickle.dump(MFWF, f)

# # %%
# MFWF.roi(xlow=40, ylow=30, xrange=30, yrange=30, plot=True)
# MFWF.imageAlign(nslice = 3, referN = 0, rr=10, plot = True)

# # %%
# px=60
# py=80
# MFWF.myFavoriatePlot(px, py)
# # %%

# MFWF.roi(xlow=60, ylow=60, xrange=20, yrange=20, plot=True)
# MFWF.roiMultiESRfit()
# MFWF.roiDEmap(plot=True)
# # %%
# linecut = [[70, 20],[70, 130]]
# MFWF.WFList[4].waterfallPlot(lineCut = linecut, stepSize = 4,  spacing = 0.01, plotTrace = True, plot = True)
# # %%
# MFWF.WFList[4].MWintmap(plot = True)
# # %%
#%%
testWF = wf.WFimage(ltfolderpath + fileName)
testWF.norm()

# %%
linecut = [[85, 30],[85, 120]]
initguess = [2.75, 2.85, 3.02, 3.08]
xlist=np.array([85])
ylist=np.arange(30,120,1)
testWF.multiESRfit(xlist, ylist, max_peak = 6, initGuess = initguess)
testWF.waterfallPlot(lineCut = linecut, stepSize = 4,  spacing = 0.01, plotTrace = True,plotFit=True, plot = True)
# %%
