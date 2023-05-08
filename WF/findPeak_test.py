# In[0]:
import multipeak_fit as mf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.integrate import quad
from scipy.signal import find_peaks
import scipy as sp
import random as rd
from WF_mat_data_class import WFimage
path = "F:/NMR/NMR/py_projects/WF/ODMRcode/WF/raw_data/"
filename= '100xobj_Bz0p3A.mat'
# In[1]:
# Generate sample data
WF = WFimage(path + filename)
WF.norm()
xpos = 14
ypos = 37
#In[2]:
WF.myFavoriatePlot(xpos, ypos)

#In[3]:
yr = np.arange(0, 115, 1)
xr = np.arange(0, 115, 1)
WF.maskData()
WF.multiESRfit(xr, yr)
NPlot = WF.multiNpeakplot()
NPimage = NPlot*WF.dataMask
t = WF.npeakManualCorrection()
fig, ax = plt.subplots(nrows=1, ncols= 2, figsize= (12,6))
## plot the image
cmap = plt.cm.get_cmap('Wistia', 8)
NPplot = ax[0].imshow(NPimage, cmap = cmap)
divider = make_axes_locatable(ax[0])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(NPplot, cax=cax)

NNPimage = WF.Npeak*WF.dataMask
NNPplot = ax[1].imshow(NNPimage, cmap = cmap)
divider = make_axes_locatable(ax[0])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(NNPplot, cax=cax)
plt.savefig(path + "pic/numpeaks_0p3A.png")
plt.show()

# cmap = plt.cm.jet
# norm = mcolors.LogNorm(vmin=1e-8, vmax=1)
# chiplot = ax[1].imshow(chiImage, cmap=cmap, norm=norm)
# divider = make_axes_locatable(ax[1])
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(chiplot, cax=cax)
# plt.show()

# %%
