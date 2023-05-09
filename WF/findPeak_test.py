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
import pickle
from WF_mat_data_class import WFimage
path = "F:/NMR/NMR/py_projects/WF/ODMRcode/WF/raw_data/"
# filename= '20K_n36dbm_N3569_3A.mat'
filename= '20K_n36dbm_N3287_2p5A.mat'
# filename= '20K_n36dbm_N2108_2A.mat'

# In[1]:
# Generate sample data
WF = WFimage(path + filename)
testimg = WF.sliceImage(Nslice=100)
fig, ax = plt.subplots(nrows=1, ncols= 1, figsize= (6,6))
ax.imshow(testimg)
plt.show()
WF.norm()
#In[2]:
xpos = 5
ypos = 54
WF.myFavoriatePlot(xpos, ypos)

#In[3]:
yr = np.arange(0, 96, 1)
xr = np.arange(0, 96, 1)
WF.multiESRfit(xr, yr, backupOption=np.array([3.85, 4.25]))
# t = WF.npeakManualCorrection(eps = 1e-3)
# In[4]:
WF.maskData(.3e-3)
NPlot = WF.multiNpeakplot()
NPimage = WF.Npeak#*WF.dataMask
fig, ax = plt.subplots(nrows=1, ncols= 2, figsize= (12,6))
## plot the image
cmap = plt.cm.get_cmap('Wistia', 4)
NPplot = ax[0].imshow(NPimage, cmap = cmap)
divider = make_axes_locatable(ax[0])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(NPplot, cax=cax)

# NNPimage = WF.Npeak*WF.dataMask
# NNPplot = ax[1].imshow(NNPimage, cmap = cmap)
# divider = make_axes_locatable(ax[0])
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(NNPplot, cax=cax)
# plt.savefig(path + "pic/numpeaks_0p3A.png")
# plt.show()

XX, YY, ZZ = WF.maskContourgen()
Contourplot = ax[1].contourf(XX, YY, ZZ)
ax[1].set_ylim(ax[1].get_ylim()[::-1])
asp = np.abs(np.diff(ax[1].get_xlim())[0] / np.diff(ax[1].get_ylim())[0])
asp /= np.abs(np.diff(ax[0].get_xlim())[0] / np.diff(ax[0].get_ylim())[0])
ax[1].set_aspect(asp)
divider = make_axes_locatable(ax[1])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(Contourplot, cax=cax)
plt.savefig(path + "pic/20K_2p5A_fitting_summary.png")
plt.show()
# In[5]:
Dm, Em = WF.DEmap()
Dm = Dm#*WF.dataMask
Em = Em#*WF.dataMask
fig, ax = plt.subplots(nrows=1, ncols= 2, figsize= (12,6))
Dplot = ax[0].imshow(Dm, vmin=3.95, vmax=4.1)
divider = make_axes_locatable(ax[0])
cax = divider.append_axes("right", size=0.1, pad=0.1)
plt.colorbar(Dplot, cax=cax)

Eplot = ax[1].imshow(Em, vmin=0.036, vmax=0.23)
divider = make_axes_locatable(ax[1])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(Eplot, cax=cax)
plt.savefig(path + "pic/20K_2p5A_DEmap.png")
plt.show()



# In[6]:
# Serialize the object to a file
with open(path + 'WF20K3A.pkl', 'wb') as f:
    pickle.dump(WF, f)

# Deserialize the object from the file
with open(path + 'WF20K3A.pkl', 'rb') as f:
    WFload = pickle.load(f)

xpos = 5
ypos = 54
WFload.myFavoriatePlot(xpos, ypos)
# %%
