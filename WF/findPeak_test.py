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
image = WF.sliceImage(10)
WF.norm()
xpos = 60
ypos = 40
spESR = WF.pointESR(xpos, ypos)
#In[2]:
fig, ax = plt.subplots(nrows=1, ncols= 2, figsize= (10,6))
## plot the image
IMGplot = ax[0].imshow(image)
ax[0].add_patch(Rectangle((xpos - 1, ypos -1), 2, 2, fill=None, alpha = 1))
divider = make_axes_locatable(ax[0])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(IMGplot, cax=cax)


## plot single ESR
ESRplot = ax[1].plot(WF.fVals, spESR, '-')
peaks = WF.singleESRpeakfind(xpos, ypos)
popt, pcov, chisq = WF.singleESRfit(xpos, ypos)
print(chisq)
ax[1].plot(WF.fVals[peaks], spESR[peaks], 'x')
for i in np.arange(int(np.floor(len(popt)/3))):
   params= popt[1+3*i:4+3*i]
   ax[1].plot(WF.fVals,popt[0]+mf.lorentzian(WF.fVals,*params), '-')

# plt.plot(x_data, y_data)
# plt.plot(x_data[peaks], y_data[peaks], "x")
plt.show()

#In[3]:
yr = np.arange(0, 115, 1)
xr = np.arange(0, 115, 1)
WF.multiESRfit(xr, yr)
NPimage = WF.multiNpeakplot()
chiImage = WF.multichisqplot()
fig, ax = plt.subplots(nrows=1, ncols= 2, figsize= (10,6))
## plot the image
NPplot = ax[0].imshow(NPimage)
divider = make_axes_locatable(ax[0])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(NPplot, cax=cax)

cmap = plt.cm.jet
norm = mcolors.LogNorm(vmin=1e-8, vmax=1)
chiplot = ax[1].imshow(chiImage, cmap=cmap, norm=norm)
divider = make_axes_locatable(ax[1])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(chiplot, cax=cax)
plt.show()

# %%
