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
filename= ['WF20K2A.pkl', 'WF20K2p5A.pkl', 'WF20K3A.pkl']
# In[1]
WFList = []
for fi in filename:
    with open(path + fi, 'rb') as f:
        WFList.append(pickle.load(f))

# In[2]
imageList = []
for wf in WFList:
    imageList.append(wf.sliceImage(Nslice=100))

nC = len(imageList)
fig, ax = plt.subplots(nrows=1, ncols= nC, figsize= (nC*6,6))
## plot the image
for i in range(nC):
    ax[i].imshow(imageList[i])
plt.show()
# In[3]
roi = [[3,23],[40,60]]
tmpImg = WFList[0].generateTmp(ROI = roi, Nslice = 100)
corrPos = np.zeros((len(WFList), 2))
dPos = np.zeros((len(WFList), 2))
for i in range(len(WFList)):
    corrPos[i] = WFList[i].imgCorr(tmpImg = tmpImg)
print(corrPos)
for i in range(len(WFList)):
    dPos[i] = corrPos[i] - corrPos[0]
# In[4]
i = 0
for wf in WFList:
    wf.shiftDat(dx = dPos[i][0], dy = dPos[i][1])
    i += 1


# %%
