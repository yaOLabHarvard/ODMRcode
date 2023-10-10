# %%
import WF_mat_data_class as wf
import WF_data_processing as pr
from matplotlib import pyplot as plt, cm
from matplotlib.patches import Rectangle
from scipy.signal import correlate2d
import pickle
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
ltfolderpath='F:/NMR/NMR/py_projects/WF/ODMRcode/forEsther/data/150K/'
fileName = ['bnop-150K-cali-250mw-n30db-magb-2A','bnop-150K-cali-250mw-n30db-magb-1p5A',\
            'bnop-150K-cali-250mw-n30db-magb-1A', 'bnop-150K-cali-250mw-n30db-magb-0p5A',\
                  'bnop-150K-cali-250mw-n30db-0A']
def saveFig(fileName):
    WFtest = wf.WFimage(filename=ltfolderpath + fileName + '.mat')
    WFtest.norm()
    bigImage = WFtest.dat[:,:,3].copy()
    bigImage -= np.mean(bigImage)
    picName = fileName + '.txt'
    np.savetxt(ltfolderpath+picName, bigImage)

def loadFig(fileName):
    return np.loadtxt(ltfolderpath+fileName + '.txt')

def plotList(imgList):
    Nimg = len(imgList)
    fig, ax = plt.subplots(nrows=1, ncols= Nimg, figsize= (6*Nimg,6))

    for i in range(Nimg):
        ax[i].imshow(imgList[i])

    plt.show()
    plt.close()

def plotroi(roi, img):
    fig, ax = plt.subplots(nrows=1, ncols= 1, figsize= (6,6))
    img1 = ax.imshow(img)
    ax.add_patch(Rectangle((roi[1][0], roi[0][0]), roi[1][1] - roi[1][0],\
                           roi[0][1] -roi[0][0], fill=None, alpha = 1, color = 'red'))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img1, cax=cax)

    plt.show()
    plt.close()

# ax[0].add_patch(Rectangle((roi[0][0], roi[1][0]), roi[0][1] - roi[0][0],\
#                            roi[1][1] -roi[1][0], fill=None, alpha = 1, color = 'red'))
# divider = make_axes_locatable(ax[0])
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(img1, cax=cax)

def normalImg(img):
    img = np.array(img)
    maxx = img.max()
    minn = img.min()
    return (2*img - maxx - minn)/(maxx - minn)
# %%
for f in fileName:
    saveFig(f)
# %%
imgList = [loadFig(f) for f in fileName]
Nimg = len(imgList)
imgList = [normalImg(img) for img in imgList]
roi = [[40,70],[30,60]]
refImg = imgList[0]
plotroi(roi, refImg)
# %%
# method 1
smallImage = refImg[roi[0][0]:roi[0][1], roi[1][0]:roi[1][1]]
fig, ax = plt.subplots(nrows=1, ncols= 1, figsize= (6,6))
img1 = ax.imshow(smallImage, vmax = 1, vmin = -1)
plt.show()
plt.close()
corr1 = [correlate2d(img, smallImage, mode='same') for img in imgList]
rr = 10
corrSmall = [corr[roi[0][0]-rr:roi[0][1]+rr, roi[1][0]-rr:roi[1][1]+rr] for corr in corr1]
xys = np.zeros((Nimg, 2))
for i in range(Nimg):
    xx, yy  = np.unravel_index(np.argmax(corrSmall[i]), corrSmall[i].shape)
    xx += roi[0][0]-rr
    yy += roi[1][0]-rr
    xys[i][0] = xx
    xys[i][1] = yy
    print("WF {}: x is found at {} px; y is found at {} px".format(i , xys[i][0], xys[i][1]))
plotList(imgList)
plotList(corr1)

# %%
fileName = 'bnop-150K-cali-250mw-n30db-magb-2A'
WFtest = wf.WFimage(filename=ltfolderpath + fileName + '.mat')
WFtest.norm()

# %%
linecut = [[70,0],[70,150]]
_,ax = WFtest.waterfallPlot(lineCut=linecut, stepSize=5, plotTrace=True, plot=True, spacing = 0.005)

# %%
