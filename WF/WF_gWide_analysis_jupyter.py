#In[1]:
import multipeak_fit as mf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.integrate import quad
import scipy as sp
path = "F:/NMR/NMR/py_projects/WF/ODMRcode/WF/raw_data/"
filename= '440mW_50um_n31dbm_23K_5ms.mat'

# In[2]:
# ### load the .mat data
fVals, dat, xFrom, xTo, X, Y, npoints = mf.read_matfile(path + filename, normalize= False)
img1 = dat[:,:,3].copy()
print("data loaded")

# # In[3]:
# ## plot a specific point ESR
# ii = 250
# jj = 300
# dat= mf.normalize_widefield(dat)
# yVals = np.squeeze(dat[ii, jj,:])
# ax = plt.subplot()
# ijplot = ax.plot(fVals, yVals)
# plt.show()

# In[3]:
xlow =  150
xhigh = 350
ylow = 190
yhigh = 390
fig, ax = plt.subplots(nrows=2, ncols= 2, figsize= (10,6),gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [1, 1]})
##fig.tight_layout(h_pad = 2)
## plot the image before the normalization
dat= mf.normalize_widefield(dat)
IMGplot = ax[0,0].imshow(img1)
ax[0, 0].add_patch(Rectangle((xlow, ylow), xhigh - xlow + 1, yhigh - ylow + 1, fill=None, alpha = 1))
divider = make_axes_locatable(ax[0,0])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(IMGplot, cax=cax)


## plot the center of the ESR

ESRcenter = np.squeeze(np.copy(dat[:,:,int(npoints/2)]))
ESRcenterplot = ax[1,0].imshow(1 - ESRcenter)
divider = make_axes_locatable(ax[1,0])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(ESRcenterplot, cax=cax)


## plot a single point ESR

row= 250
col= 250
# plot results
yVals= np.squeeze(dat[row, col,:])
np.savetxt(path + "23K.txt",np.transpose([fVals, yVals]))
singleESRplot = ax[0, 1].plot(fVals, yVals, 'o')
# pOpt, pCov= mf.fit_data(fVals, yVals, init_params= mf.generate_pinit([2.87,2.87], [False, False]), fit_function= mf.lor_fit)
# ax[0, 1].plot(fVals, mf.lor_fit(fVals, *pOpt), '--')
# for i in np.arange(int(np.floor(len(pOpt)/3))):
#    params= pOpt[1+3*i:4+3*i]
#    ax[0, 1].plot(fVals,pOpt[0]+mf.lorentzian(fVals,*params), '-')

## plot the total MW effect region
MWmap = np.zeros((X,Y), dtype = float)
for i in range(X):
    for j in range(Y):
        MWmap[i, j] = npoints - sum(dat[i, j, :])

MWmapplot = ax[1,1].imshow(MWmap)
divider = make_axes_locatable(ax[1,1])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(MWmapplot, cax=cax)
plt.savefig(path + "pic/23K_n31dbm.png")
plt.show()

# In[4]:
## Do fit to multiple pixels
single_peakQ = False
rows = np.arange(xlow, xhigh, 1)
cols = np.arange(ylow, yhigh, 1)
maxFFev = 150*(npoints + 1)
seed_pOpts = mf.generate_seed_parameters_auto(dat, fVals, rows, cols)
##print(seed_pOpts)
##print(seed_pOpts.shape)
if single_peakQ:
    freqfit = np.zeros((X,Y))
    pintfit = np.zeros((X,Y))
datfit= np.empty((X,Y),dtype= object)
boolfit= np.full((X,Y),False, dtype= bool)

r = 0
for i in rows:
    c = 0
    for j in cols:
        i = int(i)
        j = int(j)
        ##print("{},{}".format(i,j))
        if boolfit[i,j]:
            continue
        yVals= np.squeeze(dat[i,j,:])
        yVals= yVals/np.mean(yVals[-5:])
        if np.any(np.isnan(yVals)):
            print('Pixel:[%i,%i] is skipped'%(i,j))
            continue
        p_init= list(seed_pOpts[r,c])
        try:
            pOpt, pCov= mf.fit_data(fVals,yVals, init_params= p_init, fit_function= mf.lor_fit, maxFev = maxFFev)
            if single_peakQ:
                freqfit[i,j] = pOpt[3]
                pintfit[i,j] = pOpt[1]
                print(freqfit[i,j])
            datfit[i,j]= {'pOpt': pOpt, 'pCov': pCov}
            boolfit[i,j]= True
        except sp.optimize.OptimizeWarning:
            print('Pixel:[%i,%i] OptimizeWarning'%(i,j))
            continue
        except RuntimeError:
            print('Pixel:[%i,%i] encountered RuntimeError'%(i,j))
            continue
        c += 1
    r += 1

# In[5]:
if single_peakQ:
    fig, ax = plt.subplots(nrows=1, ncols= 2, figsize= (15, 6))
    plt.sca(ax[0])
    plt.imshow(pintfit,vmax = -5e-4,vmin = -3.5e-3)
    plt.colorbar()
    plt.title('Intensity', fontsize= 20)

    plt.sca(ax[1])
    plt.imshow(freqfit, vmin = 2.875, vmax = 2.881)
    plt.colorbar()

    plt.title('freq [GHz]', fontsize= 20)
    plt.savefig(path + "pic/23K_n25dbm_single_peak_fit.png")
    plt.show()
##print(datfit[250-2:250+2, 250-2:250+2])
np.save(path + filename.replace('.mat','_datfit.npy'), datfit)
np.save(path + filename.replace('.mat','_boolfit.npy'), boolfit)
print("fitting info are saved!!")


# In[6]:
## load the fitting info from .npy file
datfit= np.load(path + filename.replace('.mat','_datfit.npy'),allow_pickle=True)
boolfit= np.load(path + filename.replace('.mat','_boolfit.npy'),allow_pickle=True)

freq= mf.getFitInfo(datfit, numpeaks= 2, param= 'frequency')
int= mf.getFitInfo(datfit, numpeaks= 2, param= 'linewidth')
fig, ax = plt.subplots(nrows=1, ncols= 2, figsize= (15, 6))
plt.sca(ax[0])
plt.imshow(freq)
plt.colorbar()
plt.title('freq [GHz]', fontsize= 20)

plt.sca(ax[1])
plt.imshow(int, vmax = 0.05, vmin = 0)
plt.colorbar()
plt.title('linewidth', fontsize= 20)
plt.show()

# In[7]:
haha = 0
filename= 'loading11-pp2-gWide-zeroB-2-95-3-35-251points.mat'
fVals, dat, xFrom, xTo, X, Y, npoints= read_matfile(filename, normalize= False)
img= dat[:,:,10].copy()
##
##
## In[8]:
##
##row= int(X/2)
##col= int(Y/2)
## plot results
##fig, ax= plt.subplots(nrows= 1, ncols= 2, figsize= (10,4))
##img= np.squeeze(np.copy(dat[:,:,10]))
## img[row-2:row+2,col-2:col+2]= np.nan
##ax[0].imshow(img);
##
##yVals= np.squeeze(dat[row, col,:])
##ax[1].plot(fVals, yVals);
##pOpt, pCov= fit_data(fVals,yVals, init_params= generate_pinit([3.05,3.2], [ False, False]), fit_function= gauss_fit_c)
##plt.plot(fVals, gauss_fit_c(fVals, *pOpt))
##for i in np.arange(2):
##    params= pOpt[1+3*i:4+3*i]
##    plt.plot(fVals,pOpt[0]+gaussian_c(fVals,*params))
##
##plt.show()
##
##
## In[4]:
##
##seed_pOpts= generate_seed_parameters(dat, fVals, rowVals= np.arange(0,X,75), colVals= np.arange(0,Y,75))
##
##
## In[7]:
##
## datfit= np.empty((X,Y),dtype= object)
## boolfit= np.full((X,Y),False, dtype= bool)
##
##for i in range(0,X):
##    r= int(i/75)
##    for j in range(0,Y):
##        if boolfit[i,j]:
##            continue
##        yVals= np.squeeze(dat[i,j,:])
##        yVals= yVals/np.mean(yVals[-5:])
##        if np.any(np.isnan(yVals)):
##            print('Pixel:[%i,%i] is skipped'%(i,j))
##            continue
##        c= int(j/75)
##        p_init= list(seed_pOpts[r,c])
##        try:
##            pOpt, pCov= fit_data(fVals,yVals, init_params= p_init, fit_function= gauss_fit_c)
##            datfit[i,j]= {'pOpt': pOpt, 'pCov': pCov}
##            boolfit[i,j]= True
##        except sp.optimize.OptimizeWarning:
##            print('Pixel:[%i,%i] OptimizeWarfning'%(i,j))
##            continue
##        except RuntimeError:
##            print('Pixel:[%i,%i] encountered RuntimeError'%(i,j))
##            continue
##
##np.save(filename.replace('.mat','_datfit.npy'),datfit)
##np.save(filename.replace('.mat','_boolfit.npy'),boolfit)
##
##
## In[22]:
##
##datfit= np.load(filename.replace('.mat','_datfit.npy'),allow_pickle=True)
##boolfit= np.load(filename.replace('.mat','_boolfit.npy'),allow_pickle=True)
##
##f1, f2= getFitInfo(datfit, numpeaks= 2, param= 'frequency')
##fig, ax= plt.subplots(nrows=1, ncols= 2, figsize= (15, 6))
##plt.sca(ax[0])
##plt.imshow(f1,vmax= 3.02, vmin= 3.09)
##plt.colorbar()
##plt.title('Shift [MHz]', fontsize= 20)
##
##plt.sca(ax[1])
##
##plt.imshow(f2,vmax= 3.15, vmin= 3.25)
##plt.colorbar()
##plt.title('Splitting [MHz]', fontsize= 20)
##
##
## In[62]:
##
## masked_array = np.ma.array (a, mask=np.isnan(a))
##
##shift= (0.5*(f1+f2)-2.87)*1e3
##splitting= (f2-f1)*1e3
##res= vget_sigma(shift, splitting)
##sig_p,sig_zz= res
##sig_p= np.abs(sig_p)
##import matplotlib as mpl
##
##cmap = mpl.cm.get_cmap("viridis").copy()
##cmap.set_bad('black',1.)
##mpl.rcParams['font.family']= 'Arial'
##img_show= img.copy()
##cmap2= mpl.cm.get_cmap("viridis")
##low= 180
##high= low+10*512//200
##img_show[200:202,low:high]= np.nan
##fig,ax= plt.subplots(1,3, figsize= (14,4))
##plt.sca(ax[0])
##
##plt.imshow(img_show[:220,:225], vmax= 17e3,vmin= 0, cmap= cmap2)
## plt.colorbar()
##plt.sca(ax[1])
##sig_p= np.where(sig_p> 6, np.nan, sig_p)
##plt.imshow(np.ma.array (sig_p, mask=np.isnan(sig_p))[:220,:225], vmax= 5, vmin=0, cmap= cmap)
##cbar= plt.colorbar()
##cbar.ax.tick_params(labelsize= 20)
## cbar.ax.set_ylabel('GPa', fontsize= 20)
##plt.title('$\sigma_{\perp}$ [GPa]', fontsize= 25)
##plt.sca(ax[2])
##sig_zz= np.where(sig_zz>60, np.nan, sig_zz)
##plt.imshow(np.ma.array (sig_zz, mask=np.isnan(sig_zz))[:220,:225], vmin= 45, vmax= 55, cmap= cmap)
##cbar= plt.colorbar()
##cbar.ax.tick_params(labelsize= 20)
##plt.title('$\sigma_{zz}$ [GPa]', fontsize= 25)
## cbar.ax.set_ylabel('GPa', fontsize= 20)
##for axis in ax:
##    plt.sca(axis)
##    plt.xticks([])
##    plt.yticks([])
##plt.tight_layout()
##plt.tight_layout()
##plt.savefig('/Users/prabudhya/Pictures/l11_pp2.pdf', dpi= 1000, transparent= True)
##
##
## ### Pressure point 3
##
## In[63]:
##
##filename= 'loading11-pp3-gWide-zeroB-3-1-3-45-251points.mat'
##fVals, dat, xFrom, xTo, X, Y, npoints= read_matfile(filename, normal= False)
##
##
## In[64]:
##
##dat= normalize_widefield(dat)
##
##
## In[65]:
##
##row= int(X/2)
##col= int(Y/2)
## plot results
##fig, ax= plt.subplots(nrows= 1, ncols= 2, figsize= (10,4))
##img= np.squeeze(np.copy(dat[:,:,10]))
## img[row-2:row+2,col-2:col+2]= np.nan
##ax[0].imshow(img);
##
##yVals= np.squeeze(dat[row, col,:])
##ax[1].plot(fVals, yVals);
##pOpt, pCov= fit_data(fVals,yVals, init_params= generate_pinit([3.18, 3.35], [ False, False]), fit_function= gauss_fit_c)
##plt.plot(fVals, gauss_fit_c(fVals, *pOpt))
##for i in np.arange(2):
##    params= pOpt[1+3*i:4+3*i]
##    plt.plot(fVals,pOpt[0]+gaussian_c(fVals,*params))
##
##plt.show()
##
##
## In[66]:
##
##seed_pOpts= generate_seed_parameters(dat, fVals, rowVals= np.arange(0,X,75), colVals= np.arange(0,Y,75))
##
##
## In[69]:
##
## datfit= np.empty((X,Y),dtype= object)
## boolfit= np.full((X,Y),False, dtype= bool)
##
## for i in range(0,X):
##     r= int(i/75)
##     for j in range(0,Y):
##         if boolfit[i,j]:
##             continue
##         yVals= np.squeeze(dat[i,j,:])
##         yVals= yVals/np.mean(yVals[-5:])
##         if np.any(np.isnan(yVals)):
##             print('Pixel:[%i,%i] is skipped'%(i,j))
##             continue
##         c= int(j/75)
##         p_init= list(seed_pOpts[r,c])
##         try:
##             pOpt, pCov= fit_data(fVals,yVals, init_params= p_init, fit_function= gauss_fit_c)
##             datfit[i,j]= {'pOpt': pOpt, 'pCov': pCov}
##             boolfit[i,j]= True
##         except sp.optimize.OptimizeWarning:
##             print('Pixel:[%i,%i] OptimizeWarfning'%(i,j))
##             continue
##         except RuntimeError:
##             print('Pixel:[%i,%i] encountered RuntimeError'%(i,j))
##             continue
##
##np.save(filename.replace('.mat','_datfit.npy'),datfit)
##np.save(filename.replace('.mat','_boolfit.npy'),boolfit)
##
##
## In[71]:
##
##datfit= np.load(filename.replace('.mat','_datfit.npy'),allow_pickle=True)
##boolfit= np.load(filename.replace('.mat','_boolfit.npy'),allow_pickle=True)
##
##f1, f2= getFitInfo(datfit, numpeaks= 2, param= 'frequency')
##fig, ax= plt.subplots(nrows=1, ncols= 2, figsize= (15, 6))
##plt.sca(ax[0])
##plt.imshow(f1,vmax= 3.16, vmin= 3.2)
##plt.colorbar()
##plt.title('Shift [MHz]', fontsize= 20)
##
##plt.sca(ax[1])
##
##plt.imshow(f2,vmax= 3.31, vmin= 3.37)
##plt.colorbar()
##plt.title('Splitting [MHz]', fontsize= 20)
##
##
## ### Pressure point 4
##
## In[73]:
##
##filename= 'loading11-pp4-gWide-zeroB-3-15-3-55-251points.mat'
##fVals, dat, xFrom, xTo, X, Y, npoints= read_matfile(filename, normal= False)
##
##
## In[74]:
##
##row= int(X/2)
##col= int(Y/2)
## plot results
##fig, ax= plt.subplots(nrows= 1, ncols= 2, figsize= (10,4))
##img= np.squeeze(np.copy(dat[:,:,10]))
## img[row-2:row+2,col-2:col+2]= np.nan
##ax[0].imshow(img);
##
##yVals= np.squeeze(dat[row, col,:])
##ax[1].plot(fVals, yVals);
##pOpt, pCov= fit_data(fVals,yVals, init_params= generate_pinit([3.25, 3.42], [ False, False]), fit_function= gauss_fit_c)
##plt.plot(fVals, gauss_fit_c(fVals, *pOpt))
##for i in np.arange(2):
##    params= pOpt[1+3*i:4+3*i]
##    plt.plot(fVals,pOpt[0]+gaussian_c(fVals,*params))
##
##plt.show()
##
##
## In[75]:
##
##seed_pOpts= generate_seed_parameters(dat, fVals, rowVals= np.arange(0,X,75), colVals= np.arange(0,Y,75))
##
##
## In[78]:
##
## datfit= np.empty((X,Y),dtype= object)
## boolfit= np.full((X,Y),False, dtype= bool)
##
## for i in range(0,X):
##     r= int(i/75)
##     for j in range(0,Y):
##         if boolfit[i,j]:
##             continue
##         yVals= np.squeeze(dat[i,j,:])
##         yVals= yVals/np.mean(yVals[-5:])
##         if np.any(np.isnan(yVals)):
##             print('Pixel:[%i,%i] is skipped'%(i,j))
##             continue
##         c= int(j/75)
##         p_init= list(seed_pOpts[r,c])
##         try:
##             pOpt, pCov= fit_data(fVals,yVals, init_params= p_init, fit_function= gauss_fit_c)
##             datfit[i,j]= {'pOpt': pOpt, 'pCov': pCov}
##             boolfit[i,j]= True
##         except sp.optimize.OptimizeWarning:
##             print('Pixel:[%i,%i] OptimizeWarfning'%(i,j))
##             continue
##         except RuntimeError:
##             print('Pixel:[%i,%i] encountered RuntimeError'%(i,j))
##             continue
##
##np.save(filename.replace('.mat','_datfit.npy'),datfit)
##np.save(filename.replace('.mat','_boolfit.npy'),boolfit)
##
##
## In[77]:
##
## datfit= np.load(filename.replace('.mat','_datfit.npy'),allow_pickle=True)
## boolfit= np.load(filename.replace('.mat','_boolfit.npy'),allow_pickle=True)
##
##f1, f2= getFitInfo(datfit, numpeaks= 2, param= 'frequency')
##fig, ax= plt.subplots(nrows=1, ncols= 2, figsize= (15, 6))
##plt.sca(ax[0])
##plt.imshow(f1,vmax= 3.2, vmin= 3.28)
##plt.colorbar()
##plt.title('Shift [MHz]', fontsize= 20)
##
##plt.sca(ax[1])
##
##plt.imshow(f2,vmax= 3.4, vmin= 3.45)
##plt.colorbar()
##plt.title('Splitting [MHz]', fontsize= 20)
##
##
## ### Pressure point 5
##
## In[104]:
##
##filename= 'loading11-pp5-gWide-zeroB-3-3-3-7-201points.mat'
##fVals, dat, xFrom, xTo, X, Y, npoints= read_matfile(filename,Nx= 8,Ny= 8, Nf= 5)
##
##
## In[112]:
##
##row= int(X/2)
##col= int(Y/2)
## plot results
##fig, ax= plt.subplots(nrows= 1, ncols= 2, figsize= (10,4))
##img= np.squeeze(np.copy(dat[:,:,10]))
##img[row-2:row+2,col-2:col+2]= np.nan
##ax[0].imshow(img);
##
##yVals= np.squeeze(dat[row, col,:])
##ax[1].plot(fVals, yVals);
##pOpt, pCov= fit_data(fVals,yVals, init_params= generate_pinit([3.33, 3.55], [True, False]), fit_function= gauss_fit_c)
##plt.plot(fVals, gauss_fit_c(fVals, *pOpt))
##for i in np.arange(2):
##    params= pOpt[1+3*i:4+3*i]
##    plt.plot(fVals,pOpt[0]+gaussian_c(fVals,*params))
##
##plt.show()
##
##
## In[92]:
##
##seed_pOpts= generate_seed_parameters(dat, fVals, rowVals= np.arange(0,X,100), colVals= np.arange(0,Y,100))
##
##
## In[105]:
##
##seed_pOpts
##
##
## In[101]:
##
##A,B= seed_pOpts.shape
##for i in range(A):
##    for j in range(B):
##        seed_pOpts[i,j][1]= +1e-3
##
##
## In[108]:
##
## datfit= np.empty((X,Y),dtype= object)
## boolfit= np.full((X,Y),False, dtype= bool)
##
## for i in range(0,X):
##     r= int(i/100)
##     for j in range(0,Y):
##         if boolfit[i,j]:
##             continue
##         yVals= np.squeeze(dat[i,j,:])
##         yVals= yVals/np.mean(yVals[-5:])
##         if np.any(np.isnan(yVals)):
##             print('Pixel:[%i,%i] is skipped'%(i,j))
##             continue
##         c= int(j/100)
##         p_init= list(seed_pOpts[r,c])
##         try:
##             pOpt, pCov= fit_data(fVals,yVals, init_params= p_init, fit_function= gauss_fit_c)
##             datfit[i,j]= {'pOpt': pOpt, 'pCov': pCov}
##             boolfit[i,j]= True
##         except sp.optimize.OptimizeWarning:
##             print('Pixel:[%i,%i] OptimizeWarfning'%(i,j))
##             continue
##         except RuntimeError:
##             print('Pixel:[%i,%i] encountered RuntimeError'%(i,j))
##             continue
##
##np.save(filename.replace('.mat','_datfit.npy'),datfit)
##np.save(filename.replace('.mat','_boolfit.npy'),boolfit)
##
##
## In[109]:
##
##datfit= np.load(filename.replace('.mat','_datfit.npy'),allow_pickle=True)
##boolfit= np.load(filename.replace('.mat','_boolfit.npy'),allow_pickle=True)
##
##f1, f2= getFitInfo(datfit, numpeaks= 2, param= 'frequency')
##fig, ax= plt.subplots(nrows=1, ncols= 2, figsize= (15, 6))
##plt.sca(ax[0])
##plt.imshow(f1,vmax= 3.25, vmin= 3.4)
##plt.colorbar()
##plt.title('Shift [MHz]', fontsize= 20)
##
##plt.sca(ax[1])
##
##plt.imshow(f2,vmax= 3.5, vmin= 3.6)
##plt.colorbar()
##plt.title('Splitting [MHz]', fontsize= 20)
##
##
## ### Pressure point 6
##
## In[113]:
##
##filename= 'loading11-pp6-gWide-zeroB-3-35-3-85-301points.mat'
##fVals, dat, xFrom, xTo, X, Y, npoints= read_matfile(filename, normal=False)
##
##
## In[114]:
##
##dat= dat[:,:,10:]
##fVals= fVals[10:]
##
##row= int(X/2)
##col= int(Y/2)
## plot results
##fig, ax= plt.subplots(nrows= 1, ncols= 2, figsize= (10,4))
##img= np.squeeze(np.copy(dat[:,:,10]))
##img[row-2:row+2,col-2:col+2]= np.nan
##ax[0].imshow(img);
##
##yVals= np.squeeze(dat[row, col,:])
##ax[1].plot(fVals, yVals);
##pOpt, pCov= fit_data(fVals,yVals, init_params= generate_pinit([3.42, 3.65], [True, False]), fit_function= gauss_fit_c)
##plt.plot(fVals, gauss_fit_c(fVals, *pOpt))
##for i in np.arange(2):
##    params= pOpt[1+3*i:4+3*i]
##    plt.plot(fVals,pOpt[0]+gaussian_c(fVals,*params))
##
##plt.show()
##
##
## In[115]:
##
##seed_pOpts= generate_seed_parameters(dat, fVals, rowVals= np.arange(0,X,100), colVals= np.arange(0,Y,100))
##
##
## In[116]:
##
##A,B= seed_pOpts.shape
##for i in range(A):
##    for j in range(B):
##        seed_pOpts[i,j][1]= +1e-3
##
##
## In[120]:
##
## datfit= np.empty((X,Y),dtype= object)
## boolfit= np.full((X,Y),False, dtype= bool)
##
## for i in range(0,X):
##     r= int(i/100)
##     for j in range(0,Y):
##         if boolfit[i,j]:
##             continue
##         yVals= np.squeeze(dat[i,j,:])
##         yVals= yVals/np.mean(yVals[-5:])
##         if np.any(np.isnan(yVals)):
##             print('Pixel:[%i,%i] is skipped'%(i,j))
##             continue
##         c= int(j/100)
##         p_init= list(seed_pOpts[r,c])
##         try:
##             pOpt, pCov= fit_data(fVals,yVals, init_params= p_init, fit_function= gauss_fit_c)
##             datfit[i,j]= {'pOpt': pOpt, 'pCov': pCov}
##             boolfit[i,j]= True
##         except sp.optimize.OptimizeWarning:
##             print('Pixel:[%i,%i] OptimizeWarfning'%(i,j))
##             continue
##         except RuntimeError:
##             print('Pixel:[%i,%i] encountered RuntimeError'%(i,j))
##             continue
##
##np.save(filename.replace('.mat','_datfit.npy'),datfit)
##np.save(filename.replace('.mat','_boolfit.npy'),boolfit)
##
##
## In[121]:
##
##datfit= np.load(filename.replace('.mat','_datfit.npy'),allow_pickle=True)
##boolfit= np.load(filename.replace('.mat','_boolfit.npy'),allow_pickle=True)
##
##f1, f2= getFitInfo(datfit, numpeaks= 2, param= 'frequency')
##fig, ax= plt.subplots(nrows=1, ncols= 2, figsize= (15, 6))
##plt.sca(ax[0])
##plt.imshow(f1,vmax= 3.4, vmin= 3.5)
##plt.colorbar()
##plt.title('Shift [MHz]', fontsize= 20)
##
##plt.sca(ax[1])
##
##plt.imshow(f2,vmax= 3.6, vmin= 3.7)
##plt.colorbar()
##plt.title('Splitting [MHz]', fontsize= 20)
##
##
## ### Pressure point 7
##
## In[122]:
##
##filename= 'loading11-pp7-gWide-3-4-251points-zeroB.mat'
##fVals, dat, xFrom, xTo, X, Y, npoints= read_matfile(filename, normal= False)
##
##
## In[130]:
##
##dat= dat[:, :, 10:]
##fVals= fVals[10:]
##
##row= int(X/2)
##col= int(Y/2)
## plot results
##fig, ax= plt.subplots(nrows= 1, ncols= 2, figsize= (10,4))
##img= np.squeeze(np.copy(dat[:,:,10]))
##img[row-2:row+2,col-2:col+2]= np.nan
##ax[0].imshow(img);
##
##yVals= np.squeeze(dat[row, col,:])
##ax[1].plot(fVals, yVals);
##pOpt, pCov= fit_data(fVals,yVals, init_params= generate_pinit([3.58, 3.78], [True, False]), fit_function= gauss_fit_c)
##plt.plot(fVals, gauss_fit_c(fVals, *pOpt))
##for i in np.arange(2):
##    params= pOpt[1+3*i:4+3*i]
##    plt.plot(fVals,pOpt[0]+gaussian_c(fVals,*params))
##
##plt.show()
##
##
## In[131]:
##
##seed_pOpts= generate_seed_parameters(dat, fVals, rowVals= np.arange(0,X,100), colVals= np.arange(0,Y,100))
##
##
## In[132]:
##
##A,B= seed_pOpts.shape
##for i in range(A):
##    for j in range(B):
##        seed_pOpts[i,j][1]= +1e-3
##
##
## In[135]:
##
##datfit= np.empty((X,Y),dtype= object)
##boolfit= np.full((X,Y),False, dtype= bool)
##
##for i in range(0,X):
##    r= int(i/100)
##    for j in range(0,Y):
##        if boolfit[i,j]:
##            continue
##        yVals= np.squeeze(dat[i,j,:])
##        yVals= yVals/np.mean(yVals[-5:])
##        if np.any(np.isnan(yVals)):
##            print('Pixel:[%i,%i] is skipped'%(i,j))
##            continue
##        c= int(j/100)
##        p_init= list(seed_pOpts[r,c])
##        try:
##            pOpt, pCov= fit_data(fVals,yVals, init_params= p_init, fit_function= gauss_fit_c)
##            datfit[i,j]= {'pOpt': pOpt, 'pCov': pCov}
##            boolfit[i,j]= True
##        except sp.optimize.OptimizeWarning:
##            print('Pixel:[%i,%i] OptimizeWarfning'%(i,j))
##            continue
##        except RuntimeError:
##            print('Pixel:[%i,%i] encountered RuntimeError'%(i,j))
##            continue
##
##np.save(filename.replace('.mat','_datfit.npy'),datfit)
##np.save(filename.replace('.mat','_boolfit.npy'),boolfit)
##
##
## In[136]:
##
##datfit= np.load(filename.replace('.mat','_datfit.npy'),allow_pickle=True)
##boolfit= np.load(filename.replace('.mat','_boolfit.npy'),allow_pickle=True)
##
##f1, f2= getFitInfo(datfit, numpeaks= 2, param= 'frequency')
##fig, ax= plt.subplots(nrows=1, ncols= 2, figsize= (15, 6))
##plt.sca(ax[0])
##plt.imshow(f1,vmax= 3.4, vmin= 3.6)
##plt.colorbar()
##plt.title('Shift [MHz]', fontsize= 20)
##
##plt.sca(ax[1])
##
##plt.imshow(f2,vmax= 3.75, vmin= 3.85)
##plt.colorbar()
##plt.title('Splitting [MHz]', fontsize= 20)
##
##
## ### Pressure point 8
##
## In[157]:
##
##for i,filename in enumerate(['./loading11-pp8-gWdie-zeroB-attempt2.mat',
##                 './loading11-pp8-gWdie-zeroB-attempt3.mat']):
##    if i==1: 
##        fVals, data, xFrom, xTo, X, Y, npoints= read_matfile(filename,Nx= 8, Ny= 8, Nf= 8,  normal= False)
##    else: 
##        fVals, dat, xFrom, xTo, X, Y, npoints= read_matfile(filename,Nx= 8, Ny= 8, Nf= 8, normal= False)
##
##
## In[158]:
##
##res= dat+ data
##
##
## In[159]:
##
##dat= res
##
##
## In[161]:
##
##dat= dat[:, :, 10:]
##fVals= fVals[10:]
##
##row= int(X/2)
##col= int(Y/2)
## plot results
##fig, ax= plt.subplots(nrows= 1, ncols= 2, figsize= (10,4))
##img= np.squeeze(np.copy(dat[:,:,10]))
##img[row-2:row+2,col-2:col+2]= np.nan
##ax[0].imshow(img);
##
##yVals= np.squeeze(dat[row, col,:])
##ax[1].plot(fVals, yVals);
##pOpt, pCov= fit_data(fVals,yVals, init_params= generate_pinit([3.63, 3.88], [True, False]), fit_function= gauss_fit_c)
##plt.plot(fVals, gauss_fit_c(fVals, *pOpt))
##for i in np.arange(2):
##    params= pOpt[1+3*i:4+3*i]
##    plt.plot(fVals,pOpt[0]+gaussian_c(fVals,*params))
##
##plt.show()
##
##
## In[162]:
##
##xmin,xmax,xstep= (0,X, 50)
##ymin,ymax,ystep= (0,Y,50)
##seed_pOpts= generate_seed_parameters(dat, fVals, rowVals= np.arange(xmin,xmax,xstep), colVals= np.arange(ymin,ymax,ystep))
##
##
## In[163]:
##
##A,B= seed_pOpts.shape
##for i in range(A):
##    for j in range(B):
##        seed_pOpts[i,j][1]= +1e-3
##
##
## In[164]:
##
##datfit= np.empty((X,Y),dtype= object)
##boolfit= np.full((X,Y),False, dtype= bool)
##
##for i in range(xmin,xmax):
##    r= int((i-xmin)/xstep)
##    for j in range(0,ymax):
##        if boolfit[i,j]:
##            continue
##        yVals= np.squeeze(dat[i,j,:])
##        yVals= yVals/np.mean(yVals[-5:])
##        if np.any(np.isnan(yVals)):
##            print('Pixel:[%i,%i] is skipped'%(i,j))
##            continue
##        c= int((j-ymin)/ystep)
##        p_init= list(seed_pOpts[r,c])
##        try:
##            pOpt, pCov= fit_data(fVals,yVals, init_params= p_init, fit_function= gauss_fit_c)
##            datfit[i,j]= {'pOpt': pOpt, 'pCov': pCov}
##            boolfit[i,j]= True
##        except sp.optimize.OptimizeWarning:
##            print('Pixel:[%i,%i] OptimizeWarfning'%(i,j))
##            continue
##        except RuntimeError:
##            print('Pixel:[%i,%i] encountered RuntimeError'%(i,j))
##            continue
##
##np.save(filename.replace('.mat','_datfit.npy'),datfit)
##np.save(filename.replace('.mat','_boolfit.npy'),boolfit)
##
##
## In[166]:
##
##datfit= np.load(filename.replace('.mat','_datfit.npy'),allow_pickle=True)
##boolfit= np.load(filename.replace('.mat','_boolfit.npy'),allow_pickle=True)
##
##f1, f2= getFitInfo(datfit, numpeaks= 2, param= 'frequency')
##fig, ax= plt.subplots(nrows=1, ncols= 2, figsize= (15, 6))
##plt.sca(ax[0])
##plt.imshow(f1,vmax= 3.5, vmin= 3.7)
##plt.colorbar()
##plt.title('Shift [MHz]', fontsize= 20)
##
##plt.sca(ax[1])
##
##plt.imshow(f2,vmax= 3.85, vmin= 3.95)
##plt.colorbar()
##plt.title('Splitting [MHz]', fontsize= 20)
##
##
## ### Pressure point 9
##
## ### Pressure point 10
##
## In[ ]:
##
##
##
##
## In[41]:
##
##pOpt[1+3:4+3]


# %%
