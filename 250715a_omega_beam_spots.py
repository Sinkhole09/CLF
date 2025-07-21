import scipy
from scipy.interpolate import RectBivariateSpline
import os
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import netCDF4
import csv
import datetime
import pandas as pd

try:
    from IPython import get_ipython
    ip = get_ipython()
except:
    pass

try:
    type(once)
except:
    once       = False
    twice      = 1
    fmat       = scipy.io.loadmat('/Users/rscott/python/beam_spots/matlab/dppfilter_06_60_8col.mat')
    res_fac_pr = -1
    
if twice <= 2:
    twice+=1
    ip.magic("matplotlib")
plt.close('all')

def zcen(ar,axis=[]):
    """
    The pairwise average between successive elements along the specified axis.
    Returns the zone (or cell) centred value of an array axong a given axis.
    The returned axis is one shorter than the initial axis
    If axis is not given, axis = 0 
    """
    import numpy as np

    ndims = len(ar.shape)
    if type(axis) != int:
        axis = 0

    if ndims ==1:        
        axis = 0
        zcen = ar[0:-1]+0.5*np.diff(ar,axis=axis)
    elif ndims ==2:        
        if axis==0:
            zcen = ar[0:-1,  :  ]+0.5*np.diff(ar,axis=axis)
        elif axis ==1:
            zcen = ar[:   , 0:-1]+0.5*np.diff(ar,axis=axis)
    elif ndims ==3:        
        if axis==0:
            zcen = ar[0:-1,  :  , :  ]+0.5*np.diff(ar,axis=axis)
        elif axis ==1:
            zcen = ar[ :  , 0:-1, :  ]+0.5*np.diff(ar,axis=axis)
        elif axis ==2:
            zcen = ar[ :  ,  :  ,0:-1]+0.5*np.diff(ar,axis=axis)

    return zcen    
def smooth(ar,nsmooth):
    '''
    Simple smoothing of a 1d array, using pcen and zcen recursively.
    '''
    import pyh2d as h2d
    for n in range(nsmooth):
        ar = h2d.zcen(h2d.pcen(ar))
        n+=1
    return ar

m  = 1;
cm = 1e-2;
um = 1e-6;
cm2= 1e-4

beam_power       = 0.5e12        #Watts
foc_len          = 1.8*m         #Omega final lens focal distance
f_number         = 6.5           #Omega's f number
Lambda           = 0.351*um      #Omega 3w wavelength
res_fac          = 8             #Note: Higher resolution is required to resolve the far-field speckles

#if res_fac != res_fac_pr:
dpp_phase    = fmat['DPP_phase']
KesslerParms = fmat['KesslerParms']
nx           = KesslerParms['fftCells_x'][0][0][0][0] 
dx           = KesslerParms['nfPixSize_x'][0][0][0][0]
ny           = KesslerParms['fftCells_y'][0][0][0][0] 
dy           = KesslerParms['nfPixSize_y'][0][0][0][0]
x1d          = np.linspace(-1*nx*dx/2,nx*dx/2,nx)
y1d          = np.linspace(-1*ny*dy/2,ny*dy/2,ny)
x,y          = np.meshgrid(x1d,y1d,indexing='ij')                       #create 2D grids

#fine grid for high-res interpolation
nx_          = KesslerParms['fftCells_x'][0][0][0][0]  * res_fac
dx_          = KesslerParms['nfPixSize_x'][0][0][0][0] / res_fac
ny_          = KesslerParms['fftCells_y'][0][0][0][0]  * res_fac
dy_          = KesslerParms['nfPixSize_y'][0][0][0][0] / res_fac
x1d_         = np.linspace(-1*nx_*dx_/2,nx_*dx_/2,nx_)
y1d_         = np.linspace(-1*ny_*dy_/2,ny_*dy_/2,ny_)
x_,y_        = np.meshgrid(x1d_,y1d_,indexing='ij') #create 2D grids
phase_func   = RectBivariateSpline(x1d,y1d,dpp_phase)
dpp_phase    = phase_func(x1d_,y1d_)
nx           = nx_  
dx           = dx_  
ny           = ny_  
dy           = dy_  
x1d          = x1d_ 
y1d          = y1d_ 
x,y          = x_,y_
res_fac_pr   = res_fac
    
int_beam_env     = np.exp(np.log(0.5)*(2*np.sqrt(x**2+y**2)/(25.8*cm))**24) #Beam intensity envelope at lens

area_nf          = dx * dy
pwr_beam_env     = int_beam_env * area_nf
pwr_beam_env_tot = np.sum(pwr_beam_env)
pwr_beam_env    *= beam_power / pwr_beam_env_tot    #W
pwr_beam_env_tot = np.sum(pwr_beam_env)
int_beam_env     = pwr_beam_env / area_nf #W/m^2 
efield_beam_env  = int_beam_env**0.5    #V/m

dpp              = np.exp(-1j * dpp_phase)
near_field       = efield_beam_env * dpp                   #apply DPP phase modulation
far_field        = np.fft.fftshift(np.fft.fft2(near_field))#2D fft then shift zero-frequency component to the center of the spectrum
int_ff           = np.abs(far_field)**2

#Not sure about the rationale for this spatial re-scaling
dxff             = foc_len*Lambda/(dx*nx)                             #x conversion from near field to far field
xff1d            = np.linspace(-1*nx/2,nx/2,nx)*dxff
dyff             = foc_len*Lambda/(dy*ny)                             #y conversion from near field to far field
yff1d            = np.linspace(-1*ny/2,ny/2,nx)*dyff
xff,yff          = np.meshgrid(xff1d,yff1d,indexing='ij')

print('Far field grid scales are %2.2f um x %2.2f um, speckle scale is %2.2fum'%(dxff/um,dyff/um,f_number*Lambda/um))
#The FFT isn't conserving energy (I'm not sure it should) so re-normlising here!
area_ff          = dxff * dyff
pwr_ff           = int_ff * area_ff
pwr_ff_tot       = np.sum(pwr_ff)
area_ff_renorm   = beam_power / pwr_ff_tot
area_ff         /= area_ff_renorm
int_ff           = pwr_ff / area_ff
#xff1d           /= area_ff_renorm**0.5
#yff1d           /= area_ff_renorm**0.5

pwr_ff_tot       = np.sum(int_ff * area_ff)

#del int_beam_env, pwr_beam_env, int_beam_env, efield_beam_env, dpp, near_field, far_field

print('Power in near field %2.2f TW, power in far field = %2.2f TW (renorm factor = %2.3f)'%(pwr_beam_env_tot*1e-12, pwr_ff_tot*1e-12,area_ff_renorm))
nrows=2;ncols=4;ix=-1;iy=0
fig, ax = plt.subplots(nrows, ncols,num=0,figsize=(ncols*8,nrows*6))

ix+=1;
if ix==ncols:ix=0;iy+=1
pcm=ax[iy,ix].pcolormesh(x/cm,y/cm,dpp_phase,cmap='rainbow')
fig.colorbar(pcm,ax=ax[iy,ix])
ax[iy,ix].set_xlabel('x (cm)')
ax[iy,ix].set_ylabel('y (cm)')
ax[iy,ix].set_title('Omega SG5 phase at lens')
ax[iy,ix].text(x=1.05,y=1.05,s='Radians',transform=ax[iy,ix].transAxes)

ix+=1;
if ix==ncols:ix=0;iy+=1
pcm=ax[iy,ix].pcolormesh(x1d/cm,y1d/cm,int_beam_env,cmap='rainbow')
fig.colorbar(pcm,ax=ax[iy,ix])
ax[iy,ix].set_xlabel('x (cm)')
ax[iy,ix].set_ylabel('y (cm)')
ax[iy,ix].set_title('Beam intensity envelope at lens')
ax[iy,ix].text(x=1.05,y=1.05,s='W/cm2',transform=ax[iy,ix].transAxes)

ix+=1;
if ix==ncols:ix=0;iy+=1
pcm=ax[iy,ix].pcolormesh(x1d/cm,y1d/cm,near_field.real,cmap='rainbow')
fig.colorbar(pcm,ax=ax[iy,ix])
ax[iy,ix].set_xlabel('x (cm)')
ax[iy,ix].set_ylabel('y (cm)')
ax[iy,ix].set_title('Real near field')

ix+=1;
if ix==ncols:ix=0;iy+=1
pcm=ax[iy,ix].pcolormesh(x1d/cm,y1d/cm,near_field.imag,cmap='rainbow')
fig.colorbar(pcm,ax=ax[iy,ix])
ax[iy,ix].set_xlabel('x (cm)')
ax[iy,ix].set_ylabel('y (cm)')
ax[iy,ix].set_title('Imag. near field')

ix+=1;
if ix==ncols:ix=0;iy+=1
pcm=ax[iy,ix].pcolormesh(xff/um,yff/um,np.log10(int_ff * cm2),cmap='rainbow')
fig.colorbar(pcm,ax=ax[iy,ix])
ax[iy,ix].set_xlabel('x (um)')
ax[iy,ix].set_ylabel('y (um)')
ax[iy,ix].set_title('Far field intensity: Log10 scale')
ax[iy,ix].set_xlim(-500,500)
ax[iy,ix].set_ylim(-500,500)
ax[iy,ix].text(x=1.05,y=1.05,s='log10(Wcm$^{-2}$)',transform=ax[iy,ix].transAxes)

ix+=1;
if ix==ncols:ix=0;iy+=1
pcm=ax[iy,ix].pcolormesh(xff/um,yff/um,int_ff * cm2,cmap='rainbow')
fig.colorbar(pcm,ax=ax[iy,ix])
ax[iy,ix].set_xlabel('x (um)')
ax[iy,ix].set_ylabel('y (um)')
ax[iy,ix].set_title('Far field:\nMax.  intensity = %2.1e Wcm$^{-2}$, Power = %2.2f TW'%(np.max(int_ff)*cm2,pwr_ff_tot*1e-12))
ax[iy,ix].set_xlim(-500,500)
ax[iy,ix].set_ylim(-500,500)
ax[iy,ix].text(x=1.05,y=1.05,s='Wcm$^{-2}$',transform=ax[iy,ix].transAxes)

ix+=1;
if ix==ncols:ix=0;iy+=1
di=2
iy_=int(ny/2)
ax[iy,ix].plot(xff[:,iy_]/um,np.average(int_ff[:,iy_-di:iy_+di],axis=1) * cm2,'-k')
ax[iy,ix].set_xlabel('x (um)')
ax[iy,ix].set_ylabel('Intensity (Wm$^{-2}$)')
ax[iy,ix].set_title('Far field intensity\n averaged over %2.2fum in y'%(2*di*dyff/um))
ax[iy,ix].set_xlim(-500,500)

ix+=1;
if ix==ncols:ix=0;iy+=1
ix_=int(nx/2)
ax[iy,ix].plot(np.average(int_ff[ix_-di:ix_+di,:],axis=0) * cm2,yff[ix_,:]/um)
ax[iy,ix].set_xlabel('Intensity (Wm$^{-2}$)')
ax[iy,ix].set_ylabel('y (um)')
ax[iy,ix].set_title('Far field intensity\n averaged over %2.2f um in x'%(2*di*dxff/um))
ax[iy,ix].set_ylim(-500,500)

plt.tight_layout()
plt.show()
