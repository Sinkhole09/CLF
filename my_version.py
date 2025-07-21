# import os
# import netCDF4
# import csv
# import datetime
# import pandas as pd
# from matplotlib.animation import FuncAnimation
# import matplotlib as mpl
import scipy
from scipy.interpolate import RectBivariateSpline
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

try:
    from IPython import get_ipython  #checks if the code is running on an IPython environment
    ip = get_ipython()              #tries to retrive the interactive python shell object
except:
    pass

try:
    type(once) #checking to see if a global variable called once is defined
except: # defining some vars in a global scope
    once       = False
    twice      = 1
    source_dir = 'C:/Users/shrut/OneDrive - Imperial College London/Documents/Finding Research/UKRI_STFC_CLF/'
    fmat       = scipy.io.loadmat(  # read a MATLAB file
        source_dir + 'for_shrut_250715/dppfilter_06_60_8col.mat',
        struct_as_record=False,#convert MATLAB structures into Python sytle classes with attributes
        squeeze_me=True) # squeeze out singleton dimensions like 1x1 arrays
    res_fac_pr = -1

if twice <= 2:
    twice+=1
    # ip.magic("matplotlib") #run the magic command matplotlib. This configures matplotlib backend
plt.close('all')

def zcen(ar, axis=0):
    """
    Computes zone-centered (cell-centered) values of an array along a given axis.
    
    Parameters:
        ar : np.ndarray
            Input array (1D, 2D, or 3D).
        axis : int, optional
            The axis along which to compute centered values (default is 0).
    
    Returns:
        np.ndarray
            An array of shape reduced by 1 along the specified axis,
            where each value is the average of adjacent elements along that axis.
    
    Example:
        zcen([1, 2, 4]) → [1.5, 3.0], which is 0.5*(1+2), 0.5*(2+4)
    """
    import numpy as np

    ndims = len(ar.shape) # gets number of dimentions of input array

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

def create_2D_grids(KesslerParms, resolution_fac=1):
    nx             = KesslerParms.fftCells_x * resolution_fac # Number of pixels in the x-direction in the FFT grid (simulated near-field image)
    dx             = KesslerParms.nfPixSize_x / resolution_fac # Size of each pixel in x-direction at the near-field plane (where FFT is evaluated); I guess in meters?
    ny             = KesslerParms.fftCells_y * resolution_fac 
    dy             = KesslerParms.nfPixSize_y / resolution_fac # 0.0003125 
    x1d          = np.linspace(-1*nx/2,nx/2,nx) * dx
    y1d          = np.linspace(-1*ny/2,ny/2,ny) * dy
    x,y          = np.meshgrid(x1d,y1d,indexing='ij')

    return nx, dx, ny, dy, x1d, y1d, (x,y)
def make_big_figure(x, y, nx, ny, x1d, y1d, xff, yff, dxff, dyff, dpp_phase, int_beam_env, int_ff, pwr_ff_tot, near_field, cm, cm2, um):
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
    pcm=ax[iy,ix].pcolormesh(x1d/cm,y1d/cm,int_beam_env,cmap='rainbow', norm=LogNorm(vmin=1e-1))
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
    return fig
def normalize_2D_array(data_array, x_range, y_range):
    """"
    Uses integration to find normalizing factor for a 2D area by taking area under the graph. 
    """
    dx = x_range[1] - x_range[0]
    dy = y_range[1] - y_range[0]

    normalizing_factor = np.sum(data_array) * dx * dy
    return data_array / normalizing_factor
if __name__ == "__main__":
    #unit conversions
    m  = 1
    cm = 1e-2
    um = 1e-6
    cm2= 1e-4

    beam_power       = 0.5e12        #Watts     # in "Initial...restuls...Omega..."~Boehly et. all, it says 60TW 60-beam
    foc_len          = 1.8*m         #Omega final lens focal distance
    f_number         = 6.5           #Omega's f number
    Lambda           = 0.351*um      #Omega wavelength: 3rd harmonic of a Nd:Glass laser
    res_fac          = 1             #Note: Higher resolution is required to resolve the far-field speckles

    #if res_fac != res_fac_pr:
    dpp_phase    = fmat['DPP_phase'] #The actual phase mask of the diffractive plate in radians over a 2D grid
    KesslerParms = fmat['KesslerParms']
    nx, dx, ny, dy, x1d, y1d, (x,y) = create_2D_grids(KesslerParms)
    #fine grid for high-res interpolation
    nx_, dx_, ny_, dy_, x1d_, y1d_, (x_,y_) = create_2D_grids(KesslerParms, resolution_fac=res_fac)
    phase_func   = RectBivariateSpline(x1d,y1d,dpp_phase) #interpolate the phase array from the phase plate
    dpp_phase    = phase_func(x1d_,y1d_)
#this seems like it could be simplified, but I will leave it for now
    nx           = nx_  
    dx           = dx_  
    ny           = ny_  
    dy           = dy_  
    x1d          = x1d_ 
    y1d          = y1d_ 
    x,y          = x_,y_
    res_fac_pr   = res_fac
        
    int_beam_env     = np.exp(np.log(0.5)*(2*np.sqrt(x**2+y**2)/(25.8*cm))**24) #Beam intensity envelope at lens
                            #where does this equation come from?
    area_nf          = dx * dy
    pwr_beam_env     = int_beam_env * area_nf #power  envelope at lens
    
    #old approach to normalizing:
    pwr_beam_env_tot = np.sum(pwr_beam_env)
    pwr_beam_env    *= beam_power / pwr_beam_env_tot    #W
                    #what is happeneing here?
    pwr_beam_env_tot = np.sum(pwr_beam_env)
    
    #new appraoch to normalizing:
    # pwr_beam_env = normalize_2D_array(pwr_beam_env, x1d, y1d) * beam_power
    # pwr_beam_env_tot = np.sum(pwr_beam_env)

    int_beam_env     = pwr_beam_env / area_nf #W/m^2
                        #I guess this is the scaled intensity?
    efield_beam_env  = int_beam_env**0.5    #V/m
                       #proportionality relation. E field amplitude squared is proportional to intensity
    dpp              = np.exp(-1j * dpp_phase) #complex number of unit modulus. Product simply changes phase
    near_field       = efield_beam_env * dpp                   #apply DPP phase modulation
    #does efield_beam_env store complex numbers? Is it a wave? What the heck happens when you give it a phase?!?!
    far_field        = np.fft.fftshift(np.fft.fft2(near_field))#2D fft then shift zero-frequency component to the center of the spectrum
    int_ff           = np.abs(far_field)**2 #intensity isproportional to amplitude squared

    #Not sure about the rationale for this spatial re-scaling
    dxff, dyff = foc_len * Lambda / (dx * nx), foc_len * Lambda / (dy * ny) #both values are in meters
                    #dx * nx gives total simulated distance at near field in meters
                    #foc_len / (dx * nx) is supposed to be f_number but it isn't
                    #conversion from near field to far field
    # dxff, dyff             = f_number * Lambda, f_number * Lambda
    xff1d            = np.linspace( -1 * nx / 2, nx / 2, nx) * dxff #this multiplication happens to convert from pixels to meters
    yff1d            = np.linspace(-1 * ny / 2, ny / 2, nx) * dyff
    xff,yff          = np.meshgrid(xff1d, yff1d, indexing='ij')

    print(f"""Far field grid scales area: {dxff/um:2.2f} microns by {dyff/um:2.2f} microns, speckle scale is {f_number*Lambda/um:2.2f} microns""")

    area_ff          = dxff * dyff
    pwr_ff           = int_ff * area_ff

    #The FFT isn't conserving energy (I'm not sure it should) so re-normlising here!
    pwr_ff          *= beam_power/np.sum(pwr_ff)
    pwr_ff_tot       = np.sum(pwr_ff)
    int_ff           = pwr_ff / area_ff
    # xff1d           /= area_ff_renorm**0.5
    # yff1d           /= area_ff_renorm**0.5

    #not manually renormalizing:
    # pwr_ff_tot       = np.sum(pwr_ff)

    # del int_beam_env, pwr_beam_env, int_beam_env, efield_beam_env, dpp, near_field, far_field

    print(f"""Power in near field {pwr_beam_env_tot*1e-12:2.2f} TW, power in far field = {pwr_ff_tot*1e-12:2.2f}TW,
intensity in near field {np.sum(int_beam_env) * 1e-12:2.2f} TW/m^2, intensity in far field = {np.sum(int_ff):2.2f} idk_the_unit/m^2
area_nearfield: {area_nf:2.2e}, area_farfield: {area_ff:2.2e}
grid scale near field: {nx*dx:2.2f}m by {ny*dy:2.2f}m, focal length: {foc_len}m
f_num_x = {foc_len/(nx*dx)}, f_num_y = {foc_len/(ny*dy)}, f_number: {f_number}""")
    # fig = make_big_figure(x, y, nx, ny, x1d, y1d, xff, yff, dxff, dyff, dpp_phase, int_beam_env, int_ff, pwr_ff_tot, near_field, cm, cm2, um)
    plt.show()
