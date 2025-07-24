#%%
# import os
# import netCDF4
# import csv
# import datetime
# import pandas as pd
# from matplotlib.animation import FuncAnimation
# import matplotlib as mpl
from scipy.io import loadmat
from scipy.interpolate import RectBivariateSpline
import scipy.integrate as integrate
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

try:
    from IPython import get_ipython  #checks if the code is running on an IPython environment
    ip = get_ipython()               #tries to retrive the interactive python shell object
except:
    pass

try:
    type(once)          # checking to see if a global variable called once is defined
except:                 # defining some vars in a global scope
    once       = False
    twice      = 1
    source_dir = 'C:/Users/shrut/OneDrive - Imperial College London/Documents/Finding Research/UKRI_STFC_CLF/'
    fmat       = loadmat(                                           # read a MATLAB file
        source_dir + 'for_shrut_250715/dppfilter_06_60_8col.mat',
        struct_as_record=False,                                     # convert MATLAB structures into Python sytle classes with attributes
        squeeze_me=True)                                            # squeeze out singleton dimensions like 1x1 arrays
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
def beam_envelope(alpha, sigma, x, y, N):
        """
        np.exp(alpha * (np.sqrt (x ** 2 + y ** 2) / (sigma)) ** N)
        """
        envelope = np.exp(alpha*(np.sqrt(x**2+y**2)/(sigma))**N) #Beam intensity envelope at lens
        return envelope
def create_2D_grids(KesslerParms, resolution_fac=1):
    nx             = KesslerParms.fftCells_x * resolution_fac  # Number of pixels in the x-direction in the FFT grid (simulated near-field image)
    dx             = KesslerParms.nfPixSize_x / resolution_fac # Size of each pixel in x-direction at the near-field plane (where FFT is evaluated); I guess in meters?
    ny             = KesslerParms.fftCells_y * resolution_fac 
    dy             = KesslerParms.nfPixSize_y / resolution_fac # 0.0003125 (meters I guess?)
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
def normalize_2D_array(data_array, x_range, y_range, use_scipy=False):
    """"
    Uses integration to find normalizing factor for a 2D area by taking area under the graph. 
    """
    if use_scipy:                                                       # Integrate over y (axis=1), then over x (axis=0)
        integral_wrt_y = integrate.simpson(data_array, y_range, axis=1)
        normalizing_factor = integrate.simpson(integral_wrt_y, x_range)
    else:
        dx = x_range[1] - x_range[0]
        dy = y_range[1] - y_range[0]
        normalizing_factor = np.sum(data_array) * dx * dy
    return data_array / normalizing_factor, normalizing_factor
def expand_grid(mesh):
    """
    Take an input meshgrid and triple it's size by addings rows of zeros and columns of zeros to create a new meshgrid with the
    data of the input mesh at the center, and zeros surrounding it.
    """
    expanded_mesh = mesh
    rows, cols = len(mesh[:,0]), len(mesh[0,:])
    y_zeros = np.zeros((rows, cols))
    expanded_mesh = np.concatenate((y_zeros, expanded_mesh, y_zeros), axis=0)
    rows_new = len(expanded_mesh[:,0])
    x_zeros = np.zeros((rows_new, cols))
    expanded_mesh = np.concatenate((x_zeros, expanded_mesh, x_zeros),axis=1)
    print(f"{mesh}\n{expanded_mesh}")
    return expanded_mesh
def quantify_nonuniformity(int_ideal, int_nonuniform):
    rows, cols = np.shape(int_ideal)
    rms_nonuniformity_array = np.abs(int_ideal - int_nonuniform) ** 2
    return np.sqrt( np.sum(rms_nonuniformity_array) / (rows * cols))
def apply_polarisation_smoothing(int_ff, shift=4, conversion_factor=1):
    """
    Applies Polarisation smoothening to the far field intensity distribution. The components of beams that are polarised
    perpendicular to each other are horizontally offset by some amount. Alternative implementation can be to offset the
    beams horizontally, I don't know how the birefringent wedge works as of yet.
    """
    rows, cols = np.shape(int_ff)
    polarised_offset = np.zeros((rows, cols))
    polarised_offset[:,:-1*shift] += int_ff[:,shift:] / 2
    polarised_offset[:,shift:] += int_ff[:,:-1*shift] / 2

    return polarised_offset
def intensity_plot(x, xff, y, yff, int_nf_raw, int_ff_raw, do_PS=False, do_LogNorm=False):
  
    non_uniformity_noPS = quantify_nonuniformity(int_ideal=int_nf/cm2, int_nonuniform=int_ff/(um**2))
    non_uniformity_PS = quantify_nonuniformity(int_ideal=int_nf/cm2, int_nonuniform=int_ff_PS/(um**2))
    print(f"nonuniformity without PS: {non_uniformity_noPS:3.2e}, with PS: {non_uniformity_PS:3.2e}")
    return fig
def intensity_phase_plots(x, xff, y, yff, int_ff_ideal_raw, int_ff_raw, dpp):
    peak_int_ideal = np.max(int_ff_ideal_raw)
    int_ff_ideal = int_ff_ideal_raw / peak_int_ideal
    int_ff = int_ff_raw / peak_int_ideal
    data = [int_ff_ideal, dpp, int_ff]
    x_y = [(xff,yff), (x,y), (xff,yff)]
    unit = [('um', um), ('cm', cm), ('um', um)]
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(2*8,8))
    x_y = list(x_y)
    for pos, data_array in enumerate(data):
        x_val, y_val = x_y[pos]
        pcm = ax[pos].pcolormesh(x_val / unit[pos][1], y_val/ unit[pos][1], data_array,cmap='rainbow')
        fig.colorbar(pcm, ax=ax[pos])
        ax[pos].set_xlabel(f"x {unit[pos][0]}")
        ax[pos].set_ylabel(f"y {unit[pos][0]}")
        if pos == 2:
            ax[pos].set_xlim(-400,400)
            ax[pos].set_ylim(-400,400)
    plt.tight_layout()
    return fig
#%%
if __name__ == "__main__":
    m  = 1                          #unit conversions
    cm = 1e-2
    um = 1e-6
    cm2= 1e-4
    um2= 1e-12

    beam_power       = 0.5e12       #Watts     # in "Initial...restuls...Omega..."~Boehly et. all, it says 60TW 60-beam
    foc_len          = 1.8*m        #Omega final lens focal distance
    f_number         = 6.5          #Omega's f number
    Lambda           = 0.351*um     #Omega wavelength: 3rd harmonic of a Nd:Glass laser
    res_fac          = 1            #Note: Higher resolution is required to resolve the far-field speckles

    dpp_phase    = fmat['DPP_phase']
    KesslerParms = fmat['KesslerParms']
    nx, dx, ny, dy, x1d, y1d, (x,y) = create_2D_grids(KesslerParms)
    
    nx_, dx_, ny_, dy_, x1d_, y1d_, (x_,y_) = create_2D_grids(KesslerParms, resolution_fac=res_fac) #fine grid for high-res interpolation
    phase_func   = RectBivariateSpline(x1d,y1d,dpp_phase)                                           #interpolate the phase array from the phase plate
    dpp_phase    = phase_func(x1d_,y1d_)
    nx           = nx_                                                                              #this seems like it could be simplified, but I will leave it for now
    dx           = dx_  
    ny           = ny_  
    dy           = dy_  
    x1d          = x1d_ 
    y1d          = y1d_ 
    x,y          = x_,y_
    res_fac_pr   = res_fac
#%%   
    int_beam_env     = beam_envelope(alpha=np.log(0.5), sigma= (25.8*cm) / 2, x=x_, y=y_, N=24) #Beam intensity envelope at lens
    # int_beam_env     = np.exp(np.log(0.5) * (2 * np.sqrt(x ** 2 + y ** 2) / (25.8*cm)) ** 24) 
    area_nf          = dx * dy
#%%
    #old approach to normalizing:
    pwr_beam_env     = int_beam_env * area_nf           #power  envelope at lens
    pwr_beam_env_tot = np.sum(pwr_beam_env)
    pwr_beam_env    *= beam_power / pwr_beam_env_tot    #W
    pwr_beam_env_tot = np.sum(pwr_beam_env)

    int_beam_env     = pwr_beam_env / area_nf            #W/m^2
    efield_beam_env  = int_beam_env**0.5                 #V/m    #proportionality relation. E field amplitude squared is proportional to intensity

    #idealised far field intensity distribution by taking fft before applying phase plate
    far_field_ideal = np.fft.fftshift(np.fft.fft2(efield_beam_env , norm='ortho'))
    int_ff_ideal = np.abs(far_field_ideal)**2

    #far field intensity distribution with the phase plase applied
    dpp              = np.exp(-1j * dpp_phase)                                      #complex number of unit modulus. Product simply changes phase
    near_field       = efield_beam_env * dpp                                        #apply DPP phase modulation
    far_field        = np.fft.fftshift(np.fft.fft2(near_field , norm='ortho'))      #2D fft then shift zero-frequency component to the center of the spectrum
    int_ff           = np.abs(far_field)**2                                         #intensity is proportional to amplitude squared

    dxff, dyff = foc_len * Lambda / (dx * nx), foc_len * Lambda / (dy * ny)         #both values are in meters. Limit of resolution due to diffraction
                                                                                    #dx * nx gives total simulated distance at near field in meters
                                                                                    #foc_len / (dx * nx) is supposed to be f_number (but it isn't == the global variable f_number)
                                                                                    #conversion from near field to far field
    xff1d            = np.linspace( -1 * nx / 2, nx / 2, nx) * dxff                 #this multiplication happens to convert from pixels to meters
    yff1d            = np.linspace(-1 * ny / 2, ny / 2, nx) * dyff
    xff,yff          = np.meshgrid(xff1d, yff1d, indexing='ij')
    
    sigma_ff = (25.8*um*50) / 2
    int_beam_env_ff = beam_envelope(alpha=np.log(0.5), sigma= sigma_ff, x=xff, y=yff, N=24)

    FIGURE1_PHASE_INTENSITY = intensity_phase_plots(x1d, xff1d, y1d, yff1d, int_beam_env_ff, int_ff, dpp_phase)

    print(f"""Far field grid scales area: {dxff/um:2.2f} microns by {dyff/um:2.2f} microns, speckle scale is {f_number*Lambda/um:2.2f} microns""")

    area_ff          = dxff * dyff
    pwr_ff           = int_ff * area_ff

    #The FFT isn't conserving energy (not sure it should) so re-normlising here!
    pwr_ff          *= beam_power/np.sum(pwr_ff)
    pwr_ff_tot       = np.sum(pwr_ff)
    int_ff           = pwr_ff / area_ff

    # del int_beam_env, pwr_beam_env, int_beam_env, efield_beam_env, dpp, near_field, far_field

    print(f"""Power in near field {pwr_beam_env_tot*1e-12:2.2e} TW, power in far field = {pwr_ff_tot* 1e-12:2.2e}TW,
total intensity in near field {np.sum(int_beam_env) * 1e-12:2.2e} TW/m^2, total intensity in far field = {np.sum(int_ff)* 1e-12:2.2e} TW/m^2
area_nearfield: {area_nf:2.2e}, area_farfield: {area_ff:2.2e}
grid scale near field: {nx*dx:2.2f}m by {ny*dy:2.2f}m, focal length: {foc_len}m
f_num_x = {foc_len/(nx*dx)}, f_num_y = {foc_len/(ny*dy)}, f_number: {f_number}""")
    # fig = make_big_figure(x, y, nx, ny, x1d, y1d, xff, yff, dxff, dyff, dpp_phase, int_beam_env, int_ff, pwr_ff_tot, near_field, cm, cm2, um)
    plt.show()
