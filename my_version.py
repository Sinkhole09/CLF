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
def perform_fft_normalize(int_beam_env, area_nf,  dpp_phase, do_dpp=True, do_ssd=False, x=None, y=None, time=None, time_resolution=None):
	#old approach to normalizing:
	pwr_beam_env    	= int_beam_env * area_nf			#power  envelope at lens
	pwr_beam_env_tot	= np.sum(pwr_beam_env)
	pwr_beam_env   		*= beam_power / pwr_beam_env_tot	#W

	int_beam_env		= pwr_beam_env / area_nf			#W/m^2
	efield_beam_env		= np.exp(-1j) * np.exp(1j) * (int_beam_env)**0.5		#V/m    #proportionality relation. E field amplitude squared is proportional to intensity
	near_field 			= efield_beam_env
# idealised far field intensity distribution (fft before phase plate or smoothing)
	far_field_ideal = np.fft.fftshift(np.fft.fft2(near_field))
# apply ssd
	if do_ssd:
		near_field = apply_smoothing_by_spectral_dispersion(E_near_field=efield_beam_env, x_mesh=x, y_mesh=y, time=time, time_resolution=time_resolution)
# intensity in far field
	if do_dpp:
		dpp              = np.exp(-1j * dpp_phase)                  # complex number of unit modulus. Product simply changes phase
		near_field       *= dpp                                     # apply DPP phase modulation
	far_field        = np.fft.fftshift(np.fft.fft2(near_field))		# 2D fft then shift zero-frequency component to the center of the spectrum
	int_ff = np.abs(far_field) ** 2
	dxff, dyff = foc_len * Lambda / (dx * nx), foc_len * Lambda / (dy * ny)         # both values are in meters. Limit of resolution due to diffraction
																					# dx * nx gives total simulated distance at near field in meters
																					# foc_len / (dx * nx) is supposed to be f_number (but it isn't == the global variable f_number)
																					# conversion from near field to far field
	xff1d            = np.linspace( -1 * nx / 2, nx / 2, nx) * dxff                 # this multiplication happens to convert from pixels to meters
	yff1d            = np.linspace(-1 * ny / 2, ny / 2, nx) * dyff
	xff,yff          = np.meshgrid(xff1d, yff1d, indexing='ij')
	area_ff          = dxff * dyff
	pwr_ff           = int_ff * area_ff
# The FFT isn't conserving energy (not sure it should) so re-normlising here!
	pwr_ff          *= beam_power/np.sum(pwr_ff)
	int_ff           = pwr_ff / area_ff
	return xff1d, xff, dxff, yff1d, yff, dyff, int_ff, far_field_ideal, pwr_beam_env, near_field
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
def expand_grid(meshes):
	"""
	Take an input meshgrid and triple it's size by addings rows of zeros and columns of zeros to create a new meshgrid with the
	data of the input mesh at the center, and zeros surrounding it.
	"""
	expanded_meshes = []
	for mesh in meshes:
		expanded_mesh = mesh
		rows, cols = len(mesh[:,0]), len(mesh[0,:])
		y_zeros = np.zeros((rows, cols))
		expanded_mesh = np.concatenate((y_zeros, expanded_mesh, y_zeros), axis=0)
		rows_new = len(expanded_mesh[:,0])
		x_zeros = np.zeros((rows_new, cols))
		expanded_mesh = np.concatenate((x_zeros, expanded_mesh, x_zeros),axis=1)
		expanded_meshes.append(expanded_mesh)
	return expanded_meshes
def quantify_nonuniformity(int_ideal, int_nonuniform, sigma_0=None):
	rows, cols = np.shape(int_ideal)
	rms_nonuniformity_array = np.abs(int_ideal - int_nonuniform) ** 2
	sigma_rms = np.sqrt(np.sum(rms_nonuniformity_array) / (rows * cols))
	nonuniformity_percentage = None
	if sigma_0:
		nonuniformity_percentage = sigma_rms / sigma_0 * 100
	return nonuniformity_percentage, sigma_rms
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
def util_phase_modulation_eqn(x, y, t):
	TWO_PI 	= 2*np.pi
	phi 	= np.zeros(np.shape(x))
# B integral
# SSD
	delta_x, delta_y 	= 14.3, 6.15 			# no units; the modulation depth
	v_x, v_y			= 10.4e9, 3.30e9		# Hz; the rf modulation frequency
	omega_x, omega_y 	= TWO_PI*v_x, TWO_PI*v_y# rads/s; angular rf modulation frequency
	zeta_x, zeta_y		= 0.300e-9, 1.13e-9		#s/m; describes the variation in phase across the beam due to the angular grating dispersion
	phi_2D_ssd 			= 3 * delta_x*np.sin(omega_x * (t + zeta_x * x)) + 3 * delta_y*np.sin(omega_y * (t + zeta_y * y)) # equation (3) from Regan et. all (2005)
	phi 				+= phi_2D_ssd
	return phi
def apply_smoothing_by_spectral_dispersion(E_near_field, time, x_mesh, y_mesh, time_resolution):
	timesteps 			= [i*(time / time_resolution) for i in range(time_resolution)] # dividing the total time into discrete steps
	rows, cols 			= np.shape(E_near_field)
	E_near_field_ssd 	= np.zeros((rows,cols), dtype=complex)
	for timestep in timesteps:
		E_near_field_ssd += E_near_field * np.exp(1j * util_phase_modulation_eqn(x_mesh, y_mesh, timestep))
	return E_near_field_ssd / time_resolution
def util_find_relative_intensity(nf_x, ff_x, nf_y, ff_y, int_ff_ideal_raw, int_ff_raw):
	x, dx 			= nf_x
	xff, dxff 		= ff_x
	y, dy 			= nf_y
	yff, dyff 		= ff_y
	peak_int_ideal 	= np.max(int_ff_ideal_raw)
	int_ff_ideal 	= int_ff_ideal_raw / peak_int_ideal		# relative intensity found when we divide by peak intensity
	peak_int_ideal *= (dx * dy) / (dxff * dyff)				# scaling to far field
	int_ff = int_ff_raw / peak_int_ideal

	return x, xff, y, yff, int_ff_ideal, int_ff
def util_plt_one_column_or_row(data_arrays, plt_attributes, one_col=True):
	if one_col:
		fig, ax = plt.subplots(nrows=len(data_arrays), ncols=1, figsize=(8,2*8))
	else:
		fig, ax = plt.subplots(nrows=1, ncols=len(data_arrays), figsize=(2*8,8)) 
	for pos, data in enumerate(data_arrays):
		x, y, data_array 														= data
		plot_type, xlabel, ylabel, norm, xlim, ylim, x_scale, y_scale, varname 	= plt_attributes[pos]
		ax[pos]	.set_xlabel(xlabel)
		ax[pos]	.set_ylabel(ylabel)
		if plot_type == 'img':
			pcm		= ax[pos].pcolormesh(x / x_scale, y / y_scale, data_array, norm=norm, cmap='rainbow')
			cb 		= fig.colorbar(pcm, ax=ax[pos])
			cb		.set_label(varname)
		if plot_type == 'line':
			line1, line2, line3 = data_array
			lower, upper = ylim
			ax[pos].plot(x / x_scale, line1, linestyle='--', color='red', label='Ideal Lineshape')
			ax[pos].plot(x / x_scale, line2, color='black', label='Without PS')
			ax[pos].plot(x / x_scale, line3, color='grey', label='With PS')
			ax[pos].set_yscale(norm)
			ax[pos].set_xlim(-1*xlim, xlim)
			ax[pos].set_ylim(lower, upper) #if plot_type is line, expect ylim to be a tuple of two ints
			ax[pos].legend()
			continue
		if xlim:
			ax[pos]	.set_xlim(-1*xlim, xlim)
		if ylim:	
			ax[pos]	.set_ylim(-1*ylim, ylim)
	return fig
def intensity_plot_old(nf_x, ff_x, nf_y, ff_y, int_ff_ideal_raw, int_ff_raw, ssd_data_items=None, do_PS=False, do_PS_SSD=False, ps_shift=4, do_LogNorm=False, do_line=False, plot_ssd=False):
	"""
	Make plots of ff intensity distribution with different levels of smoothing applied.
	"""
	int_ff_ssd_raw, ssd_time_resolution, ssd_duration = ssd_data_items
	data_to_plot 	= []
	img_norm 		= None
	if do_LogNorm:
		img_norm 	= LogNorm(vmin=1e-1)
	plt_attributes 	= [] # (plot_type, xlabel, ylabel, norm, xlim, ylim, xscale, ysclae, label/title)
	ff_attribute	= ('img','x (um)', 'y (um)', img_norm, 400, 400, um, um)
	ff_data			= (ff_x[0], ff_y[0])
	nf_attribute	= ('img','x (cm)', 'y (cm)', img_norm, None, None, cm, cm)
	nf_data			= (nf_x[0], nf_y[0])
	_, xff, _, yff, int_ff_ideal, int_ff_onlyDPP	= util_find_relative_intensity(nf_x, ff_x, nf_y, ff_y,
																			 int_ff_ideal_raw, int_ff_raw)
	data_to_plot									.append(ff_data + (int_ff_onlyDPP,))					# only phase plate applied
	plt_attributes									.append(ff_attribute + ("DPP INT",))
	if plot_ssd:
		_, _, _, _, _, int_ff_ssd	= util_find_relative_intensity(nf_x, ff_x, nf_y, ff_y,
									int_ff_ideal_raw, int_ff_ssd_raw)
		data_to_plot				.append(ff_data + (int_ff_ssd,))						# phase plate and ssd applied
		plt_attributes				.append(ff_attribute + (f"DPP SSD INT, t_res: {ssd_time_resolution}\n duration: {ssd_duration:2.0e}s",))
	else: do_PS_SSD=False
	if do_PS:
		if do_PS_SSD: to_smooth = int_ff_ssd
		else: to_smooth 		= int_ff_onlyDPP
		int_ff_DPP_and_PS 	= apply_polarisation_smoothing(to_smooth, shift=ps_shift)					# phase plate and ps applied
		data_to_plot		.append(ff_data + (int_ff_DPP_and_PS,))
		plt_attributes		.append(ff_attribute+ (f"DPP PS INT",))
	if do_line:
		center 									= int(np.ceil(len(int_ff[0,:]) * 0.5))
		int_ff_ideal_center_row					= int_ff_ideal[center]
		int_ff_onlyDPP_center_row				= int_ff_onlyDPP[center]
		data_lines 								= (int_ff_ideal_center_row, int_ff_onlyDPP_center_row)
		if do_PS:
			int_ff_DPP_and_PS_center_row		= int_ff_DPP_and_PS[center]
			data_lines							+= (int_ff_DPP_and_PS_center_row,)
		if plot_ssd:
			int_ff_DPP_SSD_center_row		= int_ff_ssd[center]
			data_lines						+= (int_ff_DPP_SSD_center_row,) 
		data_to_plot					.append((xff, yff,) + (data_lines,))
		plt_attributes					.append(('line','x (um)', 'Intensity Normalized', 'log', 600, (None,None), um, 1, 'lineout'))
	num_plts 							= len(data_to_plot)
	if num_plts < 4:
		FIGURE	= util_plt_one_column_or_row(data_to_plot, plt_attributes)
	_, sigma_0	= quantify_nonuniformity(int_ff_ideal, int_ff_onlyDPP)
	nonuniformity_percentage_PS, _ 	= quantify_nonuniformity(int_ff_ideal, int_ff_DPP_and_PS, sigma_0) 	if do_PS 	else -1, _
	nonuniformity_percentage_ssd, _ = quantify_nonuniformity(int_ff_ideal, int_ff_ssd, sigma_0) 		if plot_ssd else -1, _
	return FIGURE, nonuniformity_percentage_PS, nonuniformity_percentage_ssd, 100
# def intensity_plots(int_ideal, intensity_distributions, do_PS=False, do_LogNorm=False):
# 	norm = None
# 	if do_LogNorm:
# 		norm=LogNorm(vmin=1e-1)
# # 							 (plot_type, xlabel, ylabel, norm, xlim, ylim, xscale, ysclae, label/title)
# 	ff_intensity_attribute = ('img','x (um)', 'y (um)', norm, 400, 400, um, um, 'Intensity far field')
# 	nf_intensity_attribute = ('img','x (cm)', 'y (cm)', norm, None, None, cm, cm, 'Intensity near field')
# 	data = []
# 	attributes = []
# 	for int_dist in intensity_distributions:
# 		intensity_distribution, scope = int_dist
# 		data.append(intensity_distribution)
# 		if scope == 'nf':
# 			attributes.append(nf_intensity_attribute)
# 		elif scope == 'ff':
# 			attributes.append(ff_intensity_attribute)
# 	fig = util_plt_one_column_or_row(data, attributes)

# 	return fig
def intensity_phase_plots(nf_x, ff_x, nf_y, ff_y, int_ff_ideal_raw, int_ff_raw, dpp):
	x, xff, y, yff, int_ff_ideal, int_ff 	= util_find_relative_intensity(nf_x, ff_x, nf_y, ff_y, int_ff_ideal_raw, int_ff_raw)
	data 									= [int_ff_ideal, dpp, int_ff]
	x_y 									= [(xff,yff), (x,y), (xff,yff)]
	unit 									= [('um', um), ('cm', cm), ('um', um)]
	fig, ax									= plt.subplots(nrows=1, ncols=3, figsize=(2*8,8))
	for pos, data_array in enumerate(data):
		x_val, y_val	= x_y[pos]
		pcm 			= ax[pos].pcolormesh(x_val / unit[pos][1], y_val/ unit[pos][1], data_array,cmap='rainbow')
		cb				= fig.colorbar(pcm, ax=ax[pos])
		ax[pos].set_xlabel(f"x {unit[pos][0]}")
		ax[pos].set_ylabel(f"y {unit[pos][0]}")
		if pos == 2:
			ax[pos].set_xlim(-400,400)
			ax[pos].set_ylim(-400,400)
	plt.tight_layout()
	return fig
def util_shift_2_pi(phase_array, TWO_PI):
	make_periodic 						= np.copy(phase_array)
	if np.any(make_periodic 			> TWO_PI):
		mask_above_2pi 					= make_periodic > TWO_PI
		make_periodic[mask_above_2pi] 	-= TWO_PI
		make_periodic 					= util_shift_2_pi(make_periodic, TWO_PI)
	if np.any(make_periodic				<= 0):
		mask_below_zero 				= make_periodic <= 0
		make_periodic[mask_below_zero] 	+= TWO_PI
		make_periodic 					= util_shift_2_pi(make_periodic, TWO_PI)
	return make_periodic
def make_periodic_phase(dpp_phase):
	"""
	Make a phase array periodic between 2pi and 0
	Parameters:
		dpp_phase (array of arrays of floats): 2d array containing phase values
	"""
	TWO_PI = 2*np.pi
	make_periodic = util_shift_2_pi(dpp_phase, TWO_PI)
	return make_periodic
def print_function(var_list):
	
	(nx, dx, dxff, ny, dy, dyff, area_nf, area_ff,
	int_beam_env, int_ff, pwr_beam_env_tot, pwr_ff_tot,
	foc_len, Lambda,
	nonuni_DPP, nonuni_DPP_PS, nonuni_DDP_SSD, ssd_time_resoution) = tuple(var_list)
	print(f"""Far field grid scales area: {dxff/um:2.2f} microns by {dyff/um:2.2f} microns, speckle scale is {foc_len/(dx*nx)*Lambda/um:2.2f} microns

Power in near field {pwr_beam_env_tot*1e-12:2.2e} TW, power in far field = {pwr_ff_tot* 1e-12:2.2e}TW,

total intensity in near field {np.sum(int_beam_env) * 1e-12:2.2e} TW/m^2, total intensity in far field = {np.sum(int_ff)* 1e-12:2.2e} TW/m^2

area_nearfield: {area_nf:2.2e}, area_farfield: {area_ff:2.2e}

grid scale near field: {nx*dx:2.2f}m by {ny*dy:2.2f}m, focal length: {foc_len}m

f_num_x = {foc_len/(nx*dx)}, f_num_y = {foc_len/(ny*dy)}, f_number: {f_number}

Nonuniformity with only DPP: {nonuni_DPP:3.2f}% (by definition), with PS: {nonuni_DPP_PS:3.2f}%, with ssd: {nonuni_DDP_SSD:3.2f}%, ssd time resolution: {ssd_time_resoution}""")
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
#%%
if __name__ == "__main__":
# global variables
	m, cm, um, cm2, um2  = 1, 1e-2, 1e-6, 1e-4, 1e-12	#unit conversions

	beam_power       = 0.5e12       					#Watts     # in "Initial...restuls...Omega..."~Boehly et. all, it says 60TW 60-beam
	foc_len          = 1.8*m        					#Omega final lens focal distance
	f_number         = 6.5          					#Omega's f number
	Lambda           = 0.351*um     					#Omega wavelength: 3rd harmonic of a Nd:Glass laser
	res_fac          = 1            					#Note: Higher resolution is required to resolve the far-field speckles
# extracting phaseplate data and making grids
	dpp_phase    							= fmat['DPP_phase']
	KesslerParms 							= fmat['KesslerParms']
	nx, dx, ny, dy, x1d, y1d, (x,y) 		= create_2D_grids(KesslerParms)
	nx_, dx_, ny_, dy_, x1d_, y1d_, (x_,y_) = create_2D_grids(KesslerParms, resolution_fac=res_fac) #fine grid for high-res interpolation
	phase_func   							= RectBivariateSpline(x1d,y1d,dpp_phase)				#interpolate the phase array from the phase plate
	dpp_phase    							= phase_func(x1d_,y1d_)
	use_periodic_phase 						= True
	if use_periodic_phase:
		dpp_phase = make_periodic_phase(dpp_phase)
	nx, dx, x1d, ny, dy, y1d, (x,y), res_fac_pr = nx_, dx_, x1d_, ny_, dy_, y1d_, (x_,y_), res_fac
# initial envelope and ffts
	int_beam_env     = beam_envelope(alpha=np.log(0.5), sigma= (25.8*cm) / 2, x=x_, y=y_, N=24) #Beam intensity envelope at lens
	area_nf          = dx * dy
	xff1d, xff, dxff, yff1d, yff, dyff, int_ff, far_field_ideal, pwr_beam_env, near_field = perform_fft_normalize(int_beam_env, area_nf,
																											   dpp_phase, do_dpp=True, do_ssd=False)
# applying ssd
	ssd_time_resolution						= 100
	ssd_duration							= 1e-9
	_, _, _, _, _, _, int_ff_ssd, _, _, _ 	= perform_fft_normalize(int_beam_env, area_nf,
															   dpp_phase, do_dpp=True, do_ssd=False,
															   x=x, y=y, time=ssd_duration, time_resolution=ssd_time_resolution)
# this must be done after the previous perform_fft_normalize() function call becuase we are changing int_beam_env
	pwr_beam_env_tot 	= np.sum(pwr_beam_env)				
	area_ff 			= dxff * dyff
	pwr_ff_tot      	= np.sum(int_ff * area_ff)
	int_beam_env 		= pwr_beam_env / area_nf
# making figures
	# FIGURE1_PHASE_INTENSITY = intensity_phase_plots((x1d,dx), (xff1d,dxff), (y1d,dy), (yff1d,dyff), int_beam_env, int_ff, dpp_phase)
	ONLY_INTENSITY, nonuniform_DPP_and_PS, nonuniformity_ssd, nonuniform_DPP = intensity_plot_old(nf_x=(x1d,dx), ff_x=(xff1d,dxff), nf_y=(y1d,dy), ff_y=(yff1d,dyff),
																		int_ff_ideal_raw=int_beam_env, int_ff_raw=int_ff,
																		ssd_data_items=(int_ff_ssd, ssd_time_resolution, ssd_duration),
																		do_PS=True, plot_ssd=False, do_PS_SSD=True, ps_shift=10, do_line=True)
	# fig = make_big_figure(x, y, nx, ny, x1d, y1d, xff, yff, dxff, dyff,
	# 				   dpp_phase, int_beam_env, int_ff, pwr_ff_tot, near_field,
	# 				   cm, cm2, um)
# printing key results
	# var_list = [nx, dx, dxff, ny, dy, dyff,
	# 			area_nf, area_ff,
	# 			int_beam_env, int_ff, pwr_beam_env_tot, pwr_ff_tot,
	# 			foc_len, Lambda,
	# 			nonuniform_DPP, nonuniform_DPP_and_PS, nonuniformity_ssd, ssd_time_resolution]
	# print_function(var_list)
	
	plt.show()
