#%%
# import netCDF4
# import csv
# import datetime
# import pandas as pd
# from matplotlib.animation import FuncAnimation
# import matplotlib as mpl
import os
from scipy.io import loadmat
from scipy.interpolate import RectBivariateSpline
import scipy.optimize as sciop
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm
import pandas as pd
#%%
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
def super_gaussian(cords, amp, sigma, N):
	x, y = cords
	r2 = np.abs(x/sigma) ** 2 + np.abs(y/sigma) ** 2
	g = amp * np.exp(
			-1*(r2) ** N 
	)
	return np.ravel(g)
def beam_envelope(cords, N, amp=1, alpha=1, sigma=1, super_g_x_zero=1, super_g_y_zero=1):
	"""
	amp * np.exp(
			alpha * (np.sqrt(
			(x / super_g_x_zero) ** 2 + (y / super_g_y_zero) ** 2
		) / (sigma)) ** N
	"""
	x, y = cords
	envelope 	= amp * np.exp(
			alpha * (
			(np.abs(x / super_g_x_zero) ** 2 + np.abs(y / super_g_y_zero) ** 2) / sigma**2
		) ** N
	) #Beam intensity envelope at lens
	return envelope
def create_2D_grids(KesslerParms, resolution_fac=1, scale_factor=1):
	nx             = KesslerParms.fftCells_x * resolution_fac  # Number of pixels in the x-direction in the FFT grid (simulated near-field image)
	dx             = KesslerParms.nfPixSize_x / resolution_fac # Size of each pixel in x-direction at the near-field plane (where FFT is evaluated); I guess in meters?
	ny             = KesslerParms.fftCells_y * resolution_fac 
	dy             = KesslerParms.nfPixSize_y / resolution_fac # 0.0003125 (meters I guess?)
	nx *= scale_factor
	ny *= scale_factor
	x1d          = np.linspace(-1*nx/2,nx/2,nx) * dx
	y1d          = np.linspace(-1*ny/2,ny/2,ny) * dy
	x,y          = np.meshgrid(x1d,y1d,indexing='ij')
	return nx, dx, ny, dy, x1d, y1d, (x,y)
def util_scale_int_env(int_beam_env, area, beam_power):
	pwr_beam_env    	= int_beam_env * area				#power  envelope at lens
	pwr_beam_env_tot	= np.sum(pwr_beam_env)
	pwr_beam_env   		*= beam_power / pwr_beam_env_tot	#W
	int_beam_env			= pwr_beam_env / area		#W/m^2
	return int_beam_env
def util_calc_ff_parameters(foc_len, Lambda, nx, dx, ny, dy):
	dxff, dyff = foc_len * Lambda / (dx * nx), foc_len * Lambda / (dy * ny)         # both values are in meters. Limit of resolution due to diffraction
																					# dx * nx gives total simulated distance at near field in meters
																					# foc_len / (dx * nx) is supposed to be f_number (but it isn't == the global variable f_number)
																					# conversion from near field to far field
	xff1d            = np.linspace( -1 * nx / 2, nx / 2, nx) * dxff                 # this multiplication happens to convert from pixels to meters
	yff1d            = np.linspace(-1 * ny / 2, ny / 2, nx) * dyff
	xff,yff          = np.meshgrid(xff1d, yff1d, indexing='ij')
	area_ff          = dxff * dyff
	return xff1d, dxff, xff, yff1d, dyff, yff, area_ff 
def util_compute_fft(near_field, do_abs_squared=True):
	efield_ff =  np.fft.fftshift(
					np.fft.fft2(
						near_field, norm='ortho'))
	efield 		= np.abs(efield_ff)
	if do_abs_squared: efield = efield**2
	return efield
def util_make_folder_of_intensity_distribution_files(int_ff, time, area_ff, beam_power):
	int_ff		= util_scale_int_env(int_ff, area_ff, beam_power)
	folder_path = "./Intensity_Distributions/"
	if not os.path.exists(folder_path):
		os.makedirs(folder_path)
	ps 			= 1e-12
	file_path 	= folder_path + f"Instantaneous intensity distribution at {time/ps:3.2}ps.txt"
	rows, cols 	= np.shape(int_ff)
	with open(file_path, "w") as file:
		file.write("\t".join(str(a) for a in np.arange(cols)) + "\n")
		for row in range(rows):
			file.write("\t".join(str(val) for val in int_ff[row]) + "\n")
	return
def util_saving_sigma_rms(int_ideal_ff, int_ff_ssd, int_ff_ssd_ps, idx:int,  scale_from_max:int, sigma_rms_ssd_all:list, sigma_rms_ssd_ps_all:list):
	_, sigma_rms_ssd 	= quantify_nonuniformity(int_ideal_ff, int_ff_ssd / (idx + 1), sigma_rms_ssd_all[0], scale_from_maximum=scale_from_max)
	_, sigma_rms_ssd_ps = quantify_nonuniformity(int_ideal_ff, int_ff_ssd_ps / (idx + 1), sigma_rms_ssd_ps_all[0], scale_from_maximum=scale_from_max)
	sigma_rms_ssd_all	.append(sigma_rms_ssd)
	sigma_rms_ssd_ps_all.append(sigma_rms_ssd_ps)
	return (sigma_rms_ssd_all, sigma_rms_ssd_ps_all)
def util_write_sigma_rms_to_file(sigma_rms_ssd_all, sigma_rms_ssd_ps_all):
	with open("output.txt", "w") as file:
		file.write(f"sigma_rms_ssd\tsigma_rms_ssd_ps\n")
		file.writelines(f"{a}\t{b}\n" for a,b in zip(sigma_rms_ssd_all, sigma_rms_ssd_ps_all))
def perform_fft_normalize(int_beam_env, dpp_phase, beam_power, int_ideal_ff=None, area_nf=None, area_ff=None, do_dpp=True, do_ssd=False, scale_from_max=2,
						  x=None, y=None, time=None, time_resolution=None, write_to_file=False):
	efield_beam_env			= np.exp(-1j) * np.exp(1j) * (int_beam_env)**0.5		#V/m    #proportionality relation. E field amplitude squared is proportional to intensity
	near_field 				= efield_beam_env
# idealised far field intensity distribution (fft before phase plate or smoothing)
	far_field_ideal = util_compute_fft(near_field)
# applying ssd and finding far field intensity distribution
	if do_dpp:
		dpp				= np.exp(1j * dpp_phase)					# complex number of unit modulus. Product simply changes phase
		near_field		*= dpp										# apply DPP phase modulation
		int_ff_onlyDPP	=  util_compute_fft(near_field)
	if do_ssd:
		int_ff_ssd, int_ff_ssd_ps		= np.zeros(np.shape(near_field)), np.zeros(np.shape(near_field))
		timesteps 						= [i*(time / time_resolution) for i in range(time_resolution)] # dividing the total time into discrete steps
		sigma_rms_ssd_all				= [quantify_nonuniformity(int_ideal_ff, int_ff_onlyDPP, scale_from_maximum=scale_from_max)[1]]
		sigma_rms_ssd_ps_all			= [quantify_nonuniformity(int_ideal_ff, apply_polarisation_smoothing(int_ff_onlyDPP), scale_from_maximum=scale_from_max)[1]]
		for idx, near_field in tqdm(enumerate(apply_smoothing_by_spectral_dispersion(
			E_near_field=near_field, x_mesh=x, y_mesh=y, timesteps=timesteps, time_resolution=time_resolution, use_gen=True
			)), desc="Applying smoothing by spectral dispersion smoothing"):
			int_ff										= util_compute_fft(near_field)
			int_ff										= util_scale_int_env(int_beam_env=int_ff, area=area_ff, beam_power=beam_power)
			int_ff_ssd									+= int_ff
			int_ff_ssd_ps								= apply_polarisation_smoothing(int_ff_ssd)
			# (sigma_rms_ssd_all, sigma_rms_ssd_ps_all) 	= util_saving_sigma_rms(int_ideal_ff, int_ff_ssd, int_ff_ssd_ps, idx, 
			# 														  scale_from_max, sigma_rms_ssd_all, sigma_rms_ssd_ps_all)
			if write_to_file:
				util_make_folder_of_intensity_distribution_files(int_ff, timesteps[idx], area_ff, beam_power)
		# if write_to_file:
		# 	util_write_sigma_rms_to_file(sigma_rms_ssd_all, sigma_rms_ssd_ps_all)
		sigma_rms_ssd_all 		= np.array(sigma_rms_ssd_all) / sigma_rms_ssd_all[0]
		sigma_rms_ssd_ps_all	= np.array(sigma_rms_ssd_ps_all) / sigma_rms_ssd_ps_all[0]
		int_ff					= int_ff_ssd / time_resolution
	else:
		int_ff 									= util_compute_fft(near_field)
		sigma_rms_ssd_all, sigma_rms_ssd_ps_all = None, None
# The FFT isn't conserving energy (not sure it should) so re-normlising here!
	int_ff	= util_scale_int_env(int_ff, area_ff, beam_power)
	return int_ff, far_field_ideal, int_ff_onlyDPP, sigma_rms_ssd_all, sigma_rms_ssd_ps_all
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
def expand_grid(meshes, scale_factor=3):
	"""
	Take an input meshgrid and triple it's size by addings rows of zeros and columns of zeros to create a new meshgrid with the
	data of the input mesh at the center, and zeros surrounding it.
	"""
	expanded_meshes = []
	for mesh in tqdm(meshes, desc="Expanding grids"):
		rows, cols = np.shape(mesh)
		scale = int(np.ceil(scale_factor))
		exp_rows, exp_cols = rows*scale, cols*scale
		expanded_mesh = np.zeros((exp_rows,exp_cols))
		start = int(np.floor((exp_rows - rows) / 2))
		end = exp_rows - start
		expanded_mesh[start:end,start:end] += mesh
		expanded_meshes.append(expanded_mesh)
	return expanded_meshes
def quantify_nonuniformity(int_ideal_unfiltered, int_nonuniform_unfiltered, sigma_0=None, scale_from_maximum=2):
	mask_fwhm					= int_ideal_unfiltered >= (np.max(int_ideal_unfiltered) / scale_from_maximum)
	int_ideal, int_nonuniform	= int_ideal_unfiltered[mask_fwhm], int_nonuniform_unfiltered[mask_fwhm]
	rms_nonuniformity_array = np.abs(int_ideal - int_nonuniform) ** 2
	sigma_rms = np.sqrt(np.mean(rms_nonuniformity_array))
	nonuniformity_percentage = (sigma_rms / sigma_0) * 100 	if sigma_0 else None
	return nonuniformity_percentage, sigma_rms
def apply_polarisation_smoothing(int_ff, shift=2, conversion_factor=1):
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
def apply_smoothing_by_spectral_dispersion(E_near_field, x_mesh, y_mesh, timesteps, time_resolution, use_gen=True):
	rows, cols 			= np.shape(E_near_field)
	E_near_field_ssd 	= np.zeros((rows,cols), dtype=complex)
	if not use_gen:
		for timestep in tqdm(timesteps, desc="Applying polarization smoothing"):
			E_near_field_ssd += E_near_field * np.exp(1j * util_phase_modulation_eqn(x_mesh, y_mesh, timestep))
		return E_near_field_ssd / time_resolution
	else:
		for pos, timestep in enumerate(timesteps):
			yield E_near_field * np.exp(1j * util_phase_modulation_eqn(x_mesh, y_mesh, timestep))
def util_find_relative_intensity(nf_x, ff_x, nf_y, ff_y, int_ff_ideal_raw, int_ff_raw):
	x, dx 			= nf_x
	xff, dxff 		= ff_x
	y, dy 			= nf_y
	yff, dyff 		= ff_y
	peak_int_ideal 	= np.max(int_ff_ideal_raw)
	int_ff_ideal 	= int_ff_ideal_raw / peak_int_ideal			# relative intensity found when we divide by peak intensity
	# peak_int_ideal *= (dx * dy) / (dxff * dyff)				# scaling to far field
	int_ff = int_ff_raw / peak_int_ideal
	return x, xff, y, yff, int_ff_ideal, int_ff
def util_img_plot(fig, ax, x, y, data, attributes, show_cb=True):
	"""
	Plot a 2D image (pseudocolor mesh) of the provided data on given axes.

	Args:
		fig (matplotlib.figure.Figure): Figure object to attach the colorbar to.
		ax (matplotlib.axes.Axes): Axes object to plot into.
		x (ndarray): X-axis grid (spatial or temporal coordinates).
		y (ndarray): Y-axis grid (spatial or temporal coordinates).
		data (ndarray): 2D array of data values to plot.
		attributes (tuple): Plot attributes in the form
			(plot_type, xlabel, ylabel, norm, xlim, ylim, x_scale, y_scale, varname)
			where:
				* plot_type (str): Expected to be "img".
				* xlabel, ylabel (str): Axis labels.
				* norm (str or None): "log" for logarithmic normalization, else None.
				* xlim, ylim (float or None): Axis limits, symmetric about zero.
				* x_scale, y_scale (float): Scaling factors for coordinates.
				* varname (str): Label for the colorbar.

	Returns:
		None

	Notes:
		- Uses a rainbow colormap by default.
		- Applies symmetric limits if `xlim` or `ylim` is provided.
		- Applies logarithmic normalization if `norm="log"`.
	"""
	if attributes is None:
		pcm = ax.pcolormesh(data, cmap='rainbow')
		fig.colorbar(pcm, ax=ax)
		return pcm
	plot_type, xlabel, ylabel, set_norm, xlim, ylim, x_scale, y_scale, varname = attributes
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	norm = LogNorm(vmin=1e-5) if set_norm == 'log' else None
	pcm		= ax.pcolormesh(x / x_scale, y / y_scale, data, norm=norm, cmap='rainbow')
	if show_cb:
		cb 		= fig.colorbar(pcm, ax=ax)
		cb		.set_label(varname)
	if xlim:
		ax.set_xlim(-1*xlim, xlim)
	if ylim:	
		ax.set_ylim(-1*ylim, ylim)
	return pcm
def util_line_plot(ax, x, y, data, attributes, xnorm=None, do_scatter=False, do_top_edge=False, top_items=(None, None), show_legend=True):
	"""
	Plot one or multiple line plots on given axes.

	Args:
		ax (matplotlib.axes.Axes): Axes object to plot into.
		x (ndarray): 1D array of x-axis values.
		y (ndarray): Placeholder (not directly used unless paired with `data`).
		data (ndarray or list of ndarray): Y-data for line plots.
			- If plotting a single line, this is a 1D array.
			- If plotting multiple lines, this is a list/2D array.
		attributes (tuple): Plot attributes in the form
			(plot_type, xlabel, ylabel, norm, xlim, ylim, x_scale, line_design, varname)
			where:
				* plot_type (str): Expected to be "line".
				* xlabel, ylabel (str): Axis labels.
				* norm (str or None): "log" for logarithmic y-scale, else None.
				* xlim (float or tuple or None): Limits for x-axis.
				* ylim (tuple or None): Limits for y-axis (min, max).
				* x_scale (float): Scaling factor for x-axis values.
				* line_design (tuple or list of tuple):
					- Single (linestyle, color, label) tuple for one line.
					- List of such tuples for multiple lines.
				* varname (str): Label for legend entries.
		xnorm (str or None, optional): Normalization for x-axis ("log", etc.).
		do_scatter (bool): Make scatter plot if true else regular line plot.

	Returns:
		None

	Notes:
		- Adds a legend automatically if labels are provided.
		- If `ylim` is a tuple, sets exact lower and upper limits.
		- Marks single lines with "x" markers.
	"""
	plot_type, xlabel, ylabel, norm, xlim, ylim, x_scale, line_design, varname = attributes
	if np.shape(line_design) == (3,):
		line_style, colour, label = line_design
		if do_scatter:	ax.scatter(x / x_scale, data, color=colour, label=label, marker='x')
		else: 			ax.plot(x / x_scale, data, linestyle=line_style, color=colour, label=label, marker='x')
	else:
		for index, line in enumerate(data):
			line_style, colour, label = line_design[index]
			if do_scatter:	ax.scatter(x / x_scale, line, color=colour, label=label, marker='x')
			else: 			ax.plot(x / x_scale, line, linestyle=line_style, color=colour, label=label)
			# ax.plot(x / x_scale, line, linestyle=line_style, color=colour, label=label)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	if xlim is not None:
		if isinstance(xlim, (int, float)):
			ax.set_xlim(-1*xlim, xlim)
		elif isinstance(xlim, (tuple, list)) and len(xlim)==2:
			ax.set_xlim(xlim[0], xlim[1])
	if ylim:
		lower, upper = ylim
		ax.set_ylim(lower, upper) #if plot_type is line, expect ylim to be a tuple of two ints
	if norm:
		ax.set_yscale(norm)
	if xnorm:
		ax.set_xscale(xnorm)
	if do_top_edge:
		top_data, top_label = top_items
		top = ax.twiny()
		top.set_xlim(np.min(top_data), np.max(top_data))
		top.set_xscale(xnorm)
		top.set_xlabel(top_label)
	if show_legend: ax.legend()
def util_plt_one_column_or_row(data_arrays, plt_attributes, one_col=True):
	if one_col: fig, ax = plt.subplots(nrows=len(data_arrays), ncols=1, figsize=(8,2*8))
	else: 		fig, ax = plt.subplots(nrows=1, ncols=len(data_arrays), figsize=(2*8,8)) 
	for pos, data in tqdm(enumerate(data_arrays), desc="Plotting"):
		x, y, data_array 	= data
		plot_type 			= plt_attributes[pos][0]
		if plot_type == 'img':
			util_img_plot(fig, ax[pos], x, y, data_array, plt_attributes[pos])
		if plot_type == 'line':
			util_line_plot(ax[pos], x, y, data_array, plt_attributes[pos])
	plt.tight_layout()
	return fig
def util_plt_MbyN_grid(data_Arrays, plt_attributes, nrows=3, fig_scale=8):
	num_plots = len(data_Arrays)
	ncols = int(np.ceil(num_plots / nrows))
	fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(nrows*fig_scale, ncols*fig_scale))
	for pos, data in tqdm(enumerate(data_Arrays), desc="Plotting"):
		ix = pos % nrows
		iy = pos // nrows
		x, y, data_array 	= data
		plot_type 			= plt_attributes[pos][0]
		if plot_type == 'img':
			util_img_plot(fig, ax[ix, iy], x, y, data_array, plt_attributes[pos])
		if plot_type == 'line':
			util_line_plot(ax[ix, iy], x, y, data_array, plt_attributes[pos])
	plt.tight_layout()
	return fig
def util_add_plot_items(plot_data, data, plot_attributes, attribute):
	plot_data.append(data)
	plot_attributes.append(attribute)
	return plot_data, plot_attributes
def util_add_line_items(field, data_tuple, tuple_design, line_design: list):
	data_tuple += (field,)
	line_design.append(tuple_design)
	return data_tuple, line_design
def util_make_ideal_ff_using_curvefit(xff, yff, int_ff_ssd, int_ff_DPP_and_PS, int_ff_onlyDPP, scale_from_max):
	cords 	= np.meshgrid(xff, yff)
	p0		= [1, 358e-4*cm, 2]
	bounds 	= ([0, 1e-5, 1], [3, 1e-3, 8])
	ssd_pobv, _ 		= sciop.curve_fit(super_gaussian, cords, int_ff_ssd.ravel(), p0, bounds=bounds)
	ideal_fit_ssd		= super_gaussian(cords, ssd_pobv[0], ssd_pobv[1], ssd_pobv[2],).reshape(np.shape(int_ff_ssd))
	ssd_ps_pobv, _ 		= sciop.curve_fit(super_gaussian, cords, int_ff_DPP_and_PS.ravel(), p0)
	ideal_fit_ssd_ps	= super_gaussian(cords,ssd_ps_pobv[0], ssd_ps_pobv[1], ssd_ps_pobv[2]).reshape(np.shape(int_ff_ssd))
	_, sigma_0_ssd		= quantify_nonuniformity(ideal_fit_ssd, int_ff_onlyDPP, scale_from_maximum=scale_from_max)
	_, sigma_0_ssd_ps	= quantify_nonuniformity(ideal_fit_ssd_ps, int_ff_onlyDPP, scale_from_maximum=scale_from_max)
	# data_to_plot, plt_attributes = util_add_plot_items(data_to_plot, ff_data + (ideal_fit_ssd,), plt_attributes, ff_attribute + ("IDEAL INT SSD",))
	# data_to_plot, plt_attributes = util_add_plot_items(data_to_plot, ff_data + (ideal_fit_ssd_ps,), plt_attributes, ff_attribute + ("IDEAL INT SSD_with_PS",))
	return ideal_fit_ssd, ideal_fit_ssd_ps, sigma_0_ssd, sigma_0_ssd_ps
def intensity_plot_old(nf_x, ff_x, nf_y, ff_y, int_ff_ideal_raw, int_ff_raw, ideal_int_nf=True,
					   ssd_data_items=None, do_ideal=True, do_DPP=True, do_PS=False, do_PS_SSD=False, ps_shift=4,
					   do_LogNorm=False, do_line=False, plot_ssd=False, return_fig=True, use_sciop=False, scale_from_max=2):
	"""
	Make plots of ff intensity distribution with different levels of smoothing applied.
	"""
	int_ff_ssd_raw, ssd_time_resolution, ssd_duration = ssd_data_items
	data_to_plot 	= [] # (x, y, data_array)
	img_norm 		= 'log' if do_LogNorm else None
	line_norm		= 'linear'
	plt_attributes 	= [] # (plot_type, xlabel, ylabel, norm, xlim, ylim, xscale, ysclae, label/title)
	ff_attribute	= ('img','x (um)', 'y (um)', img_norm, 400, 400, um, um)
	nf_attribute	= ('img','x (cm)', 'y (cm)', img_norm, 14, 14, cm, cm)
	ff_data			= (ff_x[0], ff_y[0])
	nf_data			= (nf_x[0], nf_y[0])
	_, xff, _, yff, int_ff_ideal, int_ff_onlyDPP	= util_find_relative_intensity(nf_x, ff_x, nf_y, ff_y,
																			 int_ff_ideal_raw, int_ff_raw)
	if do_ideal:
		(data, attribute) = (nf_data, nf_attribute) if ideal_int_nf else (ff_data, ff_attribute)
		data_to_plot, plt_attributes = util_add_plot_items(data_to_plot, data + (int_ff_ideal,), plt_attributes, attribute + ("IDEAL INT",))
	if do_DPP:
		data_to_plot, plt_attributes = util_add_plot_items(data_to_plot, ff_data + (int_ff_onlyDPP,), plt_attributes, ff_attribute + ("DPP INT",))
	if plot_ssd:
		_, _, _, _, _, int_ff_ssd	= util_find_relative_intensity(nf_x, ff_x, nf_y, ff_y,
														int_ff_ideal_raw, int_ff_ssd_raw)
		data_to_plot, plt_attributes = util_add_plot_items(data_to_plot, ff_data + (int_ff_ssd,),
													 plt_attributes, ff_attribute + (f"DPP SSD INT, t_res: {ssd_time_resolution}\n duration: {ssd_duration:2.0e}s",))
	else: do_PS_SSD=False
	if do_PS:
		if do_PS_SSD: to_smooth, plot_name 	= int_ff_ssd, f"DPP PS SSD INT"
		else: to_smooth, plot_name			= int_ff_onlyDPP, f"DPP PS INT"
		int_ff_DPP_and_PS 		= apply_polarisation_smoothing(to_smooth, shift=ps_shift)					# phase plate and ps applied
		data_to_plot			.append(ff_data + (int_ff_DPP_and_PS,))
		plt_attributes			.append(ff_attribute+ (plot_name,))
	else: use_sciop = False
	if use_sciop:
		ideal_fit_ssd, ideal_fit_ssd_ps, sigma_0_ssd, sigma_0_ssd_ps = util_make_ideal_ff_using_curvefit(xff, yff, int_ff_ssd, int_ff_DPP_and_PS, int_ff_onlyDPP, scale_from_max)
	else:
		ideal_fit_ssd 		= int_ff_ideal
		ideal_fit_ssd_ps 	= int_ff_ideal
		_, sigma_0_ssd		= quantify_nonuniformity(ideal_fit_ssd, int_ff_onlyDPP, scale_from_maximum=scale_from_max)
		sigma_0_ssd_ps		= sigma_0_ssd
	if do_line:
		center 							= int(np.ceil(len(int_ff_ideal[0,:]) * 0.5))
		data_lines, line_designs		= util_add_line_items(int_ff_ideal[center], (), ('--', 'red', 'Ideal lineshape'), [])
		mask_fwhm						= np.isclose(int_ff_ideal[center], np.max(int_ff_ideal[center]) / scale_from_max, rtol=0.005)
		data_lines, line_designs		= util_add_line_items(mask_fwhm, data_lines, ('--', 'purple', 'cutoff'), line_designs)
		if do_DPP:
			data_lines, line_designs	= util_add_line_items(int_ff_onlyDPP[center], data_lines, (':', 'black', 'Only Phase Plate'), line_designs) 
		if do_PS:
			data_lines, line_designs	= util_add_line_items(int_ff_DPP_and_PS[center], data_lines, ('-', 'orange', 'with PS'), line_designs)
		if plot_ssd:
			data_lines, line_designs	= util_add_line_items(int_ff_ssd[center], data_lines,('-.', 'blue', 'with SSD'), line_designs)
		if use_sciop:
			data_lines, line_designs	= util_add_line_items(ideal_fit_ssd_ps[center], data_lines, ('--', 'black', 'Ideal fit'), line_designs)
		data_to_plot, plt_attributes = util_add_plot_items(data_to_plot, (xff, yff,) + (data_lines,),
													 plt_attributes, ('line','x (um)', 'Intensity Normalized', line_norm, 900, (None,None), um, line_designs, 'lineout'))
	num_plts 	= len(data_to_plot)
	(nonuniformity_percentage_PS, _) 	= quantify_nonuniformity(ideal_fit_ssd_ps, int_ff_DPP_and_PS, sigma_0_ssd_ps, scale_from_maximum=scale_from_max) 	if do_PS 	else (-1, _)
	(nonuniformity_percentage_ssd, _) 	= quantify_nonuniformity(ideal_fit_ssd, int_ff_ssd, sigma_0_ssd, scale_from_maximum=scale_from_max) 				if plot_ssd else (-1, _)
	if return_fig: FIGURE = util_plt_one_column_or_row(data_to_plot, plt_attributes, one_col=False) if num_plts < 4 else util_plt_MbyN_grid(data_to_plot, plt_attributes)
	else: FIGURE = None
	return FIGURE, nonuniformity_percentage_PS, nonuniformity_percentage_ssd, 100
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
def util_apply_twoD_hamming_filter(int_dist):
	rows, cols = np.shape(int_dist)
	hamming_y = np.hamming(rows) # the hamming equation in one dimension
	hamming_x = np.hamming(cols)
	two_dimentional_hamming_array = np.outer(hamming_y, hamming_x) # performing an outter product
	return int_dist * two_dimentional_hamming_array
def util_perform_azimuthal_sum_twoD(array, x, y, nbins, k_min=0, do_avg=False, do_normalize=False):
	r 			= np.sqrt(x*x + y*y) 	# distance of each point in the 2D array from center
	
	r 			= r.ravel()				# flattening the 2D array into a single 1D line- all separated rows
	dist 		= array.ravel()			# concatenated one after another in one line

	r_bins 		= np.linspace(np.min(r), np.max(r), nbins+1)  	#) collecting the radii into discrete bins. nbins+1 becuase #edges = #bins+1.
	r_centers	= (r_bins[1:] + r_bins[:-1]) * 0.5
	r_bincount 	= np.digitize(r, bins=r_bins)					#) identifying which bin each radius falls into,
																# 0 for < r_min, nbins+1 for > r_max, 1...nbins for proper bins.
	radial_sum 			= np.bincount(r_bincount, weights=dist, #) sums up all values in dist in the same bin for all bins
									minlength=len(r_bins)+1) 	# and returns array of len r_bincount.
	radial_count 		= np.bincount(r_bincount,				#) counting how many items fall in each bin
							  minlength=len(r_bins)+1)			# minlength set to nbins+2 so we get exactly enough (0, 1->nbins, nbins+1)
	radial_sum, radial_count 	= radial_sum[1:-1], radial_count[1:-1]
	if do_avg:
		mask_not_zero 					= radial_count > 0
		radial_profile					= np.zeros(np.shape(radial_sum))
		radial_profile[mask_not_zero] 	= radial_sum[mask_not_zero] / radial_count[mask_not_zero]
	else: radial_profile = radial_sum
	mask_cutoff				 	= r_centers > k_min
	r_centers					= r_centers[mask_cutoff]
	radial_profile 				= radial_profile[mask_cutoff]
	if do_normalize: radial_profile /= np.max(radial_profile)
	return radial_profile, r_centers
def power_spectrum_against_wavenumber_plots(int_dist_untapered, x, y, norm='linear',
											nbins=1000, other_data=None, k_cutoff=2, k_min=0,
											show_untapered=False, do_avg=True, do_normalize=True, apply_hamming=True):
	int_dist 				= util_apply_twoD_hamming_filter(int_dist_untapered) if apply_hamming else int_dist_untapered
	fourier_amps_squared 	= util_compute_fft(int_dist, do_abs_squared=True)
	dx, dy 					= x[1,0]-x[0,0], y[0,1]-y[0,0]
	kx1d, ky1d				= 2*np.pi * np.fft.fftfreq(x.shape[0], d=dx), 2*np.pi * np.fft.fftfreq(y.shape[0], d=dy)
	kx, ky 					= np.meshgrid(np.fft.fftshift(kx1d), np.fft.fftshift(ky1d), indexing='ij')
	collection 				= []
	if show_untapered:
		power_spectrum, k 	= util_perform_azimuthal_sum_twoD(int_dist, x, y, nbins=nbins, do_avg=do_avg, do_normalize=do_normalize)
		xlabel, ylabel 		= 'Radius (um)', 'Intensity distribution'
		xscale				= um
		center 									= int(np.ceil(len(int_ff[0,:]) * 0.5))
		int_ff_ideal_center_row					= int_dist[center]
		pos_only_mask							= x[:,0] > 0
		int_ff_ideal_center_row					= int_ff_ideal_center_row[pos_only_mask] / np.max(int_ff_ideal_center_row)
		collection.append(
			(
			(int_ff_ideal_center_row, "ff", "Intensity", norm, "line"),  # data_to_plot
			x[:,0][pos_only_mask], None, None, ("-", "blue", "Center lineout"), norm, None, None, xscale, xlabel, ylabel
			)
		)
	else:
		power_spectrum, k 			= util_perform_azimuthal_sum_twoD(fourier_amps_squared, kx, ky, nbins=nbins, k_min=k_min, do_avg=do_avg, do_normalize=do_normalize)
		xlabel, ylabel 				= r'Wavenumber ($\um^{-1}$)', 'Normalized Power'
		xscale						= 1/um
		if other_data:
			for (x, y) in other_data:
				collection.append(
					(
						(y, "ff", "Digitized from Regan et. all", norm, "line"),  # data_to_plot
						x, None, None, ("-", "blue", "Digitized from Regan et. all"), norm, None, None, 1, xlabel, ylabel
					)
				)
	# Prepare plot collection
	collection.append(
		(
			(power_spectrum, "ff", "Power Spectrum", norm, "line"),  # data_to_plot
			k, None, None, ("-", "black", "Spectrum"), norm, (None, k_cutoff), (None, k_cutoff), xscale, xlabel, ylabel
		)
	)
	fig = make_plots(collection, scatter=True, do_top_edge=True, top_items=(500*um*k, 'l mode'), show_legend=False)
	return fig, int_dist
def make_plots(collection, scatter=False, do_top_edge=False, top_items=(None,None), show_legend=True):
	"""
	Create a figure and plot one or more 2D images or line plots from a collection.

	Each item in `collection` contains data and metadata describing whether to plot
	a near-field (nf) or far-field (ff) distribution, and whether the data should be
	shown as an image or a line plot.

	Args:
		collection (list of tuple): Each item should unpack as
			(data_to_plot, x, y, xff, yff, xnorm, nf_lim, ff_lim), where:
			
			* data_to_plot (tuple): (data_array, scope, varname, norm, plt_type)
				- data_array (ndarray): The numerical data to be plotted.
				- scope (str): Either "nf" (near field) or "ff" (far field).
				- varname (str): Label for the colorbar or axis.
				- norm (str or None): "log" for logarithmic normalization, else None.
				- plt_type (str): Either "img" for 2D image or "line" for 1D line plot.
			
			* x, y (ndarray): Grids for near-field spatial coordinates.
			* xff, yff (ndarray): Grids for far-field spatial coordinates.
			* xnorm (str or None): Normalization for x-axis in line plots ("log", etc.).
			* nf_lim (float or tuple): Plot limits for near-field axes.
			* ff_lim (float or tuple): Plot limits for far-field axes.
			* x_scale (float): How much to scale the x-axis by given the data
			* xlabel (str): Name for the x-axis
			* ylabel (str): Name for the y-axis 

	Returns:
		matplotlib.figure.Figure: The figure containing the generated plots.

	Notes:
		- For line plots, `yff` is interpreted as a tuple defining line style,
		  color, and label. For multiple lines, it should be a list of such tuples.
		- Axis labels and scales are automatically assigned based on `scope`
		  and `plt_type`.
	"""
	fig, ax = plt.subplots()
	if type(collection) != list: # just a 2D array was given to be plotted, without any x or y
		util_img_plot(fig, ax, None, None, collection, None, None)
		return fig
	for items in collection:
		if type(collection[-1]) != tuple: items=tuple(collection)
		data_to_plot, x, y, xff, yff, xnorm, nf_lim, ff_lim, x_scale, xlabel, ylabel= items
		data_array, scope, varname, norm, plt_type 									= data_to_plot
													#line_style, colour, label
				# this is weird, but when ploting lines, yff stores the line design as a tuple
		yscale = [yff, yff] if plt_type == 'line' else [um, cm]
		xscale = [x_scale, x_scale] 	if plt_type == 'line' else [um, cm]
		# xlabel = 'time (ns)' 	if plt_type == 'line' else 'x'
		# ylabel = scope 			if plt_type == 'line' else 'y'
	# (plot_type, xlabel, ylabel, norm, xlim, ylim, xscale, ysclae, label/title)
		ff_attribute	= (plt_type, xlabel, ylabel, norm, ff_lim, ff_lim, xscale[0], yscale[0])
		nf_attribute	= (plt_type, xlabel, ylabel, norm, nf_lim, nf_lim, xscale[1], yscale[1])
		
		attribute 			= ff_attribute + (varname,) if scope == 'ff' else nf_attribute + (varname,)
		(x_grid, y_grid)	= (xff, yff)				if scope == 'ff' else (x, y)
		if plt_type 	== 'img'	: util_img_plot(fig, ax, x_grid, y_grid, data_array, attribute)
		elif plt_type 	== 'line'	: util_line_plot(ax, x, y, data_array, attribute, xnorm=xnorm, do_scatter=scatter, do_top_edge=do_top_edge, top_items=top_items, show_legend=show_legend)
		if type(collection[-1]) != tuple: break
	return fig
def plot_moving_speckels(int_beam_env,dpp_phase, nf_x, ff_x, nf_y, ff_y, time, time_resolution, area_ff, beam_power, nf_lim=14, ff_lim=100, img_norm=None):
	plt.ion()		
	fig, ax			= plt.subplots()
	# fig, ax			= plt.subplots(nrows=2, ncols=1)
	ff_attribute	= ('img','x (um)', 'y (um)', img_norm, ff_lim, ff_lim, um, um, '')
	nf_attribute	= ('img','x (cm)', 'y (cm)', img_norm, nf_lim, nf_lim, cm, cm, '')
	x, y = ff_x[0], ff_y[0]
	near_field		= np.exp(-1j) * np.exp(1j) * (int_beam_env)**0.5
# applying ssd and finding far field intensity distribution
	dpp				= np.exp(1j * dpp_phase)					# complex number of unit modulus. Product simply changes phase
	near_field		*= dpp										# apply DPP phase modulation
	int_ff_onlyDPP	= util_compute_fft(near_field)
	int_ff_onlyDPP 	= util_scale_int_env(int_beam_env=int_ff_onlyDPP, area=area_ff, beam_power=beam_power)
	_, _, _, _, int_ideal, int_ff_onlyDPP		= util_find_relative_intensity(nf_x, ff_x, nf_y, ff_y,
													int_beam_env, int_ff_onlyDPP)
	pcm				= util_img_plot(fig, ax, x, y, int_ff_onlyDPP, ff_attribute)
	timesteps 		= [i*(time / time_resolution) for i in range(time_resolution)] # dividing the total time into discrete steps
	for timestep, near_field in tqdm(enumerate(apply_smoothing_by_spectral_dispersion(
		E_near_field=near_field, x_mesh=x, y_mesh=y, timesteps=timesteps, time_resolution=time_resolution, use_gen=True
		)), desc="Applying smoothing by spectral dispersion smoothing"):
		int_ff_ssd_instantaneous				= util_compute_fft(near_field)
		int_ff_ssd_instantaneous				= util_scale_int_env(int_beam_env=int_ff_ssd_instantaneous, area=area_ff, beam_power=beam_power)
		_, _, _, _, _, int_ff_ssd_instantaneous	= util_find_relative_intensity(nf_x, ff_x, nf_y, ff_y,
														int_beam_env, int_ff_ssd_instantaneous)
		plt.pause(1)
		pcm.set_array(int_ff_ssd_instantaneous)
		fig.canvas.draw()
	return
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
	nonuni_DPP, nonuni_DPP_PS, nonuni_DDP_SSD, ssd_time_resoution, ssd_time_duration) = tuple(var_list)
	print(f"""Far field grid scales area: {dxff/um:2.2f} microns by {dyff/um:2.2f} microns, speckle scale is {foc_len/(dx*nx)*Lambda/um:2.2f} microns

Power in near field {pwr_beam_env_tot*1e-12:2.2e} TW, power in far field = {pwr_ff_tot* 1e-12:2.2e}TW,

total intensity in near field {np.sum(int_beam_env) * 1e-12:2.2e} TW/m^2, total intensity in far field = {np.sum(int_ff)* 1e-12:2.2e} TW/m^2

area_nearfield: {area_nf:2.2e}, area_farfield: {area_ff:2.2e}

grid scale near field: {nx*dx:2.2f}m by {ny*dy:2.2f}m, focal length: {foc_len}m

f_num_x = {foc_len/(nx*dx)}, f_num_y = {foc_len/(ny*dy)}, f_number: {f_number}

Nonuniformity with only DPP: {nonuni_DPP:3.2f}% (by definition), with ssd: {nonuni_DDP_SSD:3.2f}%, with PS: {nonuni_DPP_PS:3.2f}%,  ssd time resolution: {ssd_time_resoution}, ssd time duration: {ssd_time_duration}""")
def test_func(x, y, omega=1):
	return np.sin(
			omega * np.sqrt(x*x + y*y)
	)
def read_csv_file(file_loc):
	df = pd.read_csv(file_loc)
	x = df["x"]
	y = df["y"]
	return x, y
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
if __name__ == "__main__":
# global variables
	m, cm, um, cm2, um2  = 1, 1e-2, 1e-6, 1e-4, 1e-12	# unit conversions

	beam_power       = 0.5e12       					# Watts     # in "Initial...restuls...Omega..."~Boehly et. all, it says 60TW 60-beam
	foc_len          = 1.8*m        					# Omega final lens focal distance
	f_number         = 6.5          					# Omega's f number
	Lambda           = 0.351*um     					# Omega wavelength: 3rd harmonic of a Nd:Glass laser
	res_fac          = 1            					# Note: Higher resolution is required to resolve the far-field speckles
# extracting phaseplate data and making grids
	dpp_phase    							= fmat['DPP_phase']
	KesslerParms 							= fmat['KesslerParms']
	nx, dx, ny, dy, x1d, y1d, (x,y) 		= create_2D_grids(KesslerParms)
	nx_, dx_, ny_, dy_, x1d_, y1d_, (x_,y_) = create_2D_grids(KesslerParms, resolution_fac=res_fac) # fine grid for high-res interpolation
	phase_func   							= RectBivariateSpline(x1d,y1d,dpp_phase)				# interpolate the phase array from the phase plate
	dpp_phase    							= phase_func(x1d_,y1d_)
	use_periodic_phase 						= True
	if use_periodic_phase:
		dpp_phase = make_periodic_phase(dpp_phase)
	nx, dx, x1d, ny, dy, y1d, (x,y), res_fac_pr = nx_, dx_, x1d_, ny_, dy_, y1d_, (x_,y_), res_fac
# initial envelope
	area_nf										= dx * dy
	xff1d, dxff, xff, yff1d, dyff, yff, area_ff = util_calc_ff_parameters(foc_len, Lambda, nx, dx, ny, dy)
	int_beam_env_nf								= beam_envelope(alpha=np.log(0.5), sigma= (25.8*cm) / 2,
																   cords=(x,y), N=24/2)											# Beam intensity envelope at lens
	int_beam_env_nf								= util_scale_int_env(int_beam_env_nf, area_nf, beam_power)
# expanding grids
	expandGrids = True
	scale_factor								= 2
	if expandGrids:
		nx, dx, ny, dy, x1d, y1d, (x,y) 			= create_2D_grids(KesslerParms, scale_factor=scale_factor)
		xff1d, dxff, xff, yff1d, dyff, yff, area_ff = util_calc_ff_parameters(foc_len, Lambda, nx, dx, ny, dy)
		int_beam_env_nf, dpp_phase	= tuple(expand_grid([int_beam_env_nf, dpp_phase], scale_factor=scale_factor))
	int_beam_env = int_beam_env_nf
	super_g_zero								= 358e-4*cm  # scale_factor*358e-4*cm if expandGrids else 
	int_beam_env_ff								= beam_envelope(alpha=-1,cords=(xff,yff),
																  super_g_x_zero=super_g_zero, super_g_y_zero=super_g_zero, N=4.77/2)	# same env as visrad and pyodin
	int_beam_env_ff								= util_scale_int_env(int_beam_env_ff, area_ff, beam_power)
# performing ffts
	int_ff, far_field_ideal, int_ff_dpp, _, _	= perform_fft_normalize(int_beam_env=int_beam_env, beam_power=beam_power, area_nf=area_nf, area_ff=area_ff,
												dpp_phase=dpp_phase, do_ssd=False)
	scale_from_maximum							= 1.1
#%%
# applying ssd
	do_ssd				= False
	ssd_time_resolution	= 500
	ssd_duration		= 1e-9
	ssd_timesteps		= np.array([i*(ssd_duration / ssd_time_resolution) for i in range(ssd_time_resolution)])
	int_ff_ssd, _, _, sigma_rms_ssd, sigma_rms_ssd_ps = perform_fft_normalize(int_beam_env=int_beam_env, beam_power=beam_power,area_nf=area_nf, area_ff=area_ff,
													dpp_phase=dpp_phase, do_dpp=True, do_ssd=do_ssd, scale_from_max=scale_from_maximum, int_ideal_ff=int_beam_env_ff,
													x=x, y=y, time=ssd_duration, time_resolution=ssd_time_resolution, write_to_file=False)
#%%
# this must be done after the previous perform_fft_normalize() function call becuase we are changing int_beam_env
	pwr_beam_env_tot	= np.sum(int_beam_env * area_nf)
	pwr_ff_tot      	= np.sum(int_ff * area_ff)
# making figures
	# fig = make_big_figure(x, y, nx, ny, x1d, y1d, xff, yff, dxff, dyff,
	# 				   dpp_phase, int_beam_env, int_ff, pwr_ff_tot, int_beam_env_nf,
	# 				   cm, cm2, um)
	# FIGURE1_PHASE_INTENSITY = intensity_phase_plots((x1d,dx), (xff1d,dxff), (y1d,dy), (yff1d,dyff), far_field_ideal, int_ff, dpp_phase)
	nonuniform_DPP, nonuniform_DPP_and_PS, nonuniformity_ssd = -1, -1, -1
	# ONLY_INTENSITY, nonuniform_DPP_and_PS, nonuniformity_ssd, nonuniform_DPP = intensity_plot_old(nf_x=(x1d,dx), ff_x=(xff1d,dxff), nf_y=(y1d,dy), ff_y=(yff1d,dyff),
	# 																	int_ff_ideal_raw=int_beam_env_ff, int_ff_raw=int_ff, ideal_int_nf=False,
	# 																	ssd_data_items=(int_ff_ssd, ssd_time_resolution, ssd_duration),
	# 																	do_DPP=False, do_PS=True, plot_ssd=do_ssd, do_PS_SSD=False, ps_shift=5,
	# 																	do_line=True, do_LogNorm=False, do_ideal=False, return_fig=True, use_sciop=False, scale_from_max=scale_from_maximum)
	# t_type, varname, norm
	# KICKER = make_plots([(int_beam_env_nf, 'ff', 'INT ENV NEW', 'linear', 'img'), x, y, xff, yff, 'linear', 15, 330, None, 'x', 'y'])
	# ssd_duration, ssd_time_resolution = 1e-8, 50000
	# ssd_timesteps		= np.array([i*(ssd_duration / ssd_time_resolution) for i in range(ssd_time_resolution)])
	# df = pd.read_csv("./sigma_rms_ssd_5e4_timeres_1e-8_SSDduration_expandedGrids_3_scale_factor-OnePointOne.txt", sep="\t")
	# sigma_rms_ssd, sigma_rms_ssd_ps = df['sigma_rms_ssd'], df['sigma_rms_ssd_ps']
	# max = np.max(sigma_rms_ssd)
	# sigma_rms_ssd /= max
	# sigma_rms_ssd_ps /= max
	# (x_sigma_rms_ssd, y_sigma_rms_ssd) = read_csv_file("../Graphs digitized/Digitised-Regan_et_all-1THz-2DSSD-withoutPS.csv")
	# (x_sigma_rms_ssd_ps, y_sigma_rms_ssd_ps) = read_csv_file("../Graphs digitized/Digitised-Regan_et_all-1THz-2DSSD-withPS.csv")
	# collection=[
		# ((sigma_rms_ssd, 'ff', f'sgma_rms\nssd_duration: {ssd_duration} s\nssd_time_resolution: {ssd_time_resolution}', 'log', 'line'),
		# 			 ssd_timesteps, None, ssd_timesteps, (':', 'black', 'ssd only'), 'log', None, None, 1e-9, 'time (ns)', r'$\sigma_{\mathrm{rms}}$ SSD'),
		# ((sigma_rms_ssd_ps, 'ff', f'sgma_rms\nssd_duration: {ssd_duration} s\nssd_time_resolution: {ssd_time_resolution}', 'log', 'line'),
		# 			 ssd_timesteps, None, ssd_timesteps, ('-', 'grey', 'ssd with ps'), 'log', None, None, 1e-9, 'time (ns)', r'$\sigma_{\mathrm{rms}}$ SSD and PS'),
		# ((y_sigma_rms_ssd, 'ff', f'Digitized from Regan et all, with PS', 'log', 'line'),
		# 			 x_sigma_rms_ssd*1e-9, None, x_sigma_rms_ssd, ('--', 'red', 'without PS digitised'), 'log', None, None, 1e-9, 'time (ns)', r'$\sigma_{\mathrm{rms}}$'),
	# 	((y_sigma_rms_ssd_ps, 'ff', f'Digitized from Regan et all, with PS', 'log', 'line'),
	# 				 x_sigma_rms_ssd_ps*1e-9, None, x_sigma_rms_ssd_ps, ('--', 'blue', 'with PS digitised'), 'log', None, None, 1e-9, 'time (ns)', r'$\sigma_{\mathrm{rms}}$')
	# 				 ]
	# FIGURE = make_plots(collection, scatter=True, show_legend=False)
	# test_data = test_func(xff, yff, omega=1e4)
	# fig = make_plots(test_data)
	# (x_noPS, y_noPS) = read_csv_file("../Graphs digitized/Figure3-Power soectra calculated from UVETP  imgages without PS-Measured.csv")
	# POWER_SPEC_ONLY_SSD = power_spectrum_against_wavenumber_plots(int_ff_ssd, xff, yff, norm='log', nbins=1000,
	# 														   other_data=[(x_noPS, y_noPS)], k_min=2e-2 / um, k_cutoff=2.4,
	# 														   show_untapered=False, do_avg=True, do_normalize=True, apply_hamming=True)
	# int_ff_ssd_ps		= apply_polarisation_smoothing(int_ff_ssd)
	# (x_withPS, y_withPS) = read_csv_file("../Graphs digitized/Figure3-Power soectra calculated from UVETP  imgages with PS-Measured.csv")
	# POWER_SPEC_SSD_PS = power_spectrum_against_wavenumber_plots(int_ff_ssd, xff, yff, norm='log', nbins=1000,
	# 														   other_data=[(x_noPS, y_noPS)], k_min=2e-2 / um, k_cutoff=2.4,
	# 														   show_untapered=False, do_avg=True, do_normalize=True, apply_hamming=True)
	plot_moving_speckels(int_beam_env=int_beam_env, dpp_phase=dpp_phase,
					  nf_x=(x1d,dx), ff_x=(xff1d,dxff), nf_y=(y1d,dy), ff_y=(yff1d,dyff),
					  time=1e-9, time_resolution=100, area_ff=area_ff, beam_power=beam_power, ff_lim=10, img_norm='linear')
# printing key results
	# var_list = [nx, dx, dxff, ny, dy, dyff,
	# 			area_nf, area_ff,
	# 			int_beam_env, int_ff, pwr_beam_env_tot, pwr_ff_tot,
	# 			foc_len, Lambda,
	# 			nonuniform_DPP, nonuniform_DPP_and_PS, nonuniformity_ssd, ssd_time_resolution, ssd_duration]
	# print_function(var_list)
	# plt.tight_layout()
	plt.show()

#%%
#%%