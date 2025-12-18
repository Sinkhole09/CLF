from functions_utils import *

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
	ssd_time_resolution	= 50
	ssd_duration		= 1e-11
	ssd_timesteps		= np.array([i*(ssd_duration / ssd_time_resolution) for i in range(ssd_time_resolution)])
	int_ff_ssd, _, _, sigma_rms_ssd, sigma_rms_ssd_ps = perform_fft_normalize(int_beam_env=int_beam_env, beam_power=beam_power,area_nf=area_nf, area_ff=area_ff,
													dpp_phase=dpp_phase, do_dpp=True, do_ssd=do_ssd, scale_from_max=scale_from_maximum, int_ideal_ff=int_beam_env_ff,
													time=ssd_duration, time_resolution=ssd_time_resolution,
													x=x, y=y, write_to_file=False)
# applying isi.
	do_isi					= True
	carrier_frequency		= 299729458 / (351e-9)		# angular frequency of 351nm UV light
	bandwidth				= 0.02 * carrier_frequency	# in radians, given as a percentage of central frequency
	isi_duration			= 1e-12
	isi_time_resolution		= int(isi_duration // (1 / bandwidth))
	echelon_block_width		= 16						# pixels
	int_ff_isi, _, _, _, _	= perform_fft_normalize(int_beam_env=int_beam_env, int_ideal_ff=int_beam_env_ff,
							beam_power=beam_power,area_nf=area_nf, area_ff=area_ff, dpp_phase=dpp_phase,
							time=isi_duration, time_resolution=isi_time_resolution, do_isi=True, sf=scale_factor,
							carrier_freq=carrier_frequency, bandwidth=bandwidth, echelon_block_width=echelon_block_width,
							x=x, y=y)
#%%
# this must be done after the previous perform_fft_normalize() function call becuase we are changing int_beam_env
	pwr_beam_env_tot	= np.sum(int_beam_env * area_nf)
	pwr_ff_tot      	= np.sum(int_ff * area_ff)
# making figures
	# fig = make_big_figure(x, y, nx, ny, x1d, y1d, xff, yff, dxff, dyff,
	# 				   dpp_phase, int_beam_env, int_ff, pwr_ff_tot, int_beam_env_nf,
	# 				   cm, cm2, um)
	# FIGURE1_PHASE_INTENSITY = intensity_phase_plots((x1d,dx), (xff1d,dxff), (y1d,dy), (yff1d,dyff), far_field_ideal, int_ff, dpp_phase)
	
	# plotting comparison with ssd
	nonuniform_DPP, nonuniform_DPP_and_PS, nonuniformity_ssd = -1, -1, -1
	INTENSITY_COMPARISON_SSD, nonuniform_DPP_and_PS, nonuniformity_ssd, nonuniform_DPP = intensity_plot_old(nf_x=(x1d,dx), ff_x=(xff1d,dxff), nf_y=(y1d,dy), ff_y=(yff1d,dyff),
									int_ff_ideal_raw=int_beam_env_ff, int_ff_raw=int_ff, ideal_int_nf=False,
									ssd_data_items=(int_ff_ssd, ssd_time_resolution, ssd_duration),
									do_DPP=True, do_PS=False, plot_ssd=do_ssd, do_PS_SSD=True, ps_shift=1,
									do_line=True, do_LogNorm=False, do_ideal=True, return_fig=False, use_sciop=True, scale_from_max=scale_from_maximum)
	# plotting comparison with isi
	INTENSITY_COMPARISON_ISI, nonuniform_DPP_and_PS, nonuniformity_ssd, nonuniform_DPP = intensity_plot_old(nf_x=(x1d,dx), ff_x=(xff1d,dxff), nf_y=(y1d,dy), ff_y=(yff1d,dyff),
									int_ff_ideal_raw=int_beam_env_ff, int_ff_raw=int_ff, ideal_int_nf=False,
									ssd_data_items=(int_ff_isi, isi_time_resolution, isi_duration),
									do_DPP=True, do_PS=False, plot_ssd=do_isi, do_PS_SSD=False, ps_shift=1,
									do_line=False, do_LogNorm=False, do_ideal=True, return_fig=True, use_sciop=False, scale_from_max=scale_from_maximum)
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
	# plot_moving_speckels(int_beam_env=int_beam_env, int_ff_env=int_beam_env_ff, dpp_phase=dpp_phase,
	# 				  nf_x=(x1d,dx), ff_x=(xff1d,dxff), nf_y=(y1d,dy), ff_y=(yff1d,dyff),
	# 				  time=4e-10, time_resolution=400, area_ff=area_ff, beam_power=beam_power,
	# 				  ff_lim=400, img_norm='linear', do_cumulative=False, make_movie=False, fps=4, show_com=True)
# printing key results
	var_list = [nx, dx, dxff, ny, dy, dyff,
				area_nf, area_ff,
				int_beam_env, int_ff, pwr_beam_env_tot, pwr_ff_tot,
				foc_len, Lambda,
				nonuniform_DPP, nonuniform_DPP_and_PS, nonuniformity_ssd, ssd_time_resolution, ssd_duration]
	print_function(var_list)
	plt.tight_layout()
	plt.show()

#%%
#%%