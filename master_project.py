'''
Description:
This script contains various functions to calculate the +/- waveform and kick information from it.
The NRSur7dq4 model or SXS waveforms, for which the data has previously been downloaded, can be used.

Usage:
Make sure you have downloaded all modules in your environment and
save this script in a folder of your choose and import in a Jupyter notebook via
    import sys
    sys.path.append('/home/jannik/scripts')
    from master_code import *

Author:
Jannik Mielke
Max-Planck-Institut für Gravitationsphysik (Albert-Einstein-Institut)
jannik.mielke@aei.mpg.de

'''


import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simps
from scipy import signal
from scipy import interpolate
import gwsurrogate as gws
import gwtools
import surfinBH
import lal
import lalsimulation as lalsim
import scri
import h5py
import json


# load matplotlib style sheet (if you have one)
plt.style.use("~/MA/fertige-MA/gitrepo/MATPLOTLIB_RCPARAMS.sty")
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}' 


# define Surrogate models
fit_name = 'NRSur7dq4Remnant'
fit = surfinBH.LoadFits(fit_name)
sur = gws.LoadSurrogate('NRSur7dq4')



#%env LAL_DATA_PATH=/home/jannik/anaconda3/envs/lalsuiteenv/src/lalsuite-extra/data/lalsimulation


# global parameters
dt = 0.1        # timestep size, Units of M
f_low = 0       # initial frequency, f_low=0 returns the full surrogate



def h_asym(h, l, m):
    '''
    Returns the anti-symmetric waveform 
    h:  mode dictionary with (l, m) tuples as keys (dict)
    l:  greater equal 2 (int) 
    m:  greater equal 1 less equal l (int)
    '''
    return (h[(l,m)] - (-1)**l * np.conjugate(h[(l,-m)]))/2



def ampl_asym(h, l, m):
    '''
    Returns the amplitude of the anti-symmetric waveform
    h:  mode dictionary with (l, m) tuples as keys (dict)
    l:  greater equal 2 (int) 
    m:  greater equal 1 less equal l (int)
    '''
    a = np.abs(h_asym(h, l, m))
    return a



def phi_asym(h, l, m):
    '''
    Returns the phase of the anti-symmetric waveform 
    h:  mode dictionary with (l, m) tuples as keys (dict)
    l:  greater equal 2 (int) 
    m:  greater equal 1 less equal l (int)
    '''
    phi_a = np.unwrap(np.angle(h_asym(h, l, m)))
    return phi_a


    
def h_sym(h, l, m):
    '''
    Returns the symmetric waveform 
    h:  mode dictionary with (l, m) tuples as keys (dict)
    l:  greater equal 2 (int) 
    m:  greater equal 1 less equal l (int)
    '''
    return (h[(l,m)] + (-1)**l * np.conjugate(h[(l,-m)]))/2    



def ampl_sym(h, l, m):
    '''
    Returns the amplitude of the symmetric waveform 
    h:  mode dictionary with (l, m) tuples as keys (dict)
    l:  greater equal 2 (int) 
    m:  greater equal 1 less equal l (int)
    '''
    A = np.abs(h_sym(h, l, m)) 
    return A



def phi_sym(h, l, m):
    '''
    Returns the phase of the symmetric waveform 
    h:  mode dictionary with (l, m) tuples as keys (dict)
    l:  greater equal 2 (int) 
    m:  greater equal 1 less equal l (int)
    '''
    phi_s = np.unwrap(np.angle(h_sym(h, l, m))) 
    return phi_s   



def get_time_in_M_units(t, m1, m2):
    '''
    Gives a time array in units of total mass when masses of both BH are given in solar masses
    t:   time array in units of sec (arr)
    m1:  mass of BH1 in solar masses (int)
    m2:  mass of BH2 in solar masses (int)
    '''
    M = (m1 + m2)*lal.MTSUN_SI
    t_in_M = t/M
    return t_in_M 
    

    
def get_time_in_sec(t, m1, m2):
    '''
    Gives a time array in units of sec when masses of both BH are given in solar masses
    t:   time array in units of M (arr)
    m1:  mass of BH1 in solar masses (float)
    m2:  mass of BH2 in solar masses (float)
    '''
    M = (m1 + m2)*lal.MTSUN_SI
    t_in_sec = t*M
    return t_in_sec
    

    
def shift_time_to_zero(t):
    '''
    Set the first value of a time array to zero
    t:  time array (arr)
    '''
    start_time = t[0]
    if start_time > 0:
        t += start_time
    else:
        t -= start_time
    return t
    
    
    
def get_strain_in_SI_from_fni(h, m1, m2, dist):
    '''
    Gives a strain array in units of meter 
    h:     strain in units of r/M (arr)
    m1:    mass of BH1 in solar masses (float)
    m2:    mass of BH2 in solar masses (float)
    dist:  distance source to earth in parsec (float)
    '''
    m1_in_meter = m1*lal.MTSUN_SI*lal.C_SI
    m2_in_meter = m2*lal.MTSUN_SI*lal.C_SI
    dist_in_meter = dist*lal.PC_SI 

    return h*(m1_in_meter + m2_in_meter)/dist_in_meter

    
    
def get_strain_in_fni_from_SI(h, m1, m2, dist):
    '''
    Gives a strain array in units of r/M
    h:     strain in units of meter (arr)
    m1:    mass of BH1 in solar masses (float)
    m2:    mass of BH2 in solar masses (float)
    dist:  distance source to earth in parsec (float)
    '''
    m1_in_meter = m1*lal.MTSUN_SI*lal.C_SI
    m2_in_meter = m2*lal.MTSUN_SI*lal.C_SI
    dist_in_meter = dist*lal.PC_SI 

    return h*dist_in_meter/(m1_in_meter + m2_in_meter)
    

    
def f_ref_for_NRSur7dq4_from_Hz(f_ref_in_Hz, m1, m2):
    '''
    Transforms a reference frequency in units of Hz to the reference frequency in units of cycles/M, 
    which is needed to set the reference epoch in NRSur7dq4  
    '''
    M = (m1 + m2)*lal.MTSUN_SI
    f_ref = f_ref_in_Hz*M
    
    return f_ref

  
    
def omega_0_for_NRSur7dq4Remnant_from_Hz(f_ref_in_Hz, m1, m2):
    '''
    Transforms a reference frequency in units of Hz to the reference frequency in units of rad/M, 
    which is needed to set the reference epoch in NRSur7dq4Remnant  
    '''
    f_ref = f_ref_for_NRSur7dq4_from_Hz(f_ref_in_Hz, m1, m2)*np.pi
    
    return f_ref

    
    
def get_f_ref(q, chi1, chi2, t_ref):
    '''
    Transforms a reference time in units of M to the reference frequency in units of cycles/M, 
    which is needed to set the reference epoch in NRSur7dq4  
    q:     mass ratio (float)
    chi1:  dimensionless spin of heavier BH (arr)
    chi2:  dimensionless spin of lighter BH (arr)
    t_ref: reference time to define spins (float)
    '''
    
    # calculate surrogate waveform for given configuration
    dt = 0.1        # in M
    f_low = 0       # initial frequency, f_low=0 returns the full surrogate 
    time, h, dyn = sur(q, chi1, chi2, dt=dt, f_low=f_low) 
    
    # calculate h in coprecessing frame
    h_copr = modes_from_iner_to_new_frame(time, h, 'coprecessing')
    
    # get reference freq from GW phase
    phi_GW = 1/2 * (np.unwrap(np.angle(h_copr[(2,-2)])) - np.unwrap(np.angle(h_copr[(2,2)])))    # in units of rad
    phi_GW_dot = np.diff(phi_GW) / dt                                                            # in units of rad/M
    phi_GW_dot = np.append(phi_GW_dot[0], phi_GW_dot)                                            # in units of cycles/M 
    f_ref = phi_GW_dot / (2*np.pi)                                                               # in units of cycles/M 
    
    # find time node index in near of t_ref
    iref = np.argmin(np.abs(time - t_ref))
    
    return f_ref[iref]



def omega_0_for_NRSur7dq4Remnant(q, chi1, chi2, t_ref):
    '''
    Transforms a reference time in units of M to the orbital frequency in units of rad/M, 
    which is needed to set the reference epoch in NRSur7dq4Remnant 
    '''
    # calculate surrogate waveform for given configuration
    dt = 0.1        # in M
    f_low = 0       # initial frequency, f_low=0 returns the full surrogate 
    time, h, dyn = sur(q, chi1, chi2, dt=dt, f_low=f_low, precessing_opts={'return_dynamics': True}) 
    
    phi_orb = dyn['orbphase'][:,]                         # in units of rad
    omega0 = np.diff(phi_orb) / dt                        # omega0 = dphi_orb/dt in units of rad/M
    omega0 = np.append(omega0[0], omega0)
    
    # get index of the input reference time (first index of time point > given t_ref)
    indx = 0 
    while time[indx] < t_ref: 
        indx += 1
    
    return np.abs(omega0[indx])
    
    
    
def analyse_lkha_hkha(q, chi1, chi2, t_ref):
    '''
    Shrobana master equation
    '''
    # calculate surrogate waveform 
    f_ref = f_ref_for_NRSur7dq4(q, chi1, chi2, t_ref)
    omega0 = omega_0_for_NRSur7dq4Remnant(q, chi1, chi2, t_ref)
    t, h, dyn = sur(q, chi1, chi2, dt=dt, f_low=f_low, f_ref=f_ref, precessing_opts={'return_dynamics': True}) 
    
    # calculate quantities
    a = ampl_asym(h, 2, 2)
    phi_a = phi_asym(h, 2, 2)
    A = ampl_sym(h, 2, 2)
    phi_s = phi_sym(h, 2, 2)
    Phi = dyn['orbphase'][:,]    # -phi_sym(h)/2
    
    # calculate derivates of quantities
    a_dot = np.diff(a) / dt
    a_dot = np.append(a_dot[0], a_dot)
    phi_a_dot = np.diff(phi_a) / dt 
    phi_a_dot = np.append(phi_a_dot[0], phi_a_dot)
    A_dot = np.diff(A) / dt 
    A_dot = np.append(A_dot[0], A_dot)
    Phi_dot = np.diff(Phi) / dt  
    Phi_dot = np.append(Phi_dot[0], Phi_dot)
    
    # Shrobana master eq. and integration
    term1 = (a_dot*A_dot - 2*a*A*phi_a_dot*Phi_dot)*np.cos(phi_a + 2*Phi)
    term2 = - (a*A_dot*phi_a_dot + 2*a_dot*A*Phi_dot)*np.sin(phi_a + 2*Phi) 
    dPzdt = -1/(6*np.pi) * (term1 + term2)
    Pz = simps(dPzdt, dx=dt)
    
    # calculate velocity from NRSur7dq4Remnant
    vf, vf_err = fit.vf(q, chi1, chi2, omega0=omega0) 
    
    return t, a, phi_a, A, phi_s, Phi, term1, term2, dPzdt, Pz, vf
    
    
    
def modes_from_iner_to_new_frame(t, h, frame, ell_min, ell_max):
    '''
    Transforms a strain in the inertial frame into the coprecessing or corotating frame
    t:       time array (arr)
    h:       dictionary of available modes with (l, m) tuples as keys in inertial frame (dict)
    frame:   'coprecessing' or 'corotating' frame (str)
    '''
    
    # available NRSur7dq4 modes
    mode_list = [(ell,m) for ell in range(ell_min, ell_max+1) for m in range(-ell,ell+1)]

    # build scri WaveformModes object
    data = list(h.values())
    data = np.array(data).T
    waveform_modes = scri.WaveformModes(
                            dataType=scri.h,
                            t=t,
                            data=data,
                            ell_min=ell_min,
                            ell_max=ell_max,
                            frameType=scri.Inertial,
                            r_is_scaled_out=True,
                            m_is_scaled_out=True
                            )
    
    # apply scri transformation functions
    if frame == 'coprecessing':
        waveform_modes.to_coprecessing_frame()
        data_in_new_frame = waveform_modes.data.T
        
    if frame == 'corotating':
        waveform_modes.to_corotating_frame()
        data_in_new_frame = waveform_modes.data.T 
    
    # build dictonary similar to input form
    h_in_new_frame = dict(zip(mode_list, data_in_new_frame))
    
    return h_in_new_frame
    
    
    
def access_SXS_metadata(file_directory):
    '''
    returns intrinsic BBH parameters from SXS file 
    file_directory: path of the directory with the SXS metadata json-file (str)
    '''
    
    metadata_path = file_directory + '/metadata.json'
    with open(metadata_path) as file: 
        metadata = json.load(file)
        
    q = metadata['reference_mass_ratio']
    chi1 = metadata['reference_dimensionless_spin1']
    chi2 = metadata['reference_dimensionless_spin2']
    f_ref_orb = np.linalg.norm(metadata['reference_orbital_frequency'])
    t_peak = metadata['common_horizon_time']
    t_ref = metadata['reference_time']
    
    return q, chi1, chi2, f_ref_orb, t_peak, t_ref
    
    
    
def create_SXS_hlm_dict(file_directory):
    '''
    returns a mode dictonary with (l, m) tuples as keys from SXS file
    file_directory: path of the directory with the SXS rhOverM_Asymptotic_GeometricUnits_CoM H5-file (str)
    '''
    # load rhM file
    rh_data = h5py.File(file_directory + '/rhOverM_Asymptotic_GeometricUnits_CoM.h5', 'r')
    
    # get shifted and sliced time from 2,2 mode
    q, chi1, chi2, f_ref_orb, t_peak, t_ref = access_SXS_metadata(file_directory)
    t_SXS = rh_data['Extrapolated_N2.dir']['Y_l2_m2.dat'][:,0]
    idx_cut = np.argmin(np.abs(t_SXS - t_ref))
    t = (t_SXS - t_peak)[idx_cut:] 
    
    # get sliced mode array dictonary
    ell_min = 2
    ell_max = 8
    mode_list = [(ell,m) for ell in range(ell_min, ell_max+1) for m in range(-ell,ell+1)]
    hlm = []
    for lm in mode_list:
        l = str(lm[0])
        m = str(lm[1])
        idx = 'Extrapolated_N4.dir/Y_l'+l+'_m'+m+'.dat'
        rh = rh_data[idx][:,1] + 1j*rh_data[idx][:,2] 
        hlm.append(rh[idx_cut:])
    h = dict(zip(mode_list, hlm))
    
    return t, h
    
    
    

    
    
    
