#!/usr/bin/env python3

##################################################################
## MODULES
##################################################################
import os
import h5py

## always import the c-version of pickle
try: import cPickle as pickle
except ModuleNotFoundError: import pickle

from tqdm.auto import tqdm # progress bar
from scipy.optimize import curve_fit, root_scalar

## user defined libraries
from the_useful_library import *


##################################################################
## FUNCTIONS
##################################################################
def reformatFLASHField(field, num_blocks, num_procs):
    """ reformatField
    PURPOSE:
        This code reformats FLASH (HDF5) simulation data from:
            [iprocs*jprocs*kprocs, nzb, nxb, nyb] -> [simulation sub-domain number, sub-domain z-data, sub-domain x-data, sub-domain y-data]
        format to:
            [kprocs*nzb, iprocs*nxb, jprocs*nyb] -> full domain [z-data, x-data, y-data]
        for processing / visualization.
    INPUTS:
        > field      : the FLASH field.
        > num_blocks : number of pixels in each spatial direction [i, j, k] of the simulation sub-domain being simulated by each processor.
        > num_procs  : number of processors between which each spatial direction [i, j, k] of the total-simulation domain is divided.
    BASED ON: James Beattie's code (26 November 2019).
    """
    ## extract number of blocks the simulation was divided into in each [i, j, k] direction
    nxb = num_blocks[0]
    nyb = num_blocks[1]
    nzb = num_blocks[2]
    ## extract number of pixels each block had in each [i, j, k] direction
    iprocs = num_procs[0]
    jprocs = num_procs[1]
    kprocs = num_procs[2]
    ## initialise the output, organised field
    field_sorted = np.zeros([nxb*iprocs, nyb*jprocs, nzb*kprocs])
    ## sort and store the unsorted field into the output field
    for k in range(kprocs):
        for j in range(jprocs):
            for i in range(iprocs):
                field_sorted[k*nzb:(k+1)*nzb, i*nxb:(i+1)*nxb, j*nyb:(j+1)*nyb] = field[j + i*iprocs + k*jprocs*jprocs]
    return field_sorted


def loadFLASH3DFieldMag(filepath_data, num_blocks, num_procs, str_field_type, bool_print_info=False):
    ''' loadFLASH3DFieldMag
    PURPOSE: Calculate the magnitude of specified 3D vector field data stored in FLASH HDF5 data file.
    '''
    plasma_beta = 1e-10 # TODO: change to variable
    flash_file = h5py.File(filepath_data, 'r') # open hdf5 file stream: [iProc*jProc*kProc, nzb, nyb, nxb]
    names = [s for s in list(flash_file.keys()) if s.startswith(str_field_type)] # save all keys that contain the string str_field_type
    data  = sum(np.array(flash_file[i])**2 for i in names) # determine the field's magnitude
    if str_field_type.__contains__('mag'): data /= plasma_beta # normalise the magnitude
    if bool_print_info: 
        print('--------- All the keys stored in the FLASH file:\n\t' + '\n\t'.join(list(flash_file.keys()))) # print keys
        print('--------- All the keys that were used: ' + str(names))
    flash_file.close() # close the file stream
    ## reformat data
    data_sorted = reformatFLASHField(data, num_blocks, num_procs)
    return data_sorted


def loadFLASH2DField(filepath_data, num_blocks, num_procs, str_field_type, bool_print_info=False):
    ''' loadFLASH2DField
    PURPOSE: Load vector field data stored in FLASH HDF5 data file.
    '''
    plasma_beta = 1e-10 # TODO: change to variable
    flash_file = h5py.File(filepath_data, 'r') # open hdf5 file stream: [iProc*jProc*kProc, nzb, nyb, nxb]
    names = [s for s in list(flash_file.keys()) if s.startswith(str_field_type)] # save all keys that contain the string str_field_type
    data_1, data_2, data_3 = np.array([flash_file[i] for i in names])
    if str_field_type.__contains__('mag'): # normalise the magnitude
        data_1 /= plasma_beta
        data_2 /= plasma_beta
        data_3 /= plasma_beta
    if bool_print_info: 
        print('--------- All the keys stored in the FLASH file:\n\t' + '\n\t'.join(list(flash_file.keys()))) # print keys
        print('--------- All the keys that were used: ' + str(names))
    flash_file.close() # close the file stream
    ## reformat data
    data_sorted_1 = reformatFLASHField(data_1, num_blocks, num_procs)
    data_sorted_2 = reformatFLASHField(data_2, num_blocks, num_procs)
    data_sorted_3 = reformatFLASHField(data_3, num_blocks, num_procs)
    return data_sorted_1, data_sorted_2, data_sorted_3


def loadTurbData(filepath_data, var_y, t_eddy):
    """ loadTurbData
    PURPOSE: Load data (x [time], y [selected data]) from the Turb.dat data located in filepath_data. 
    """
    ## load data
    filepath_turb = createFilePath([filepath_data, "Turb.dat"])
    first_line = open(filepath_turb).readline().split()
    len_thresh = len(first_line)
    ## save x and y data
    data_x = []
    data_y = []
    prev_time = -1
    var_x = 0 # time 
    with open(filepath_turb) as file_lines:
        for line in file_lines:
            data_split = line.split()
            if len(data_split)  == len_thresh:
                if (not(data_split[var_x][0] == "#") and not(data_split[var_y][0] == "#")):
                    cur_time = float(data_split[var_x]) / t_eddy
                    ## if the simulation has been restarted, make sure that only the progressed data is used
                    if cur_time > prev_time:
                        data_x.append(cur_time) # normalise time-domain
                        data_y.append(float(data_split[var_y]))
                        prev_time = cur_time
    ## return variables
    return data_x, data_y


def loadSpectraData(filepath_data):
    data_file = open(filepath_data).readlines() # load in data
    data      = np.array([x.strip().split() for x in data_file[6:]]) # store all data. index: data[row, col]
    data_x    = np.array(list(map(float, data[:, 1])))  # variable: wave number (k)
    data_y    = np.array(list(map(float, data[:, 15]))) # variable: power spectrum
    return data_x, data_y


def saveSpectraObject(obj, filepath_file):
    ## if the file exists, then delete it
    if os.path.isfile(filepath_file): os.remove(filepath_file)
    ## save new object
    with open(filepath_file, "wb") as output: pickle.dump(obj, output, -1)


def analyseVelSpectra(filepath_data, plots_per_eddy, bool_hide_progress):
    ## initialise list of kinetic scales for the simulation
    vel_sim_time  = [] # simulation time points
    vel_k     = [] # spectra k values
    vel_power = [] # power spectra
    fit_vel_k = [] # fitted spectra k values
    fit_vel_power   = [] # fitted power spectra
    fit_vel_success = [] # spectra files that were fitted
    fit_vel_failed  = [] # spectra files that couldn't be fitted
    k_nu      = [] # measured k_nu from fitted spectra
    k_nu_p    = [] # measured k_nu_p from fitted spectra
    ## filter for kinetic spectra files
    vel_filenames = getFilesFromFolder(filepath_data, str_contains="Turb_hdf5_plt_cnt_", str_endswith="spect_vels.dat", file_index_placing=-3, file_start_index=2)
    ## loop over spectra files in domain
    print("\t\t> There are " + str(len(vel_filenames)) + " velocity spectra files")
    for vel_filename, sub_index in zip(vel_filenames, tqdm(range(len(vel_filenames)), disable=bool_hide_progress)):
        ## save the simulation time
        vel_sim_time.append( float(vel_filename.split("_")[-3]) / plots_per_eddy )
        ## load data
        k, power = loadSpectraData(createFilePath([filepath_data, vel_filename]))
        vel_k.append(k)
        vel_power.append(power)
        try:
            ## fit kinetic spectra
            fit_vel = fitVelocitySpectra(k, power)
            tmp_k, tmp_power, _, k_params = fit_vel.fit_full_k_model()
            ## extract and append scales
            fit_vel_k.append(tmp_k)
            fit_vel_power.append(tmp_power)
            k_nu.append(k_params["k_nu"])
            k_nu_p.append(k_params["k_nu_p"])
            ## note that the spectra data was fitted
            fit_vel_success.append(vel_filename)
        ## note that the spectra data could not be fitted
        except: fit_vel_failed.append(vel_filename)
    ## create dictionary of important simulation variables
    args = {
        "vel_sim_time":vel_sim_time,
        "vel_k":vel_k,
        "vel_power":vel_power,
        "fit_vel_k":fit_vel_k,
        "fit_vel_power":fit_vel_power,
        "fit_vel_success":fit_vel_success,
        "fit_vel_failed":fit_vel_failed,
        "k_nu":k_nu,
        "k_nu_p":k_nu_p
    }
    return args


def analyseMagSpectra(filepath_data, plots_per_eddy, bool_hide_progress):
    ## initialise list of magnetic scales for the simulation
    mag_sim_time  = [] # simulation time points
    mag_k     = [] # spectra k values
    mag_power = [] # power spectra
    fit_mag_k = [] # fitted spectra k values
    fit_mag_power   = [] # fitted power spectra
    fit_mag_success = [] # spectra files that were fitted
    fit_mag_failed  = [] # spectra files that couldn't be fitted
    k_max     = [] # measured k_max from fitted spectra
    k_eta     = [] # measured k_eta from fitted spectra
    k_eta_p   = [] # measured k_eta_p from fitted spectra
    ## filter for magnetic spectra files
    mag_filenames = getFilesFromFolder(filepath_data, str_contains="Turb_hdf5_plt_cnt_", str_endswith="spect_mags.dat", file_index_placing=-3, file_start_index=2)
    ## loop over spectra files in domain
    print("\t\t> There are " + str(len(mag_filenames)) + " magnetic spectra files")
    for mag_filename, sub_index in zip(mag_filenames, tqdm(range(len(mag_filenames)), disable=bool_hide_progress)):
        ## save the simulation time
        mag_sim_time.append( float(mag_filename.split("_")[-3]) / plots_per_eddy )
        ## load data
        k, power = loadSpectraData(createFilePath([filepath_data, mag_filename]))
        mag_k.append(k)
        mag_power.append(power)
        try:
            ## fit magnetic spectra
            fit_mag = fitMagneticSpectra(k, power)
            tmp_k, tmp_power, _, k_params = fit_mag.fit_full_k_model()
            ## extract and append scales
            fit_mag_k.append(tmp_k)
            fit_mag_power.append(tmp_power)
            k_max.append(k_params["k_max"])
            k_eta.append(k_params["k_eta"])
            k_eta_p.append(k_params["k_eta_p"])
            ## note that the spectra data was fitted
            fit_mag_success.append(mag_filename)
        ## indicate that spectra could not be fitted
        except: fit_mag_failed.append(mag_filename)
    ## create dictionary of important simulation variables
    args = {
        "mag_sim_time":mag_sim_time,
        "mag_k":mag_k,
        "mag_power":mag_power,
        "fit_mag_k":fit_mag_k,
        "fit_mag_power":fit_mag_power,
        "fit_mag_success":fit_mag_success,
        "fit_mag_failed":fit_mag_failed,
        "k_max":k_max,
        "k_eta":k_eta,
        "k_eta_p":k_eta_p
    }
    return args


class spectra():
    def __init__(self, sim_label,
                ## variables from velocity spectra
                vel_sim_time, vel_k, vel_power, fit_vel_k, fit_vel_power, fit_vel_success, fit_vel_failed, k_nu, k_nu_p,
                ## variables from magnetic spectra
                mag_sim_time, mag_k, mag_power, fit_mag_k, fit_mag_power, fit_mag_success, fit_mag_failed, k_max, k_eta, k_eta_p):
        ## simulation information
        self.sim_label    = sim_label
        self.vel_sim_time = vel_sim_time
        self.mag_sim_time = mag_sim_time
        ## simulation data
        self.vel_k     = vel_k
        self.vel_power = vel_power
        self.mag_k     = mag_k
        self.mag_power = mag_power
        ## fitted spectras
        self.fit_vel_k     = fit_vel_k
        self.fit_vel_power = fit_vel_power
        self.fit_mag_k     = fit_mag_k
        self.fit_mag_power = fit_mag_power
        ## list of spectra files that could/could not be fitted
        self.fit_vel_success = fit_vel_success
        self.fit_vel_failed  = fit_vel_failed
        self.fit_mag_success = fit_mag_success
        self.fit_mag_failed  = fit_mag_failed
        ## important measured scales
        self.k_nu    = k_nu
        self.k_nu_p  = k_nu_p
        self.k_max   = k_max
        self.k_eta   = k_eta
        self.k_eta_p = k_eta_p


class magModels():
    def full_k_model(x, a0, a1, a2, a3, a4, a5):
        low_k = a0 * x**a1 *  np.exp(-a2*x)
        high_k = a3 / ( 1 + ( x / a4 )**a5 ) * ( 1 - np.exp(-a2*x) )
        full_k = low_k + high_k
        return(full_k)
    def low_k_model(x, a0, a1, a2):
        low_k_power = a0 * x**a1 * np.exp(-a2*x)
        return(low_k_power)
    def high_log_k_model(x, a0, a1):
        """
        The model in linear space:
            high_k_power = a0*x**a1
        The model in log space:
            log(high_k_power) = log(a0) + a1*x
            b0 = log(a0)
            log(high_k_power) = b0 + a1*x
        """
        high_k_power = a0 + a1*x
        return(high_k_power)
    def low_log_k_model(x, a0, a1, a2):
        """
        The model in linear space:
            low_k_power = a0 * x**a1 * np.exp(-a2*x)
        The model in log space:
            log(high_k_power) = log(a0) -a1*x
            b0 = log(a0)
            log(high_k_power) = b0 - a1*x
        """
        low_k_power = a0 * x**a1 * np.exp(-a2*x)
        return(low_k_power)
    def high_k_model(x, a0, a1):
        high_k_power = a0*x**a1
        return(high_k_power)
    def root_model(x, a0, a1, a2, a3, a4):
        """
        Easiest to do in log space.
        """
        b0 = np.log(a0)
        b1 = a1
        b2 = a2
        b3 = np.log(a3)
        b4 = a4
        root = ( b0 + b1*np.log(x) - b2*x ) - ( b3 + b4*np.log(x) )
        return(root)


class velModels():
    def full_k_model(x, a0, a1, a2, a3, a4):
        low_k = a0 * np.exp(-a1*x)
        high_k = 2*a2 / ( 1 + ( x / a3 )**a4 ) * ( 1 - np.exp(-a1 * x) )
        full_k = low_k + high_k
        return(full_k)
    def low_k_model(x, a0, a1):
        low_k_power = a0 * np.exp(-a1*x)
        return(low_k_power)
    def high_k_model(x, a0, a1):
        high_k_power = a0 * x**a1
        return(high_k_power)
    def low_log_k_model(x, a0, a1):
        """
        The model in linear space:
            high_k_power = a0 * np.exp(-a1*x)
        The model in log space:
            log(high_k_power) = log(a0) -a1*x
            b0 = log(a0)
            log(high_k_power) = b0 - a1*x
        """
        low_k_power = a0 - a1 * x
        return(low_k_power)
    def high_log_k_model(x, a0, a1):
        """
        The model in linear space:
            high_k_power = a0*x**a1
        The model in log space:
            log(high_k_power) = log(a0) + a1*log(x)
            b0 = log(a0)
            log(high_k_power) = b0 + a1*log(x)
        """
        high_k_power = a0 + a1 * np.log(x)
        return(high_k_power)
    def root_model(x, a0, a1, a2, a3):
        """
        Easiest to do in log space.
        """
        b0 = np.log(a0)
        b1 = a1
        b2 = np.log(a2)
        b3 = a3
        root = ( b0 - b1 * x ) - ( b2 + b3 * np.log(x) )
        return(root)


class fitMagneticSpectra():
    def __init__(self,k,power):
        self.k = k
        self.power = power
        self.min_k = []
        self.min_power = []
        self.p0 = []
        self.low_k_params = []
        self.high_k_params = []
        if len(k) < 100:
            ## for low resolution simulations
            self.index_low2high_k = 30
            self.index_high_k = 50
        else:
            ## for higher resolution simulations
            self.index_low2high_k = 40
            self.index_high_k = 100
    def fit_low_k_model(self):
        low_k = self.k[:self.index_low2high_k]
        low_power = self.power[:self.index_low2high_k]
        params, _ = curve_fit(magModels.low_k_model, xdata=low_k, ydata=low_power)
        return(params)
    def fit_high_k_model(self):
        high_k = self.k[self.index_high_k:]
        high_power = self.power[self.index_high_k:]
        params, _ = curve_fit(magModels.high_k_model, xdata=high_k, ydata=high_power)
        return(params)
    def find_p0(self,method):
        if method == "first_guess":
            self.low_k_params = self.fit_low_k_model()
            self.high_k_params = self.fit_high_k_model()
            self.p0 = np.concatenate([self.low_k_params,self.high_k_params])
        elif method == "last_guess":
            transition_parms = np.array([self.min_power,self.min_k,-self.high_k_params[1]])
            self.p0 = np.concatenate([self.low_k_params,transition_parms])
    def find_root(self):
        a0, a1, a2, a3, a4 = self.p0
        root = root_scalar(magModels.root_model,args=(a0, a1, a2, a3, a4),bracket=[10,len(self.k)])
        self.min_k = root.root
        self.min_power = magModels.low_k_model(self.min_k, *self.low_k_params)
    def fit_full_k_model(self):
        fit_k = np.linspace(np.min(self.k), np.max(self.k), 10**4)
        self.find_p0("first_guess")
        high_k_parms = self.p0[-2:]
        self.find_root()
        self.find_p0("last_guess")
        fit_power = magModels.full_k_model(fit_k, *self.p0)
        k_params = {"A":self.p0[0],
                    "Kazantsev":self.p0[1],
                    "k_eta_p":1/self.p0[2],
                    "k_max":self.p0[1]/self.p0[2],
                    "k_eta":self.min_k,
                    "a_0_k_eta":high_k_parms[0],
                    "power_k_eta":self.min_power,
                    "diss_power_law":self.p0[-1]}
        return(fit_k,fit_power,self.p0,k_params)


class fitVelocitySpectra():
    def __init__(self,k,power):
        self.k = k
        self.power = power
        self.min_k = []
        self.min_power = []
        self.p0 = []
        self.low_k_params = []
        self.high_k_params = []
        self.index_k_exp_end = 5
        self.index_low2high_k = 20
    def fit_low_k_model(self):
        low_k = self.k[:self.index_low2high_k]
        low_power = self.power[:self.index_low2high_k]
        params, _ = curve_fit(velModels.low_k_model, xdata=low_k, ydata=low_power)
        return(params)
    def fit_high_k_model(self):
        high_k = self.k[self.index_low2high_k:]
        high_power = self.power[self.index_low2high_k:]
        params, _ = curve_fit(velModels.high_k_model, xdata=high_k, ydata=high_power)
        return(params)
    def fit_log_low_k_model(self):
        low_k = self.k[:self.index_k_exp_end]
        low_power = self.power[:self.index_k_exp_end]
        np.seterr(divide = "ignore")  # ignore devide by zero error when model can"t be fit
        params, _ = curve_fit(velModels.low_log_k_model, xdata=low_k, ydata=np.log(low_power))
        np.seterr(divide = "warn")
        a0, a1 = params
        # undo model transformation
        a0 = np.exp(a0)
        params = np.array([a0,a1])
        return(params)
    def fit_log_high_k_model(self):
        high_k = self.k[self.index_low2high_k:]
        high_power = self.power[self.index_low2high_k:]
        params, _ = curve_fit(velModels.high_log_k_model, xdata=high_k, ydata=np.log(high_power))
        a0, a1 = params
        # undo model transformation
        a0 = np.exp(a0)
        params = np.array([a0,a1])
        return(params)
    def find_p0(self,method):
        if method == "first_guess":
            self.low_k_params = self.fit_log_low_k_model()
            self.high_k_params = self.fit_log_high_k_model()
            self.p0 = np.concatenate([self.low_k_params,self.high_k_params])
        elif method == "last_guess":
            self.p0 = np.concatenate([self.low_k_params,self.high_k_params])
    def find_root(self):
        a0, a1, a2, a3 = self.p0
        root = root_scalar(velModels.root_model,args=(a0, a1, a2, a3),bracket=[2,len(self.k)/2])
        self.min_k = root.root
        self.min_power = velModels.low_k_model(self.min_k, *self.low_k_params)
    def fit_full_k_model(self):
        fit_k = np.linspace(np.min(self.k), np.max(self.k), 10**4)
        self.find_p0("first_guess")
        high_k_parms = self.p0[-2:]
        self.find_root()
        self.find_p0("last_guess")
        a0, a1, a2, a3 = self.p0
        fit_power = velModels.full_k_model(fit_k,a0,a1,self.min_power,self.min_k,-self.p0[-1])
        k_params = {"A":self.p0[0],
                    "k_nu_p":1/self.p0[1],
                    "k_nu":self.min_k,
                    "a_0_k_nu":high_k_parms[0],
                    "power_k_nu":self.min_power,
                    "diss_power_law":self.p0[-1]}
        return(fit_k,fit_power,self.p0,k_params)


## END OF LIBRARY