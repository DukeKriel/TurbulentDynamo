##################################################################
## MODULES
##################################################################
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from IPython import get_ipython

##################################################################
## PREPARE TERMINAL/WORKSPACE/CODE
#################################################################
os.system('clear')                  # clear terminal window
get_ipython().magic('reset -sf')    # clear workspace
plt.close('all')                    # close all pre-existing plots
mpl.style.use('classic')            # plot in classic style

##################################################################
## USER VARIABLES
##################################################################
t_eddy           = 10 # number of spectra files per eddy turnover
## specify where files are located and needs to be saved
folder_main      = os.path.dirname(os.path.realpath(__file__)) # locate the directory where this file is stored
## first plot's information
folder_name_1    = 'dyna288_Bk10'
label_1_kin      = r'$\mathcal{P}_{k_{B}=10, \mathregular{kin}}$'
label_1_mag      = r'$\mathcal{P}_{k_{B}=10, \mathregular{mag}}$'
## second plot's information
folder_name_2    = 'dyna288_Bk100'
label_2_kin      = r'$\mathcal{P}_{k_{B}=100, \mathregular{kin}}$'
label_2_mag      = r'$\mathcal{P}_{k_{B}=100, \mathregular{mag}}$'
folder_data      = 'spectFiles' # subfolder where visualisation is saved
bool_disp_header = bool(1)
## specify which variables you want to plot
var_iter         = 1 # time point (simulation time)
var_time         = var_iter/t_eddy # time point (t_eddy: normalised by eddy-turnover time)
## specify which variables you want to plot
global var_x, var_y
var_x            = 1  # variable: wave number (k)
var_y            = 15 # variable: power spectrum
## set the figure's axis limits
xlim_min         = 1.0
xlim_max         = 1.3e+02
ylim_min         = 1.0e-25
ylim_max         = 4.2e-03
## should the plot be saved?
bool_save_fig    = bool(0)

##################################################################
## FUNCTIONS
##################################################################
def createFilePath(names):
    return ('/'.join([x for x in names if x != '']) + '/')

def meetsCondition(element):
    return bool(element.endswith('mags.dat') or element.endswith('vels.dat'))

def loadData(directory):
    global bool_disp_header
    global var_x, var_y
    filedata     = open(directory).readlines() # load in data
    header       = filedata[5].split() # save the header
    data         = np.array([x.strip().split() for x in filedata[6:]]) # store all data. index: data[row, col]
    if bool_disp_header:
        print('\nHeader names: for ' + directory.split('/')[-1])
        print('\n'.join(header)) # print all header names (with index)
    data_x = list(map(float, data[:, var_x]))
    data_y = list(map(float, data[:, var_y]))
    return data_x, data_y

##################################################################
## INITIALISING VARIABLES
##################################################################
name_file_kin = 'Turb_hdf5_plt_cnt_' + '{0:04}'.format(var_iter) + '_spect_vels.dat' # kinetic file
name_file_mag = 'Turb_hdf5_plt_cnt_' + '{0:04}'.format(var_iter) + '_spect_mags.dat' # magnetic file
fig           = plt.figure(figsize=(10, 7), dpi=100)
ax            = fig.add_subplot()
filepath_1    = createFilePath([folder_main, folder_name_1, folder_data])
filepath_2    = createFilePath([folder_main, folder_name_2, folder_data])

##################################################################
## LOAD DATA
##################################################################
## load dataset 1
data_x_1_kin, data_y_1_kin = loadData(filepath_1 + '/' + name_file_kin) # kinetic power spectrum
data_x_1_mag, data_y_1_mag = loadData(filepath_1 + '/' + name_file_mag) # magnetic power spectrum
## load dataset 2
data_x_2_kin, data_y_2_kin = loadData(filepath_2 + '/' + name_file_kin) # kinetic power spectrum
data_x_2_mag, data_y_2_mag = loadData(filepath_2 + '/' + name_file_mag) # magnetic power spectrum

##################################################################
## PLOT DATA
##################################################################
## plot dataset 1
line_1_kin, = plt.plot(data_x_1_kin, data_y_1_kin, 'k', label=label_1_kin) # kinetic power spectrum
line_1_mag, = plt.plot(data_x_1_mag, data_y_1_mag, 'k--', label=label_1_mag) # magnetic power spectrum
## plot dataset 2
line_1_kin, = plt.plot(data_x_2_kin, data_y_2_kin, 'b', label=label_2_kin) # kinetic power spectrum
line_1_mag, = plt.plot(data_x_2_mag, data_y_2_mag, 'b--', label=label_2_mag) # magnetic power spectrum

##################################################################
## LABEL and ADJUST PLOT
##################################################################
## scale axies
ax.set_xscale('log')
ax.set_yscale('log')
## set axis limits
plt.xlim(xlim_min, xlim_max)
plt.ylim(ylim_min, ylim_max)
## annote time (eddy tunrover-time)
ax.text(0.5, 0.95,
    r"$t/t_{\mathregular{eddy}} = %0.1f$"%0, 
    fontsize=20, color='black', 
    ha="center", va='top', transform=ax.transAxes)
# label plots
plt.xlabel(r'$k$',           fontsize=20)
plt.ylabel(r'$\mathcal{P}$', fontsize=20)
# add legend
ax.legend(loc='upper right', fontsize=17, frameon=False)
## major grid
ax.grid(which='major', linestyle='-', linewidth='0.5', color='black', alpha=0.35)
## minor grid
ax.grid(which='minor', linestyle='--', linewidth='0.5', color='black', alpha=0.2)

##################################################################
## SAVE IMAGE
##################################################################
## save figure
if bool_save_fig:
        name_fig = folder_main + '/spectra_compare=%0.1f'%var_time
        name_fig = name_fig.replace(".", "p") + '.png'
        plt.savefig(name_fig)
        print('\nFigure saved: ' + name_fig)
## display the plot
plt.show()

## END OF PROGRAM