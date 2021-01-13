#!/usr/bin/env python3

##################################################################
## MODULES
##################################################################
import os
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.collections import LineCollection
from scipy.interpolate import make_interp_spline
from scipy.optimize import curve_fit

## user defined librariers
from the_plotting_library import *
from the_dynamo_library import *
from the_useful_library import *
from the_matplotlib_styler import *


#################################################################
## PREPARE TERMINAL/WORKSPACE/CODE
#################################################################
os.system("clear")  # clear terminal window
plt.close("all")    # close all pre-existing plots
## work in a non-interactive mode
mpl.use("Agg")
plt.ioff()


##################################################################
## FUNCTIONS
##################################################################
def fnc_exp_linearised(x, a0, a1):
    return np.log(a0) + a1 * x

def fnc_exp(x, a0, a1):
    return a0 * np.exp(a1 * x)

def sci_notation(number, sig_fig=2):
    ret_string = "{0:.{1:d}e}".format(number, sig_fig)
    a, b = ret_string.split("e")
    return a + r"$\times 10^{" + str(int(b)) + r"}$"


##################################################################
## INPUT COMMAND LINE ARGUMENTS
##################################################################
ap = argparse.ArgumentParser(description="a bunch of input arguments")
## ------------------- DEFINE OPTIONAL ARGUMENTS
ap.add_argument("-vis_folder",  type=str, default="vis_folder", required=False, help="where figures are saved")
## ------------------- DEFINE REQUIRED ARGUMENTS
ap.add_argument("-base_path",   type=str, required=True, help="filepath to the base of dat_folders")
ap.add_argument("-dat_folders", type=str, required=True, help="where Turb.dat is stored", nargs="+")
ap.add_argument("-dat_labels",  type=str, required=True, help="data labels", nargs="+")
ap.add_argument("-pre_name",    type=str, required=True, help="figure name")
ap.add_argument("-t_eddy",      type=float, required=True, help="eddy turnover time := L / (cs * Mach)")
ap.add_argument("-start_time",  type=int, required=True, help="First file to process", nargs="+")
ap.add_argument("-end_time",    type=int, required=True, help="end of time range", nargs="+")
## ---------------------------- OPEN ARGUMENTS
args = vars(ap.parse_args())
## ---------------------------- SAVE PARAMETERS
filepath_base   = args["base_path"]   # home directory
folders_data    = args["dat_folders"] # list of subfolders where data is stored
labels_data     = args["dat_labels"]  # list of labels for plots
folder_plot     = args["vis_folder"]  # subfolder where plots should be saved
pre_name        = args["pre_name"]    # name of figures
t_eddy          = args["t_eddy"]      # eddy turnover time
start_time      = args["start_time"]  # starting processing frame
end_time        = args["end_time"]    # the last file to process


##################################################################
## GET USER INPUT (choose which variables to plot)
##################################################################
## accept input for the y-axis variable
print("Which variable do you want to plot on the y-axis?")
print("\tOptions: 6 (E_kin), 8 (rms_Mach), 29 (E_mag)")
var_y = int(input("\tInput: "))
while ((var_y != 6) and (var_y != 8) and (var_y != 29)):
    print("\tInvalid input. Choose an option from: 6 (E_kin), 8 (rms_Mach), 29 (E_mag)")
    var_y = int(input("\tInput: "))
print(" ")
## initialise variables
if var_y == 6:
    ## mach number
    label_y       = r"$E_{\nu}/E_{\nu 0}$"
    bool_norm_dat = bool(1)
    var_scale     = "log"
    var_name      = "E_kin"
elif var_y == 8:
    ## mach number
    label_y       = r"$\mathcal{M}$"
    bool_norm_dat = bool(0)
    var_scale     = "linear"
    var_name      = "rms_Mach"
else:
    ## magnetic field
    label_y       = r"$E_{B}$"
    bool_norm_dat = bool(1)
    var_scale     = "log"
    var_name      = "E_mag"


##################################################################
## INITIALISING VARIABLES
##################################################################
## folders where spectra data files are stored for each simulation
filepaths_data = []
for index in range(len(folders_data)): filepaths_data.append(createFilePath([filepath_base, folders_data[index]]))
## create folder where the figures will be saved
filepath_plot = createFilePath([filepath_base, folder_plot])
createFolder(filepath_plot)
## print information to screen
print("Base filepath: ".ljust(20) + filepath_base)
for index in range(len(filepaths_data)): 
    print("Data folder {:d}: ".format(index).ljust(20) + filepaths_data[index])
print("Figure folder: ".ljust(20) + filepath_plot)
print("Figure name: ".ljust(20) + pre_name)
print(" ")


##################################################################
## LOAD + APPEND DATA
##################################################################
## initialise datasets
data_xs = []
data_ys = []
## loop over each simulation folder
print("Loading datasets...")
for index in range(len(filepaths_data)):
    print("\t> " + filepaths_data[index])
    ## loading data
    data_x, data_y = loadTurbData(filepaths_data[index], var_y, t_eddy)
    ## append data
    data_xs.append(data_x)
    data_ys.append(data_y)
print(" ")


##################################################################
## PLOT TIME EVOLUTION
##################################################################
## create figure
fig, ax = plt.subplots(constrained_layout=True)
## loop over each simulation
print("Plotting time evaluations...")
for data_x, data_y, index in zip(data_xs, data_ys, range(len(filepaths_data))):
    ## fit exponential to magnetic energy evolution in kinematic range
    tmp_label = ""
    if (var_y == 29):
        ## find indices that bound the kinematic range
        file_indexes_in_time_range = [ tmp_index for tmp_index, tmp_time in enumerate(data_x) if (tmp_time >= start_time[index]) and (tmp_time <= end_time[index]) ]
        index_E_low  = min(file_indexes_in_time_range)
        index_E_high = max(file_indexes_in_time_range)
        ## interpolate data
        interp_spline = make_interp_spline(data_x[index_E_low:index_E_high], data_y[index_E_low:index_E_high])
        interp_data_x = np.linspace(data_x[index_E_low], data_x[index_E_high], 10**2)
        interp_data_y = interp_spline(interp_data_x)
        ## fit exponential function in log-linear domain
        fit_params, _ = curve_fit(fnc_exp_linearised, interp_data_x, np.log(interp_data_y))
        ## plot fitted lines
        fit_data = np.linspace(data_x[index_E_low], data_x[index_E_high], 10**3)
        tmp_line = LineCollection([np.column_stack( (fit_data, fnc_exp(fit_data, *fit_params)) )],
            colors="k", ls=(0, (5, 2)), linewidth=1.5, zorder=10)
        ax.add_collection(tmp_line, autolim=False) # don"t scale axes to fit the line
        tmp_label = ", "+sci_notation(fit_params[0])+r" $\exp(${:.2f}t)".format(fit_params[1])
    ## plot time evolving data
    ax.plot(data_x, data_y,  color=sns.color_palette("PuBu", n_colors=len(filepaths_data))[index], 
        linewidth=2.5, label=labels_data[index]+tmp_label)
## label plot
print("Labelling plot...")
## major grid
ax.grid(which="major", linestyle="-", linewidth="0.5", color="black", alpha=0.35)
## add legend
ax.legend(frameon=True, loc="lower right", facecolor="white", framealpha=0.5, fontsize=18)
# ## label plot
ax.set_xlabel(r"$t / t_\mathrm{eddy}$", fontsize=22)
ax.set_ylabel(label_y, fontsize=22)
## scale y-axis
ax.set_yscale(var_scale)
## save image
print("Saving the figure...")
name_fig = filepath_plot + "/" + pre_name + "_" + var_name + ".pdf"
plt.savefig(name_fig)
plt.close()
print("Figure saved: " + name_fig)
print(" ")


##################################################################
## PLOT HISTOGRAM
##################################################################
if (var_y == 8):
    ## create figure
    fig, ax = plt.subplots(constrained_layout=True)
    ## loop through datasets
    print("Plotting frequnecy (density) of Mach...")
    for data_x, data_y, index in zip(data_xs, data_ys, range(len(filepaths_data))):
        if (max(data_x) > 7.5):
            ## find the indices corresponding with the start and end of kinematic/interested range
            index_start = min(range(len(data_x)), key=lambda i: abs(data_x[i]-start_time[index]))
            if (max(data_x) > 20): index_end = min(range(len(data_x)), key=lambda i: abs(data_x[i]-end_time[index]))
            else: index_end = -1
            ## plot PDF of Mach number in 
            plotPDF(ax, data_y[index_start:index_end], sim_label=labels_data[index], num_cols=len(filepaths_data), col_index=index)
    ## label plot
    print("Labelling plot...")
    ## add legend
    ax.legend(loc="upper right", facecolor="white", framealpha=1, fontsize=20)
    ## label plot
    ax.set_xlabel(r"$\mathcal{M}$", fontsize=22)
    ax.set_ylabel(r"$p(\mathcal{M})$", fontsize=22)
    ## save image
    print("Saving the figure...")
    name_fig = filepath_plot + "/" + pre_name + "_" + var_name + "_hist.pdf"
    plt.savefig(name_fig)
    plt.close()
    print("Figure saved: " + name_fig)
    print(" ")


## END OF PROGRAM