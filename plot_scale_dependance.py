#!/usr/bin/env python3

##################################################################
## MODULES
##################################################################
import os
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns # for colors: https://medium.com/@morganjonesartist/color-guide-to-seaborn-palettes-da849406d44f

## always import the c-version of pickle
try: import cPickle as pickle
except ModuleNotFoundError: import pickle

from scipy.optimize import curve_fit, root_scalar

## user defined libraries
from the_dynamo_library import *
from the_useful_library import *
from the_matplotlib_styler import *


##################################################################
## FUNCTIONS
##################################################################
def fnc_linear(x, a0, a1):
    return a0 * x + a1

def plotDependance(ax, labels_data, list_x, list_y, str_x, str_y, str_color, bool_fit_linear=False):
    ## loop through simulation datasets
    points_x = []
    points_y = []
    print("\t Plotting data...")
    for index in range(len(labels_data)):
        print("\t\t> Plotting: " + labels_data[index])
        ## plot k peak: mean +/- std
        ax.errorbar(np.mean(list_x[index]), np.mean(list_y[index]),
            xerr=np.percentile(list_x[index], 16), yerr=np.percentile(list_y[index], 84),
            fmt="o", markersize=12, elinewidth=1.5, linestyle="None", color=sns.color_palette(str_color, n_colors=len(filepaths_data))[index], 
            label=labels_data[index], zorder=3)
        points_x.append(np.mean(list_x[index]))
        points_y.append(np.mean(list_y[index]))
    if bool_fit_linear:
        points_x = np.array(points_x)
        points_y = np.array(points_y)
        fit_params, _ = curve_fit(fnc_linear, points_x, points_y)
        ax.plot(points_x, fnc_linear(points_x, *fit_params), '--k', linewidth=2, zorder=5)
        a0, a1 = fit_params
        if a1 > 0: str_sep = r"$+$"
        else: str_sep = r"$-$"
        ax.text(0.05, 0.95, r"{0:.2f}$x$ {1} {2:.2f}".format(a0, str_sep, abs(a1)), fontsize=22, transform=ax.transAxes, ha="left", va="top")
    ## label plot
    print("\t Labelling plot...")
    ax.set_xlabel(r"$\rm %s$"%(str_x), fontsize=22)
    ax.set_ylabel(r"$k_{\rm %s}(\rm %s)$"%(str_y, str_x), fontsize=22, color="black")
    ## add legend
    ax.legend(frameon=True, loc="upper right", facecolor="white", framealpha=0.5, fontsize=18)


##################################################################
## PREPARE TERMINAL/WORKSPACE/CODE
#################################################################
os.system("clear")  # clear terminal window
plt.close("all")    # close all pre-existing plots
## work in a non-interactive mode
mpl.use("Agg")
plt.ioff()


##################################################################
## INPUT COMMAND LINE ARGUMENTS
##################################################################
ap = argparse.ArgumentParser(description="A bunch of input arguments")
## ------------------- DEFINE OPTIONAL ARGUMENTS
ap.add_argument("-fit_linear",  type=str2bool, default=False, required=False, help="fit linear line to data", nargs="?", const=True)
ap.add_argument("-vis_folder",  type=str, default="vis_folder", required=False, help="where figures are saved")
ap.add_argument("-sub_folders", type=str, default="", required=False, help="where spectras are stored in simulation folder")
## ------------------- DEFINE REQUIRED ARGUMENTS
ap.add_argument("-base_path",   type=str, required=True, help="filepath to the base of dat_folders")
ap.add_argument("-dat_folders", type=str, required=True, help="where Turb.dat is stored", nargs="+")
ap.add_argument("-dat_labels",  type=str, required=True, help="data labels", nargs="+")
ap.add_argument("-pre_name",    type=str, required=True, help="figure name")
ap.add_argument("-start_time",  type=int, required=True, help="First file to process", nargs="+")
ap.add_argument("-end_time",    type=int, required=True, help="end of time range", nargs="+")
## ---------------------------- OPEN ARGUMENTS
args = vars(ap.parse_args())
## ---------------------------- SAVE PARAMETERS
bool_fit_linear = args["fit_linear"] # fit linear line to data
start_time      = args["start_time"] # starting processing frame
end_time        = args["end_time"]   # the last file to process
## ---------------------------- SAVE FILEPATH PARAMETERS
filepath_base = args["base_path"]   # home directory
folders_data  = args["dat_folders"] # list of subfolders where each simulation"s data is stored
folders_sub   = args["sub_folders"] # where spectras are stored in simulation folder
labels_data   = args["dat_labels"]  # list of labels for plots
folder_vis    = args["vis_folder"]  # subfolder where animation and plots will be saved
pre_name      = args["pre_name"]    # name of figures


##################################################################
## INITIALISING VARIABLES
##################################################################
## folder where dependance plots will be saved
filepath_plot = createFilePath([filepath_base, folder_vis])
createFolder(filepath_plot)
## folders where spectra data is
filepaths_data = []
for folder_data in folders_data:
    filepaths_data.append(createFilePath([filepath_base, folder_data, folders_sub]))
## print filepath information to the console
print("Base filepath: ".ljust(20) + filepath_base)
for index in range(len(filepaths_data)): 
    print("Data folder {:d}: ".format(index).ljust(20)  + filepaths_data[index])
print("Figure folder: ".ljust(20) + filepath_plot)
print("Figure name: ".ljust(20)   + pre_name)
print(" ")


##################################################################
## LOAD & APPEND DATA
##################################################################
## initialise list of spectra objects
spectra_objs = []
print("Loading spectra objects...")
for filepath_data, index in zip(filepaths_data, range(len(filepaths_data))):
    print("\t> Loading from: " + filepath_data)
    ## create the file"s name
    filename = createFilePath([filepath_data, "spectra_obj.pkl"])
    ## if the file exists, then read it in
    if os.path.isfile(filename):
        with open(filename, "rb") as input:
            tmp_obj = pickle.load(input)
    else: raise Exception("\t> No spectra object found.")
    ## append object
    spectra_objs.append(tmp_obj)
print(" ")


##################################################################
## EXTRACT SCALES FROM SPECTRA OBJECTS
##################################################################
list_k_nu    = []
list_k_nu_p  = []
list_k_max   = []
list_k_eta   = []
list_k_eta_p = []
list_k_cor   = []
for spectra_obj in spectra_objs:
    ## load simulation information
    sim_label = spectra_obj.sim_label
    sim_time  = spectra_obj.vel_sim_time
    ## find indices that bound the kinematic range
    sub_index_start = min(tmp_index for tmp_index, tmp_time in enumerate(sim_time) if (tmp_time >= start_time[index]) and (tmp_time <= end_time[index]))
    sub_index_end   = max(tmp_index for tmp_index, tmp_time in enumerate(sim_time) if (tmp_time >= start_time[index]) and (tmp_time <= end_time[index]))
    ## load important scales
    k_nu    = spectra_obj.k_nu[sub_index_start:sub_index_end]
    k_nu_p  = spectra_obj.k_nu_p[sub_index_start:sub_index_end]
    k_max   = spectra_obj.k_max[sub_index_start:sub_index_end]
    k_eta   = spectra_obj.k_eta[sub_index_start:sub_index_end]
    k_eta_p = spectra_obj.k_eta_p[sub_index_start:sub_index_end]
    ## load spectra data during kinematic regime
    mag_k     = spectra_obj.mag_k[sub_index_start:sub_index_end]
    mag_power = spectra_obj.mag_power[sub_index_start:sub_index_end]
    ## calculate correlation scale during kinematic regime
    k_cor = []
    for k, power in zip(mag_k, mag_power):
        k_cor.append(sum([x*y for x,y in zip(k,power)]) / sum(power))
    ## append data
    list_k_nu.append(k_nu)
    list_k_nu_p.append(k_nu_p)
    list_k_max.append(k_max)
    list_k_eta.append(k_eta)
    list_k_eta_p.append(k_eta_p)
    list_k_cor.append(k_cor)


# ##################################################################
# ## PLOT SCALE DEPENDANCE
# ##################################################################
## create figure
fig, axs = plt.subplots(2, 2, figsize=(12,8), constrained_layout=True)
## plot histograms
plotDependance(axs[0,0], labels_data, list_k_eta,   list_k_max, "k_\eta", "max", "Blues", bool_fit_linear)
plotDependance(axs[1,0], labels_data, list_k_eta,   list_k_cor, "k_\eta", "cor", "Blues", bool_fit_linear)
plotDependance(axs[0,1], labels_data, list_k_eta_p, list_k_max, "k_\eta^\prime", "max", "Blues", bool_fit_linear)
plotDependance(axs[1,1], labels_data, list_k_eta_p, list_k_cor, "k_\eta^\prime", "cor", "Blues", bool_fit_linear)
## save image
print("Saving figure...")
fig_name = createFilePath([filepath_plot, pre_name]) + "_k_max_depend_k_eta.pdf"
plt.savefig(fig_name)
plt.close()
print("Figure saved: " + fig_name)
print(" ")

## create figure
fig, axs = plt.subplots(2, 2, figsize=(12,8), constrained_layout=True)
## plot histograms
plotDependance(axs[0,0], labels_data, list_k_nu,   list_k_max, "k_\nu", "max", "Oranges", bool_fit_linear)
plotDependance(axs[1,0], labels_data, list_k_nu,   list_k_cor, "k_\nu", "cor", "Oranges", bool_fit_linear)
plotDependance(axs[0,1], labels_data, list_k_nu_p, list_k_max, "k_\nu^{\prime}", "max", "Oranges", bool_fit_linear)
plotDependance(axs[1,1], labels_data, list_k_nu_p, list_k_cor, "k_\nu^{\prime}", "cor", "Oranges", bool_fit_linear)
## save image
print("Saving figure...")
fig_name = createFilePath([filepath_plot, pre_name]) + "_k_max_depend_nu.pdf"
plt.savefig(fig_name)
plt.close()
print("Figure saved: " + fig_name)
print(" ")


## END OF PROGRAM