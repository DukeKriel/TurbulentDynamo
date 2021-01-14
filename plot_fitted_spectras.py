#!/usr/bin/env python3

##################################################################
## MODULES
##################################################################
import os
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns # for plotting colours

## always import the c-version of pickle
try: import cPickle as pickle
except ModuleNotFoundError: import pickle

from tqdm.auto import tqdm # progress bar
from matplotlib.gridspec import GridSpec

## user defined libraries
from the_plotting_library import *
from the_dynamo_library import *
from the_useful_library import *
from the_matplotlib_styler import *


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
ap.add_argument("-hide_progress_bar", type=str2bool, default=False, required=False, help="hide the progress bar", nargs="?", const=True)
ap.add_argument("-analyse",     type=str2bool, default=True, required=False, help="(re-)fit spectra data", nargs="?", const=True)
ap.add_argument("-plot_scales", type=str2bool, default=True, required=False, help="plot the PDFs of the fitted scales", nargs="?", const=True)
ap.add_argument("-plot_spectra",type=str2bool, default=True, required=False, help="plot the evolution of the spectra", nargs="?", const=True)
ap.add_argument("-vis_folder",  type=str, default="vis_folder", required=False, help="where figures are saved")
ap.add_argument("-sub_folders", type=str, default="spect", required=False, help="where spectras are stored in simulation folder")
ap.add_argument("-dat_labels",  type=str, default=None,     required=False, help="data labels", nargs="+")
ap.add_argument("-start_time",  type=int, default=[1],      required=False, help="First file to process", nargs="+")
ap.add_argument("-end_time",    type=int, default=[np.inf], required=False, help="end of time range", nargs="+")
## ------------------- DEFINE REQUIRED ARGUMENTS
ap.add_argument("-base_path",   type=str, required=True, help="filepath to the base of dat_folders")
ap.add_argument("-dat_folders", type=str, required=True, help="simulation folds", nargs="+")
ap.add_argument("-pre_name",    type=str, required=True, help="figure name")
ap.add_argument("-plots_per_eddy", type=float, required=True, help="number of plot files in eddy turnover time")
## ---------------------------- OPEN ARGUMENTS
args = vars(ap.parse_args())
## ---------------------------- SAVE PARAMETERS
bool_hide_progress = args["hide_progress_bar"] # should the progress bar be displayed?
bool_anlyse        = args["analyse"]      # analyse spectra and overwrite objects if necessary
bool_plot_scales   = args["plot_scales"]  # plot the PDFs of the fitted scales
bool_plot_spectra  = args["plot_spectra"] # plot the evolution of the spectra
start_time         = args["start_time"]   # starting processing frame
end_time           = args["end_time"]     # the last file to process
plots_per_eddy     = args["plots_per_eddy"] # number of plot files in eddy turnover time
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
## if a time-range isn't specified for one of the simulations, then use the default time-range
if len(start_time) < len(folders_data): start_time.extend( [1] * (len(folders_data) - len(start_time)) )
if len(end_time) < len(folders_data): end_time.extend( [np.inf] * (len(folders_data) - len(end_time)) )
## if data labels weren't given, then use the data folders' names
if labels_data is None: labels_data = folders_data
## folders where spectra data is
filepaths_data = []
for folder_data in folders_data:
    filepaths_data.append(createFilePath([filepath_base, folder_data, folders_sub]))
## folder where PDF plots will be saved
filepath_plot = createFilePath([filepath_base, folder_vis])
createFolder(filepath_plot)
## folder where spectra plots will be saved
filepath_plot_spect = createFilePath([filepath_base, folder_vis, "plotSpectra"])
createFolder(filepath_plot_spect)
## print filepath information to the console
print("Base filepath: ".ljust(20)    + filepath_base)
for index in range(len(filepaths_data)):
    print("Data folder {:d}: ".format(index).ljust(20) + filepaths_data[index])
print("Figure folder: ".ljust(20)    + filepath_plot)
print("Figure name: ".ljust(20)      + pre_name)
print("Plots per T_eddy: ".ljust(20) + str(plots_per_eddy))
print(" ")


##################################################################
## LOAD & FIT SPECTRA DATA
##################################################################
## initialise list of spectra objects
spectra_objs = []
## if user wants spectra to be fitted
if bool_anlyse:
    ## loop over each simulation dataset
    print("Fitting spectra data...")
    for filepath_data, index in zip(filepaths_data, range(len(filepaths_data))):
        print("\tFitting in: " + filepath_data)
        ## analyse kinetic and magnetic spectras
        vel_args = analyseVelSpectra(filepath_data, plots_per_eddy, bool_hide_progress)
        mag_args = analyseMagSpectra(filepath_data, plots_per_eddy, bool_hide_progress)
        ## check domain of k
        vel_k = vel_args["vel_k"]
        mag_k = mag_args["mag_k"]
        ## make sure that ther are the same number of velocity and magnetic data files to fit
        tmp_vel_times = vel_args["vel_sim_times"]
        tmp_mag_times = mag_args["mag_sim_times"]
        if not(set(tmp_vel_times) == set(tmp_mag_times)):
            print("\t> Successfully fitted {:d} velocity and {:d} magnetic spectras files.".format(len(tmp_vel_times), len(tmp_mag_times)))
            print("\t> Failed to fit at time point(s):", list(set(tmp_vel_times) - set(tmp_mag_times)))
        ## check which spectra data files failed to be fitted
        fit_vel_failed = vel_args["fit_vel_failed"]
        fit_mag_failed = mag_args["fit_mag_failed"]
        list_elem_sep = "\n\t\t\t"
        if len(fit_vel_failed) > 0: print("\t\t> Failed to fit {0:d} kinetic spectra:".format(len(fit_vel_failed)),  list_elem_sep+list_elem_sep.join(fit_vel_failed))
        if len(fit_mag_failed) > 0: print("\t\t> Failed to fit {0:d} magnetic spectra:".format(len(fit_mag_failed)), list_elem_sep+list_elem_sep.join(fit_mag_failed))
        ## create spectra object
        tmp_obj = spectra(labels_data[index], **vel_args, **mag_args)
        ## save object
        filename = createFilePath([filepath_data, "spectra_obj.pkl"])
        saveSpectraObject(tmp_obj, filename) # overwrite saved object
        ## append object
        spectra_objs.append(tmp_obj)
        print(" ")
## if data has previously been fit, and the user wants to read it in
else:
    ## loop over each simulation dataset
    print("Loading spectra objects...")
    for filepath_data, index in zip(filepaths_data, range(len(filepaths_data))):
        print("\tLoading from: " + filepath_data)
        ## create the file"s name
        filename = createFilePath([filepath_data, "spectra_obj.pkl"])
        ## if the file exists, then read it in
        if os.path.isfile(filename):
            with open(filename, "rb") as input:
                tmp_obj = pickle.load(input)
        else: raise Exception("\t> No spectra object found.")
        ## check which spectra data files failed to be fitted
        fit_vel_failed = tmp_obj.fit_vel_failed
        fit_mag_failed = tmp_obj.fit_mag_failed
        list_elem_sep = "\n\t\t"
        if len(fit_vel_failed) > 0: print("\t> Failed to fit {0:d} kinetic spectra:".format(len(fit_vel_failed)),  list_elem_sep+list_elem_sep.join(fit_vel_failed))
        if len(fit_mag_failed) > 0: print("\t> Failed to fit {0:d} magnetic spectra:".format(len(fit_mag_failed)), list_elem_sep+list_elem_sep.join(fit_mag_failed))
        ## append object
        spectra_objs.append(tmp_obj)
        print(" ")


##################################################################
## PLOT PDFs FOR FITTED SCALES
##################################################################
if bool_plot_scales:
    print("Plotting PDFs of fitted scales...")
    ## initialise figure
    fig_scales = plt.figure(figsize=(9,12), constrained_layout=True)
    fig_grids = GridSpec(4, 2, figure=fig_scales)
    ax0  = fig_scales.add_subplot(fig_grids[0, :])
    ax10 = fig_scales.add_subplot(fig_grids[1, 0])
    ax11 = fig_scales.add_subplot(fig_grids[1, 1])
    ax20 = fig_scales.add_subplot(fig_grids[2, 0])
    ax21 = fig_scales.add_subplot(fig_grids[2, 1])
    ax30 = fig_scales.add_subplot(fig_grids[3, 0])
    ax31 = fig_scales.add_subplot(fig_grids[3, 1])
    ## loop over each simulation's spectra object
    for spectra_obj, index in zip(spectra_objs, range(len(spectra_objs))):
        ## load important analysis variables
        sim_label = spectra_obj.sim_label
        sim_time = getCommonElements(spectra_obj.vel_sim_times, spectra_obj.mag_sim_times)
        ## find which part of the data to have a look at
        index_start = min(range(len(sim_time)), key=lambda i: abs(sim_time[i]-start_time[index]))
        index_end   = min(range(len(sim_time)), key=lambda i: abs(sim_time[i]-end_time[index]))
        print("\t> Plotting {0} object".format(sim_label))
        ## load important scales
        k_max   = spectra_obj.k_max[index_start:index_end]
        k_nu    = spectra_obj.k_nu[index_start:index_end]
        k_nu_p  = spectra_obj.k_nu_p[index_start:index_end]
        k_eta   = spectra_obj.k_eta[index_start:index_end]
        k_eta_p = spectra_obj.k_eta_p[index_start:index_end]
        Pm      = [(a/b)**2 for a,b in zip(k_eta,k_nu)]
        Pm_p    = [(a/b)**2 for a,b in zip(k_eta_p,k_nu_p)]
        ## plot data
        plotHistogram(ax0,  k_max,   sim_label=sim_label, num_cols=len(spectra_objs), col_index=index) # plot k_max
        plotHistogram(ax10, k_nu,    sim_label=sim_label, num_cols=len(spectra_objs), col_index=index) # plot k_nu
        plotHistogram(ax11, k_nu_p,  sim_label=sim_label, num_cols=len(spectra_objs), col_index=index) # plot k_nu_p
        plotHistogram(ax20, k_eta,   sim_label=sim_label, num_cols=len(spectra_objs), col_index=index) # plot k_eta
        plotHistogram(ax21, k_eta_p, sim_label=sim_label, num_cols=len(spectra_objs), col_index=index) # plot k_eta_p
        plotHistogram(ax30, Pm,      sim_label=r"Pm"+"{:.2f}".format(np.mean(Pm)),   num_cols=len(spectra_objs), col_index=index) # plot Pm
        plotHistogram(ax31, Pm_p,    sim_label=r"Pm"+"{:.2f}".format(np.mean(Pm_p)), num_cols=len(spectra_objs), col_index=index) # plot Pm_p
    ## label axes
    ax0.set_xlabel(r"$k_{\rm max}$",        fontsize=20)
    ax10.set_xlabel(r"$k_{\nu}$",           fontsize=20)
    ax11.set_xlabel(r"$k_{\nu}^{\prime}$",  fontsize=20)
    ax20.set_xlabel(r"$k_{\eta}$",          fontsize=20)
    ax21.set_xlabel(r"$k_{\eta}^{\prime}$", fontsize=20)
    ax30.set_xlabel(r"$\rm Pm$",            fontsize=20)
    ax31.set_xlabel(r"$\rm Pm^{\prime}$",   fontsize=20)
    ax0.set_ylabel(r"Counts",  fontsize=20)
    ax10.set_ylabel(r"Counts", fontsize=20)
    ax11.set_ylabel(r"Counts", fontsize=20)
    ax20.set_ylabel(r"Counts", fontsize=20)
    ax21.set_ylabel(r"Counts", fontsize=20)
    ax30.set_ylabel(r"Counts", fontsize=20)
    ax31.set_ylabel(r"Counts", fontsize=20)
    ## add legend
    ax0.legend(fontsize=16,  loc="upper right", title="Sim. Label")
    ax30.legend(fontsize=16, loc="upper right", title="Measured")
    ax31.legend(fontsize=16, loc="upper right", title="Measured")
    ## save image
    print("Saving the figure...")
    name_fig = filepath_plot + "/" + pre_name + "_fitted_scales.pdf"
    plt.savefig(name_fig)
    plt.close()
    print("Figure saved: " + name_fig)
    print(" ")


##################################################################
## PLOT SPECTRA EVOLUTION
##################################################################
if bool_plot_spectra:
    print("Plotting evolving spectras...")
    ## loop over each simulation dataset
    for spectra_obj, index in zip(spectra_objs, range(len(spectra_objs))):
        ## load important analysis variables
        sim_label = spectra_obj.sim_label
        sim_time = getCommonElements(spectra_obj.vel_sim_times, spectra_obj.mag_sim_times)
        fit_vel_failed = spectra_obj.fit_vel_failed
        fit_mag_failed = spectra_obj.fit_mag_failed
        print("\t> Plotting: " + sim_label + " object (" + str(len(sim_time)) + " frames):")
        ## load spectra data
        vel_k     = spectra_obj.vel_k
        vel_power = spectra_obj.vel_power
        mag_k     = spectra_obj.mag_k
        mag_power = spectra_obj.mag_power
        ## load fitted spectra
        fit_vel_k     = spectra_obj.fit_vel_k
        fit_vel_power = spectra_obj.fit_vel_power
        fit_mag_k     = spectra_obj.fit_mag_k
        fit_mag_power = spectra_obj.fit_mag_power
        ## load important scales
        k_nu    = spectra_obj.k_nu
        k_nu_p  = spectra_obj.k_nu_p
        k_max   = spectra_obj.k_max
        k_eta   = spectra_obj.k_eta
        k_eta_p = spectra_obj.k_eta_p
        ## find indices of when to start and stop looking at fitted scales
        sub_index_start = min(tmp_index for tmp_index, tmp_time in enumerate(sim_time) if (tmp_time >= start_time[index]) and (tmp_time <= end_time[index]))
        sub_index_end   = max(tmp_index for tmp_index, tmp_time in enumerate(sim_time) if (tmp_time >= start_time[index]) and (tmp_time <= end_time[index]))
        ## calculate mean and variance of measured scales
        k_nu_mean    = np.mean(k_nu[sub_index_start:sub_index_end])
        k_nu_p_mean  = np.mean(k_nu_p[sub_index_start:sub_index_end])
        k_max_mean   = np.mean(k_max[sub_index_start:sub_index_end])
        k_eta_mean   = np.mean(k_eta[sub_index_start:sub_index_end])
        k_eta_p_mean = np.mean(k_eta_p[sub_index_start:sub_index_end]) 
        k_nu_std     = np.std(k_nu[sub_index_start:sub_index_end])
        k_nu_p_std   = np.std(k_nu_p[sub_index_start:sub_index_end])
        k_max_std    = np.std(k_max[sub_index_start:sub_index_end])
        k_eta_std    = np.std(k_eta[sub_index_start:sub_index_end])
        k_eta_p_std  = np.std(k_eta_p[sub_index_start:sub_index_end])
        ## plot evolution of spectra
        y_min = 1e-17
        y_max = 10
        x_min = 10**(-1)
        x_max = max(len(vel_k[-1]), len(mag_k[-1]))
        ## initialise spectra evolution figure
        fig, ax = plt.subplots(constrained_layout=True)
        ## loop over data from each time slice
        for time_index in tqdm(range(len(sim_time)), disable=bool_hide_progress):
            ## plot spectras
            ax.plot(vel_k[time_index], vel_power[time_index], label=r"V-spectra", color="blue", ls="", marker=".", markersize=8)
            ax.plot(mag_k[time_index], mag_power[time_index], label=r"B-spectra", color="red", ls="", marker=".", markersize=8)
            ###############################
            ## plot fitted kinetic spectra
            ax.plot(fit_vel_k[time_index], fit_vel_power[time_index], label=r"fitted V-spectra", color="blue", linestyle="--", dashes=(5, 2.5), linewidth=2)
            ## plot scales
            ax.axvline(x=k_nu[time_index],    ls="--", color="blue",   label=r"$k_{\nu}$")
            ax.axvline(x=k_nu_p[time_index],  ls="--", color="green",  label=r"$k_{\nu}^\prime$")
            ## show variance in variables
            ax.fill_betweenx(np.linspace(y_min, y_max,100), (k_nu_mean   - k_nu_std),   (k_nu_mean   + k_nu_std),   facecolor="blue",  alpha=0.3, zorder=1)
            ax.fill_betweenx(np.linspace(y_min, y_max,100), (k_nu_p_mean - k_nu_p_std), (k_nu_p_mean + k_nu_p_std), facecolor="green", alpha=0.3, zorder=1)
            ## add model labels
            ax.text(0.025, 0.025,
                    r"$\mathcal{P}_{\rm vel}(k) = A\exp\left\{-\frac{k}{k_\nu^\prime}\right\} + $" +
                    r"$\frac{\mathcal{P}_{\rm vel}(k_{\nu})}{1 + \left( \frac{k}{k_{\nu}} \right)^{\alpha}} \left( 1 - \exp\left\{-\frac{k}{k_\nu^\prime}\right\} \right)$",
                    va="bottom", ha="left", transform=ax.transAxes, fontsize=12)
            ###############################
            ## plot fitted magnetic spectra
            ax.plot(fit_mag_k[time_index], fit_mag_power[time_index], label=r"fitted B-spectra", color="red", linestyle="--", dashes=(5, 2.5), linewidth=2)
            ## plot scales
            ax.axvline(x=k_max[time_index],   ls="--", color="black",  label=r"$k_{\rm max}$")
            ax.axvline(x=k_eta[time_index],   ls="--", color="red",    label=r"$k_{\eta}$")
            ax.axvline(x=k_eta_p[time_index], ls="--", color="orange", label=r"$k_{\eta}^\prime$")
            ## show variance in variables
            ax.fill_betweenx(np.linspace(y_min, y_max,100), (k_max_mean  - k_max_std),  (k_max_mean  + k_max_std),  facecolor="black", alpha=0.3, zorder=1)
            ax.fill_betweenx(np.linspace(y_min, y_max,100), (k_eta_mean  - k_eta_std),  (k_eta_mean  + k_eta_std),  facecolor="red",   alpha=0.3, zorder=1)
            ax.fill_betweenx(np.linspace(y_min, y_max,100), (k_eta_p_mean- k_eta_p_std),(k_eta_p_mean+ k_eta_p_std),facecolor="orange",alpha=0.3, zorder=1)
            ## add model labels
            ax.text(0.025, 0.125,
                    r"$\mathcal{P}_{\rm mag}(k) = A k^{\alpha_1}\exp\left\{-\frac{k}{k_\eta^\prime}\right\} + $" +
                    r"$\frac{\mathcal{P}_{\rm mag}(k_{\eta})}{1 + \left( \frac{k}{k_{\eta}} \right)^{\alpha_2}} \left( 1 - \exp\left\{-\frac{k}{k_\eta^\prime}\right\} \right)$",
                    va="bottom", ha="left", transform=ax.transAxes, fontsize=12)
            ###############################
            ## indicate when spectra fit was used in analysis
            if (sub_index_start <= time_index) and (time_index <= sub_index_end):
                ax.text(0.975, 0.875, r"Fitted", va="top", ha="right", transform=ax.transAxes, fontsize=16)
            ## add time stamp
            ax.text(0.975, 0.975, r"$t/t_{\rm eddy} = $ "+"{:.2f}".format(sim_time[time_index]), va="top", ha="right", transform=ax.transAxes, fontsize=16)
            ## add legend
            ax.legend(frameon=True, loc="upper left", facecolor="white", framealpha=0.5, fontsize=12)
            ## adjust figure axes
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ## label axes
            ax.set_xlabel(r"$k$")
            ax.set_ylabel(r"$\mathcal{P}$")
            ## save image
            temp_name = createFilePath([filepath_plot_spect, (pre_name + "_" + sim_label + "_spectra={0:04}".format(int(time_index)) + ".png")])
            plt.savefig(temp_name)
            ## clear axis
            ax.clear()
        plt.close()
        ## animate splectra plots
        filepath_input  = createFilePath([filepath_plot_spect, (pre_name + "_" + sim_label + "_spectra=%*.png")])
        filepath_output = createFilePath([filepath_plot, (pre_name + "_" + sim_label + "_ani_spectra.mp4")])
        ffmpeg_input    = ("ffmpeg -start_number 0 -i "                           + filepath_input + 
                        " -vb 40M -framerate 40 -vf scale=1440:-1 -vcodec mpeg4 " + filepath_output)
        print("Animating plots...")
        os.system(ffmpeg_input) 
        print("Animation finished: " + filepath_output)
        print(" ")


## END OF PROGRAM