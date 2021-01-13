#!/usr/bin/env python3

##################################################################
## MODULES
##################################################################
import os
import argparse
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from tqdm import tqdm

## user defined libraries
from the_matplotlib_styler import *
from the_useful_library import *
from the_dynamo_library import *


##################################################################
## FUNCTIONS
##################################################################
def plotPDF(ax, vals, num_bins=10, sim_label="", num_cols=1, col_index=0):
    ## if no figure axis was passed, then create one
    if ax is None: ax = plt.gca() # not really sure I should over-engineer this, but oh well
    ## calculate density of data
    dens, bin_edges = np.histogram(vals, bins=num_bins, density=True)
    ## normalise density
    dens_norm = dens / dens.sum()
    ## fill PDF with colour
    ax.fill_between(bin_edges[1:], dens_norm, step="pre", alpha=0.2, color=sns.color_palette("PuBu", n_colors=num_cols)[col_index])
    ## plot PDF lines
    ax.plot(bin_edges[1:], dens_norm, label=sim_label, drawstyle="steps", color=sns.color_palette("PuBu", n_colors=num_cols)[col_index])
    return ax


def plotHistogram(ax, vals, num_bins=10, sim_label="", num_cols=1, col_index=0):
    ## if no figure axis was passed, then create one
    if ax is None: ax = plt.gca() # not really sure I should over-engineer this, but oh well
    ## stacked histogram
    ax.hist(vals, bins=num_bins, histtype="step", fill=True, alpha=0.4, color=sns.color_palette("PuBu", n_colors=num_cols)[col_index], label=sim_label)
    ax.hist(vals, bins=num_bins, histtype="step", fill=False, color=sns.color_palette("PuBu", n_colors=num_cols)[col_index])
    return ax


def plotLIC_2D(vecs_x, vecs_y, kernel_num_pixels):
    ''' plotLIC_2D
    BASED ON: https://gitlab.com/szs/lic
    '''
    vecs_x = np.squeeze(vecs_x)
    vecs_y = np.squeeze(vecs_y)
    ## check data is the correct format
    assert len(vecs_x.shape) == 2
    assert len(vecs_y.shape) == 2
    assert vecs_x.shape[0] == vecs_y.shape[0]
    assert vecs_x.shape[1] == vecs_y.shape[1]
    assert kernel_num_pixels > 0
    assert kernel_num_pixels < max(vecs_x.shape[0], vecs_y.shape[0])
    assert kernel_num_pixels < max(vecs_x.shape[1], vecs_y.shape[1])
    ## generate kernel
    kernel = np.ones(kernel_num_pixels)
    ## generate noise
    np.random.seed(24041997)
    noise = np.random.random(vecs_x.shape)
    noise[0, :] = noise[:, 0] = noise[-1, :] = noise[:, -1] = 0.5
    ## calculate LIC image
    LIC_image = 1/2 * ( calcLIC_2D(+vecs_x, +vecs_y, noise, kernel[(kernel_num_pixels // 2):]) +      # advect forwards
                        calcLIC_2D(-vecs_x, -vecs_y, noise, kernel[(kernel_num_pixels // 2)-1::-1]) ) # advect backwards
    ## scale data
    LIC_image = normaliseData(LIC_image)
    return LIC_image


def calcLIC_2D(vecs_x, vecs_y, noise, kernel):
    ''' calcLIC_2D
    BASED ON: https://gitlab.com/szs/lic
    '''
    kernel_len = len(kernel)
    data_shape = noise.shape
    num_rows, num_cols = data_shape
    line_fx = np.zeros((num_rows, num_cols, kernel_len), dtype=np.int64)
    line_fy = np.zeros((num_rows, num_cols, kernel_len), dtype=np.int64)
    for row_index in range(num_rows): line_fy[row_index, :, 0] = list(range(num_cols))
    for col_index in range(num_cols): line_fx[:, col_index, 0] = list(range(num_rows))
    fx = np.full(data_shape, 0.5)
    fy = np.full(data_shape, 0.5)
    tx = np.full(data_shape, np.inf)
    ty = np.full(data_shape, np.inf)
    for loop_index in range(1, kernel_len):
        tx, ty, fx, fy, line_fx, line_fy = calcLIC_2D_point(vecs_x, vecs_y, tx, ty, fx, fy, line_fx, line_fy, loop_index, data_shape)
        fy = np.where((tx < ty),  ( fy + tx * vecs_y[line_fx[:,:,loop_index],line_fy[:,:,loop_index],] ), fy)
        fx = np.where((tx >= ty), ( fx + ty * vecs_x[line_fx[:,:,loop_index],line_fy[:,:,loop_index],] ), fx)
    lines_f = noise[line_fx, line_fy] * kernel
    return lines_f.mean(axis=2)


def calcLIC_2D_point(data_x, data_y, tx, ty, fx, fy, line_fx, line_fy, i, data_shape):
    ''' calcLIC_2D_point
    BASED ON: https://gitlab.com/szs/lic
    '''
    EPSILON = 1e-6
    num_rows, num_cols = data_shape
    tx = np.where(data_x > +EPSILON, (1 -fx)/ data_x, tx)
    tx = np.where(data_x < -EPSILON,    -fx / data_x, tx)
    ty = np.where(data_y > +EPSILON, (1 -fy)/ data_y, ty)
    ty = np.where(data_y < -EPSILON,    -fy / data_y, ty)
    line_fx[:, :, i] = np.where((tx <= ty) & (data_x >= 0), (line_fx[:,:, i-1]+1), line_fx[:,:, i])
    line_fx[:, :, i] = np.where((tx <= ty) & (data_x <= 0), (line_fx[:,:, i-1]-1), line_fx[:,:, i])
    line_fx[:, :, i] = np.where((tx >= ty),                  line_fx[:,:, i-1],    line_fx[:,:, i])
    line_fy[:, :, i] = np.where((tx >= ty) & (data_y >= 0), (line_fy[:,:, i-1]+1), line_fy[:,:, i])
    line_fy[:, :, i] = np.where((tx >= ty) & (data_y <= 0), (line_fy[:,:, i-1]-1), line_fy[:,:, i])
    line_fy[:, :, i] = np.where((tx <= ty),                  line_fy[:,:, i-1],    line_fy[:,:, i])
    fx = np.where((tx <= ty) & (data_x >= 0), 0, fx)
    fx = np.where((tx <= ty) & (data_x <= 0), 1, fx)
    fy = np.where((tx >= ty) & (data_y >= 0), 0, fy)
    fy = np.where((tx >= ty) & (data_y <= 0), 1, fy)
    line_fx = np.where(line_fx <= 0,        0,          line_fx)
    line_fx = np.where(line_fx >= num_rows, num_rows-1, line_fx)
    line_fy = np.where(line_fy <= 0,        0,          line_fy)
    line_fy = np.where(line_fy >= num_cols, num_cols-1, line_fy)
    return tx, ty, fx, fy, line_fx, line_fy


def plotData_3D(filepath_data, sim_time_iter, t_eddy, str_field_type, filepath_plot, pre_name):
    ''' plotData_3D
    BASED ON: https://stackoverflow.com/questions/40853556/3d-discrete-heatmap-in-matplotlib
    '''
    data = loadFLASH3DFieldMag(createFilePath([filepath_data, "Turb_hdf5_plt_cnt_"+str(sim_time_iter).zfill(4)]), 
        np.array([36, 36, 48]), np.array(([8, 8, 6])), str_field_type)
    x = np.linspace(0, 1, data.shape[0])
    X, Y = np.meshgrid(x, x)
    Z = np.ones(X.shape)
    ## initialise the figure
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ## plot data as contours
    print('\t> Plotting contours...')
    col_map_min = np.min(data)
    col_map_max = np.max(data)
    col_top   = mpl.cm.ScalarMappable(cmap="plasma", norm=mpl.colors.LogNorm()).to_rgba(data[0, :, :,])
    col_right = mpl.cm.ScalarMappable(cmap="plasma", norm=mpl.colors.LogNorm()).to_rgba(data[:, :, 0,])
    col_left  = mpl.cm.ScalarMappable(cmap="plasma", norm=mpl.colors.LogNorm()).to_rgba(data[:, 0, :,])
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, shade=False, facecolors=col_top)   # top
    ax.plot_surface(Z, X, Y, rstride=1, cstride=1, shade=False, facecolors=col_right) # left
    ax.plot_surface(X, Z, Y, rstride=1, cstride=1, shade=False, facecolors=col_left)  # right
    print('\t> Anotating plot...')
    ## set figure range
    plt.axis('off')
    ax.grid(False)
    ax.set_xlim((0, 1)); ax.set_ylim((0, 1)); ax.set_zlim((0, 1))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ## set 3D viewing angle
    ax.view_init(elev=25, azim=45)
    ## add time anotation
    ax.text2D(0.5, 1.0, r'$t/t_{\rm eddy} = $' + u' %0.1f'%(sim_time_iter / t_eddy), ha='center', va='top', transform=ax.transAxes)
    ## save image
    print("Saving figure...")
    fig_name = createFilePath([filepath_plot, pre_name]) + "_3D_" + str_field_type + "_" + str(sim_time_iter).zfill(3) + ".pdf"
    plt.savefig(fig_name)
    plt.close()
    print("Figure saved: " + fig_name)
    print(" ")


## END OF LIBRARY