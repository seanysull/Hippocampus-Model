# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 20:55:44 2020

@author: seano
"""

import matplotlib.pyplot as plt
from matplotlib.image import NonUniformImage
from matplotlib import cm
import matplotlib
matplotlib.use('Agg')
import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras import Model
import pandas as pd
from scipy.stats import binned_statistic_2d, binned_statistic, binned_statistic_dd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from scipy.ndimage import gaussian_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from denoiser import kl_divergence_regularizer

def custom_colorbar(mappable):
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
# =============================================================================
#     plt.clim(vmin=0,vmax=1)
# =============================================================================
    return cbar

def occupancy_map(xy_path, num_bins=20):
    with h5py.File(xy_path, 'r') as f:
        vec_data = f["vector_obs"][:]

    x_positions = vec_data[:,0]
    z_positions = vec_data[:,1]
    dummy_vals = np.ones(len(x_positions))         

    res = binned_statistic_2d(x_positions, z_positions, dummy_vals,
                              statistic="count", bins=num_bins)
    bin_statistic, x_edges, z_edges, binnumbers = res
# =============================================================================
#     bin_statistic = np.clip(bin_statistic,0,20)        
# =============================================================================
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, title="occupancy",
            aspect='equal', xlim=x_edges[[0, -1]], ylim=z_edges[[0, -1]])
    ax.grid(False)
    im = NonUniformImage(ax, interpolation='bilinear', cmap=cm.Reds)
    
    xcenters = (x_edges[:-1] + x_edges[1:]) / 2
    ycenters = (z_edges[:-1] + z_edges[1:]) / 2
    
    im.set_data(xcenters, ycenters, bin_statistic)

    ax.images.append(im)
    fig.colorbar(im)
    fig.savefig('rate_maps/occupancy_stretch.png',format='png', dpi=300)    
    
def plot_ratemaps_cae(xy_path, embed_path, 
                      stat_type="mean", num_bins=100):
    
    with h5py.File(xy_path, 'r') as f:
        vec_data = f["vector_obs"][:]
        
    with h5py.File(embed_path, 'r') as f:    
        embeddings = f["embeddings"][:]
    
    x_positions = vec_data[:,0]
    z_positions = vec_data[:,2]
    
    for cell in range(embeddings.shape[1]):
# =============================================================================
#         for cell in range(20):
# =============================================================================
        res = binned_statistic_2d(x_positions, z_positions, 
                                  embeddings[:,cell], 
                                  stat_type, bins=num_bins)
        bin_statistic, x_edges, z_edges, binnumbers = res
        bin_statistic = np.nan_to_num(bin_statistic)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, title='lec_cell'+str(cell),
                aspect='equal', xlim=x_edges[[0, -1]], ylim=z_edges[[0, -1]])
        ax.grid(False)
        im = NonUniformImage(ax, interpolation='nearest', cmap=cm.Greys)
        
        xcenters = (x_edges[:-1] + x_edges[1:]) / 2
        ycenters = (z_edges[:-1] + z_edges[1:]) / 2
        
        im.set_data(xcenters, ycenters, bin_statistic)
        ax.images.append(im)
        fig.colorbar(im)
        fig.savefig('rate_maps/lec/lec_cell_drop'+str(cell)+'.png',
                    format='png', dpi=300)
        plt.close(plt.gcf())
    
    print("images written")

def plot_ratemaps_hae(xy_path, embed_path, hippo_path,
                  stat_type="mean", num_bins=100, stretched = False):
    
    with h5py.File(xy_path, 'r') as f:
        vec_data = f["vector_obs"][:]
        
    with h5py.File(embed_path, 'r') as f:    
        embeddings = f["embeddings"][:]
    
    
    hippocampus = tf.keras.models.load_model(hippo_path, 
                                             custom_objects={"kl_divergence_regularizer":kl_divergence_regularizer},
                                             #custom_objects=None,
                                             compile=True)
    
    layer_names = ["EC_in","DG","CA3","CA1","EC_out"]
    
    layer_outputs = [hippocampus.get_layer(layer_name).output for 
                     layer_name in layer_names]
    
    intermediate_layer_model = Model(inputs=hippocampus.input,
                                     outputs=layer_outputs)
    
    EC_in, DG, CA3, CA1, EC_out = intermediate_layer_model.predict(embeddings)
    
    name_activations = zip(layer_names, (EC_in, DG, CA3, CA1, EC_out))
    
    x_positions = vec_data[:,0]
    z_positions = vec_data[:,1]
    
    for name, activation in name_activations:
        for cell in range(activation.shape[1]):
# =============================================================================
#         for cell in range(25):
# =============================================================================
            res = binned_statistic_2d(x_positions, z_positions, 
                                      activation[:,cell], 
                                      stat_type, bins=num_bins)
            bin_statistic, x_edges, z_edges, binnumbers = res
            bin_statistic = np.nan_to_num(bin_statistic)
# =============================================================================
#             scaler = MinMaxScaler()
#             scaled = scaler.fit_transform(bin_statistic)
# =============================================================================
# =============================================================================
#             thresh = bin_statistic < bin_statistic.max()*0.9
#             bin_statistic[thresh] = 0
# ============================================================================= 
            blurred = gaussian_filter(bin_statistic, sigma=1.5)

            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, title=name+' : '+str(cell),
                    aspect='equal', xlim=x_edges[[0, -1]], ylim=z_edges[[0, -1]])
            ax.grid(False)
            im = NonUniformImage(ax, interpolation='bilinear', cmap=cm.Greens)
            
            xcenters = (x_edges[:-1] + x_edges[1:]) / 2
            zcenters = (z_edges[:-1] + z_edges[1:]) / 2
            
            im.set_data(xcenters, zcenters, blurred)
# =============================================================================
#             im.set_clim(0,1)
# =============================================================================

            ax.images.append(im)
            colorbar(im)            
# =============================================================================
#             fig.colorbar(im).set_clim(0,1)
# =============================================================================
            if stretched:
                fig.savefig('rate_maps/'+name+'_stretched/'+name+'_sigmoid_reg_test'+str(cell)+'.png',
                            format='png', dpi=300)
            else:
                fig.savefig('rate_maps/'+name+'/'+name+'_sigmoid_reg'+str(cell)+'.png',
                            format='png', dpi=300)
            plt.close(plt.gcf())
    
        print("images written")

def plot_ratemaps_orientation(xy_path, embed_path, hippo_path,
                  stat_type="mean", num_bins=4):
    
    with h5py.File(xy_path, 'r') as f:
        orientations = f["vector_obs"][:,4]
        
    with h5py.File(embed_path, 'r') as f:    
        embeddings = f["embeddings"][:]
    
    
    hippocampus = tf.keras.models.load_model(hippo_path, 
                                             custom_objects={"kl_divergence_regularizer":kl_divergence_regularizer}, 
                                             compile=True)
    
    layer_names = ["EC_in","DG","CA3","CA1","EC_out"]
    
    layer_outputs = [hippocampus.get_layer(layer_name).output for 
                     layer_name in layer_names]
    
    intermediate_layer_model = Model(inputs=hippocampus.input,
                                     outputs=layer_outputs)
    
    EC_in, DG, CA3, CA1, EC_out = intermediate_layer_model.predict(embeddings)
    
    name_activations = zip(layer_names, (EC_in, DG, CA3, CA1, EC_out))
        
    for name, activation in name_activations:
        for cell in range(activation.shape[1]):
# =============================================================================
#         for cell in range(5):
# =============================================================================
            res = binned_statistic(orientations, 
                                   activation[:,cell], 
                                   stat_type, 
                                   bins=num_bins)
            
            bin_statistic, bin_edges, binnumbers = res

            bin_statistic = np.nan_to_num(bin_statistic)
            
# =============================================================================
#             thresh = bin_statistic < bin_statistic.max()*0.9
#             bin_statistic[thresh] = 0
# =============================================================================

            blurred = gaussian_filter(bin_statistic,sigma=50)
            scaled = minmax_scale(blurred.reshape(-1, 1))
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, title=name+' : '+str(cell))
# =============================================================================
#                     aspect='equal', xlim=x_edges[[0, -1]], ylim=z_edges[[0, -1]])
# =============================================================================
            ax.grid(False)
            ax.plot(bin_edges[1:], scaled, color="Purple")
# =============================================================================
#             ax.set_ylim(0,1)
#             ax.set_yticks([0.1,0.3,0.5,0.7,0.9])
# =============================================================================
            fig.savefig('rate_maps/'+name+'_Orient'+'/'+name+'_sigmoid_bl3_loadzabins'+str(cell)+'.png',
                        format='png', dpi=300)
            plt.close(plt.gcf())
    
        print("images written")

def conjunctive_place_orient(xy_path, embed_path, hippo_path,
                  stat_type="mean", num_bins=100):

    with h5py.File(xy_path, 'r') as f:
        orientations = f["vector_obs"][:,[0,1,4]]
        
    with h5py.File(embed_path, 'r') as f:    
        embeddings = f["embeddings"][:]
    
    
    hippocampus = tf.keras.models.load_model(hippo_path, 
                                             custom_objects={"kl_divergence_regularizer":kl_divergence_regularizer}, 
                                             compile=True)
    
    layer_names = ["EC_in","DG","CA3","CA1","EC_out"]
    
    layer_outputs = [hippocampus.get_layer(layer_name).output for 
                     layer_name in layer_names]
    
    intermediate_layer_model = Model(inputs=hippocampus.input,
                                     outputs=layer_outputs)
    
    EC_in, DG, CA3, CA1, EC_out = intermediate_layer_model.predict(embeddings)
    
    name_activations = zip(layer_names, (EC_in, DG, CA3, CA1, EC_out))
        
    for name, activation in name_activations:
        for cell in range(activation.shape[1]):
# =============================================================================
#         for cell in range(5):
# =============================================================================
            res = binned_statistic_dd(orientations, 
                                   activation[:,cell], 
                                   stat_type, 
                                   bins=num_bins)
            
            bin_statistic, bin_edges, binnumbers = res
            bin_statistic = np.nan_to_num(bin_statistic)
# =============================================================================
#             scaled = minmax_scale(bin_statistic.reshape(-1, 1))
# =============================================================================
# =============================================================================
#             bin_statistic = np.nan_to_num(bin_statistic)
# =============================================================================
# =============================================================================
#             thresh = bin_statistic < bin_statistic.max()*0.9
#             bin_statistic[thresh] = 0
# =============================================================================

            
            fig = plt.figure(figsize=(20, 20))
            for ind in range(num_bins[-1]):
                x_edges = bin_edges[0]
                z_edges = bin_edges[1]
                blurred = gaussian_filter(bin_statistic[:,:,ind],sigma=1.5)
                ax = fig.add_subplot(2,3,ind+1, title=name+' : '+str(cell),
                        aspect='equal', xlim=x_edges[[0, -1]], ylim=z_edges[[0, -1]])
                ax.grid(False)
                im = NonUniformImage(ax, interpolation='bilinear', cmap=cm.Greens)
                
                xcenters = (x_edges[:-1] + x_edges[1:]) / 2
                zcenters = (z_edges[:-1] + z_edges[1:]) / 2
                
                im.set_data(xcenters, zcenters, blurred)
    # =============================================================================
    #             im.set_clim(0,1)
    # =============================================================================
    
                ax.images.append(im)
            fig.colorbar()
            fig.savefig('rate_maps/'+name+'_conjunctive'+'/'+name+'_'+str(cell)+'.png',
                        format='png', dpi=300)
            plt.close(plt.gcf())
    
        print("images written")            
def normalise_velocity_orientation():
    with h5py.File(xy_path, 'r') as f:
        vec_data = f["vector_obs"][:10,4]
    
    a=1
    
if __name__ == "__main__":
    xy_path = "data/simulation_data_2807_50000steps.h5"
    embed_path = "data/simulation_data_2807_50000steps.h5_denoiseV4_50000.h5"
    hippo_path = "trained_models/denoiseV4.hdf5-07.hdf5_hippocampus_V10_sigmoid_reg.h5"

# =============================================================================
#     occupancy_map(xy_path)    
#     plot_ratemaps_hae(xy_path, embed_path, hippo_path, 
#                       stat_type="mean", num_bins=[25,25], stretched=True)
# =============================================================================
# =============================================================================
#     plot_ratemaps_orientation(xy_path, embed_path, hippo_path, stat_type="mean", num_bins=2880)
# =============================================================================
# =============================================================================
#     normalise_velocity_orientation()
# =============================================================================
    conjunctive_place_orient(xy_path, embed_path, 
                             hippo_path,stat_type="mean", 
                             num_bins=[50,50,6])
    
    
    
    
    