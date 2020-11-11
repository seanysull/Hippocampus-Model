# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 20:55:44 2020

@author: seano
"""
import os
import matplotlib
matplotlib.use('TkAgg',warn=False, force=True)
import matplotlib.pyplot as plt
from matplotlib.image import NonUniformImage
from matplotlib import cm
import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras import Model
import pandas as pd
from scipy.stats import binned_statistic_2d, binned_statistic, binned_statistic_dd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from scipy.ndimage import gaussian_filter, label, generate_binary_structure
# =============================================================================
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# =============================================================================
from denoiser import kl_divergence_regularizer
from tensorflow.keras.models import load_model

# =============================================================================
# def custom_colorbar(mappable):
#     last_axes = plt.gca()
#     ax = mappable.axes
#     fig = ax.figure
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.05)
#     cbar = fig.colorbar(mappable, cax=cax)
#     plt.sca(last_axes)
# # =============================================================================
# #     plt.clim(vmin=0,vmax=1)
# # =============================================================================
#     return cbar
# =============================================================================

def occupancy_map(xy_path, num_bins=20):
    with h5py.File(xy_path, 'r') as f:
        vec_data = f["xy_data"][:]

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
    fig.savefig('rate_maps/'+xy_path+'.png',format='png', dpi=300)    
    
def plot_ratemaps_hae(xy_path, embed_path, hippo_path,
                  stat_type="mean", num_bins=50, stretched = False):
    
    with h5py.File(xy_path, 'r') as f:
        vec_data = f["xy_data"][:]
        
    with h5py.File(embed_path, 'r') as f:    
        embeddings = f["embeddings"][:]
    
    
    hippocampus = tf.keras.models.load_model(hippo_path, 
                                             custom_objects={"kl_divergence_regularizer":kl_divergence_regularizer},
                                             #custom_objects=None,
                                             compile=True)
    
# =============================================================================
#     layer_names = ["EC_in","DG","CA3","CA1","EC_out"]
# =============================================================================
    layer_names = ["DG","CA3","CA1"]
    
    layer_outputs = [hippocampus.get_layer(layer_name).output for 
                     layer_name in layer_names]
    
    intermediate_layer_model = Model(inputs=hippocampus.input,
                                     outputs=layer_outputs)
    
# =============================================================================
#     EC_in, DG, CA3, CA1, EC_out = intermediate_layer_model.predict(embeddings)
#     
#     name_activations = zip(layer_names, (EC_in, DG, CA3, CA1, EC_out))
# =============================================================================

    DG, CA3, CA1 = intermediate_layer_model.predict(embeddings)
    
    name_activations = zip(layer_names, (DG, CA3, CA1))    
    
    x_positions = vec_data[:,0]
    z_positions = vec_data[:,1]
    
    for name, activation in name_activations:
# =============================================================================
#         for cell in range(activation.shape[1]):
# =============================================================================
        for cell in range(30):
            res = binned_statistic_2d(x_positions, z_positions, 
                                      activation[:,cell], 
                                      stat_type, bins=num_bins)
            bin_statistic, x_edges, z_edges, binnumbers = res
            bin_statistic = np.nan_to_num(bin_statistic)

            blurred = gaussian_filter(bin_statistic,sigma=1.5)
            thresh_95 = blurred.copy()
            thresh_95[thresh_95 < thresh_95.max()*0.96] = 0  
            blurred_2 = gaussian_filter(thresh_95,sigma=1)

            fig = plt.figure(figsize=(3.54, 3.54),frameon=False)
            ax = fig.add_subplot(111, title=name+' : '+str(cell),
                                 aspect='equal', xlim=x_edges[[0, -1]], 
                                 ylim=z_edges[[0, -1]], )
            ax.grid(False)
# =============================================================================
#             ax.imshow(thresh_95, interpolation="bilinear", cmap=cm.Greens)
# =============================================================================
            im = NonUniformImage(ax, interpolation='bilinear', cmap=cm.Purples)
            
            xcenters = (x_edges[:-1] + x_edges[1:]) / 2
            zcenters = (z_edges[:-1] + z_edges[1:]) / 2
            
            im.set_data(xcenters, zcenters, blurred_2)
# =============================================================================
#             im.set_clim(0,1)
# =============================================================================

            ax.images.append(im)
# =============================================================================
#             custom_colorbar(im)            
# =============================================================================
# =============================================================================
#             fig.colorbar(im).set_clim(0,1)
# =============================================================================
            if stretched:
                fig.savefig('rate_maps/'+name+'_stretched/'+name+'_blu96_'+str(cell)+'.png',
                            format='png', dpi=300)
            else:
                fig.savefig('rate_maps/'+name+'/'+name+'_zzzblu2_bilinear_2blu'+str(cell)+'.png',
                            format='png', dpi=300)
            plt.close(plt.gcf())
    
        print("images written")

def plot_ratemaps_orientation(xy_path, embed_path, hippo_path,
                  stat_type="mean", num_bins=4):
    
    with h5py.File(xy_path, 'r') as f:
        orientations = f["xy_data"][:,4]
        
    with h5py.File(embed_path, 'r') as f:    
        embeddings = f["embeddings"][:]
    
    
    hippocampus = tf.keras.models.load_model(hippo_path, 
                                             custom_objects={"kl_divergence_regularizer":kl_divergence_regularizer}, 
                                             compile=True)
    
    layer_names = ["DG","CA3","CA1","EC_out"]
    
    layer_outputs = [hippocampus.get_layer(layer_name).output for 
                     layer_name in layer_names]
    
    intermediate_layer_model = Model(inputs=hippocampus.input,
                                     outputs=layer_outputs)
    
    DG, CA3, CA1, EC_out = intermediate_layer_model.predict(embeddings)
    
    name_activations = zip(layer_names, (DG, CA3, CA1, EC_out))
        
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
        orientations = f["xy_data"][:,[0,1,4]]
        
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
                custom_colorbar(im)
# =============================================================================
#             fig.colorbar()
# =============================================================================
            fig.savefig('rate_maps/'+name+'_conjunctive'+'/'+name+'_'+str(cell)+'.png',
                        format='png', dpi=300)
            plt.close(plt.gcf())
    
        print("images written")            



def create_pcell_ratemaps(hippo_path,stat_type="mean", num_bins=[50,50]):
    list_filenames = [("data/basic_xy.h5",
                       "data/basi_embeddings.h5"),
                      ("data/mod25_xy.h5",
                       "data/mod25_embeddings.h5"),
                      ("data/mod50_xy.h5",
                       "data/mod50_embeddings.h5"),
                      ("data/mod75_xy.h5",
                       "data/mod75_embeddings.h5"),
                      ("data/mod100_xy.h5",
                       "data/mod100_embeddings.h5")]
# =============================================================================
#     list_filenames = [("data/simulation_data_1608_100000steps_stretched_X.h5",
#                        "data/simulation_data_1608_100000steps_stretched_X.h5_denoiseV4_embeddings.h5")]
# =============================================================================
# =============================================================================
#     for data_path, embed_path in list_filenames:
# =============================================================================
    for data_path, embed_path in list_filenames[:]:
        with h5py.File(data_path, 'r') as f:
            vec_data = f["xy_data"][:]
            
        with h5py.File(embed_path, 'r') as f:    
            embeddings = f["embeddings"][:]
            
        hippocampus = tf.keras.models.load_model(hippo_path, 
                                             custom_objects={"kl_divergence_regularizer":kl_divergence_regularizer},
                                             #custom_objects=None,
                                             compile=True)
    

        layer_names = ["DG","CA3","CA1"]
        
        layer_outputs = [hippocampus.get_layer(layer_name).output for 
                         layer_name in layer_names]
        
        intermediate_layer_model = Model(inputs=hippocampus.input,
                                         outputs=layer_outputs)
        
    
        DG, CA3, CA1 = intermediate_layer_model.predict(embeddings)
        
        name_activations = zip(layer_names, (DG, CA3, CA1))    
        
        x_positions = vec_data[:,0]
        z_positions = vec_data[:,1]
        
        
        for name, activation in name_activations:
            num_cells = activation.shape[1]
            ratemaps = np.zeros((num_cells,25,50))
            ratemapname = "rate_maps/numpy/"+data_path.replace("data/","")+"_"+name+"_50bin"
            for cell in range(num_cells):
    # =============================================================================
    #         for cell in range(10):
    # =============================================================================
                res = binned_statistic_2d(x_positions, z_positions, 
                                          activation[:,cell], 
                                          stat_type, bins=num_bins)
                bin_statistic, x_edges, binnumbers = res
                bin_statistic = np.nan_to_num(bin_statistic)
    # =============================================================================
    #             scaled = minmax_scale(bin_statistic)
    # =============================================================================
                blurred = gaussian_filter(bin_statistic,sigma=1.5)
                ratemaps[cell,:,:] = blurred
            
            np.save(ratemapname,ratemaps)
                
def circular_ratemaps(xy_path,
                        embed_path,
                        hippo_path,
                        stat_type="mean", 
                        num_bins=[25,25]):
    
    with h5py.File(xy_path, 'r') as f:
        orientations = f["xy_data"][:,4].round(decimals=3)
        
    with h5py.File(embed_path, 'r') as f:    
        embeddings = f["embeddings"][:]
    
    
    hippocampus = tf.keras.models.load_model(hippo_path, 
                                             custom_objects={"kl_divergence_regularizer":kl_divergence_regularizer}, 
                                             compile=True)
    
    layer_names = ["DG","CA3","CA1"]
    
    layer_outputs = [hippocampus.get_layer(layer_name).output for 
                     layer_name in layer_names]
    
    intermediate_layer_model = Model(inputs=hippocampus.input,
                                     outputs=layer_outputs)
    

    DG, CA3, CA1 = intermediate_layer_model.predict(embeddings)
    
    name_activations = zip(layer_names, (DG, CA3, CA1))    
        
    for name, activation in name_activations:
        num_cells = activation.shape[1]
        ratemaps = np.zeros((num_cells,num_bins))
        ratemapname = "rate_maps/numpy/"+name+"_orientation_base_bins"+str(num_bins)
            
        for cell in range(num_cells):
# =============================================================================
#         for cell in range(5):
# =============================================================================
            res = binned_statistic(orientations, 
                                   activation[:,cell], 
                                   stat_type, 
                                   bins=num_bins)
            
            bin_statistic, bin_edges, binnumbers = res

            bin_statistic = np.nan_to_num(bin_statistic)
            ratemaps[cell,:] = bin_statistic
        np.save(ratemapname,ratemaps)

def error_with_morphing():
    MODEL_NAME = "trained_models/denoiseV4.hdf5-07.hdf5_hippocampus_V11_sigmoid_reg.h5"
    base_path = "data/simulation_data_2807_50000steps.h5"
    embed_base_path = "data/simulation_data_2807_50000steps.h5_denoiseV4_embeddings.h5"
    x25_path = "data/simulation_data_2708_50000steps_morphed25.h5"
    embed_25_path = "data/simulation_data_2708_50000steps.h5_denoiseV4_morph25_embeddings.h5"
    x50_path = "data/simulation_data_2708_50000steps_morphed50.h5"
    embed_50_path = "data/simulation_data_2708_50000steps.h5_denoiseV4_morphed50_embeddings.h5"
    x75_path = "data/simulation_data_3108_50000steps_morphed75.h5"
    embed_75_path = "data/simulation_data_3108_50000steps.h5_denoiseV4_morphed75_embeddings.h5"
    x100_path = "data/simulation_data_3108_50000steps_morphed100.h5"
    embed_100_path = "data/simulation_data_3108_50000steps.h5_denoiseV4_morphed100_embeddings.h5"
    
    hippocampus = load_model(MODEL_NAME,
                             custom_objects={"kl_divergence_regularizer":kl_divergence_regularizer},
                             compile=True)  

    with h5py.File(embed_base_path, 'r') as f:
        embeddings_base = f["embeddings"][:]

    with h5py.File(embed_25_path, 'r') as f:
        embeddings_25 = f["embeddings"][:]
        
    with h5py.File(embed_50_path, 'r') as f:
        embeddings_50 = f["embeddings"][:]
        
    with h5py.File(embed_75_path, 'r') as f:
        embeddings_75 = f["embeddings"][:]
        
    with h5py.File(embed_100_path, 'r') as f:
        embeddings_100 = f["embeddings"][:]

    with h5py.File(base_path, 'r') as f:
        xy_base = f["xy_data"][:,[0,1]]

    with h5py.File(x25_path, 'r') as f:
        xy_25 = f["xy_data"][:,[0,1]]
        
    with h5py.File(x50_path, 'r') as f:
        xy_50 = f["xy_data"][:,[0,1]]
        
    with h5py.File(x75_path, 'r') as f:
        xy_75 = f["xy_data"][:,[0,1]]
        
    with h5py.File(x100_path, 'r') as f:
        xy_100 = f["xy_data"][:,[0,1]]

    pred_base = hippocampus.predict(embeddings_base)
    pred_25 = hippocampus.predict(embeddings_25 )
    pred_50 = hippocampus.predict(embeddings_50 )
    pred_75 = hippocampus.predict(embeddings_75 )
    pred_100 = hippocampus.predict(embeddings_100)

    square_diff_base = np.sum(np.square(np.subtract(embeddings_base, pred_base)),axis=1)
    square_diff_25 = np.sum(np.square(np.subtract(embeddings_25, pred_25)),axis=1)
    square_diff_50 = np.sum(np.square(np.subtract(embeddings_50, pred_50)),axis=1)
    square_diff_75 = np.sum(np.square(np.subtract(embeddings_75, pred_75)),axis=1)
    square_diff_100 = np.sum(np.square(np.subtract(embeddings_100, pred_100)),axis=1)
    
    res_base = binned_statistic_2d(xy_base[:,0], xy_base[:,1], square_diff_base, "mean", bins=25)[0]
    res_25 = binned_statistic_2d(xy_25[:,0], xy_25[:,1], square_diff_25, "mean", bins=25)[0]
    res_50 = binned_statistic_2d(xy_50[:,0], xy_50[:,1], square_diff_50, "mean", bins=25)[0]
    res_75 = binned_statistic_2d(xy_75[:,0], xy_75[:,1], square_diff_75, "mean", bins=25)[0]
    res_100 = binned_statistic_2d(xy_100[:,0], xy_100[:,1], square_diff_100, "mean", bins=25)[0]

    arrays = np.array([res_base,res_25,res_50,res_75, res_100])
    np.save("spatial_mse",arrays)

    pass


    
if __name__ == "__main__":
    xy_path = "data/basic_xy.h5"
    embed_path = "data/basic_embeddings.h5"
    hippo_path = "trained_models/denoiseV4.hdf5-07.hdf5_hippocampus_V11_sigmoid_reg.h5"

# =============================================================================
#     occupancy_map(xy_path)    
# =============================================================================
    plot_ratemaps_hae(xy_path, embed_path, hippo_path, 
                      stat_type="mean", num_bins=[50,50], stretched=True)
# =============================================================================
#     plot_ratemaps_orientation(xy_path, embed_path, hippo_path, stat_type="mean", num_bins=2880)
# =============================================================================
# =============================================================================
#     normalise_velocity_orientation()
# =============================================================================
# =============================================================================
#     conjunctive_place_orient(xy_path, embed_path, 
#                              hippo_path,stat_type="mean", 
#                              num_bins=[50,50,6])
# =============================================================================
# =============================================================================
#     create_pcell_ratemaps(hippo_path,stat_type="mean", num_bins=[25,50])
# =============================================================================
# =============================================================================
#     inspect_rate_maps()
# =============================================================================
# =============================================================================
#     plot_count_size()
# =============================================================================
# =============================================================================
#     circular_ratemaps(xy_path=xy_path,
#                         embed_path=embed_path,
#                         hippo_path=hippo_path,
#                         stat_type="mean", 
#                         num_bins=720)
# =============================================================================
# =============================================================================
#     error_with_morphing()
# =============================================================================
