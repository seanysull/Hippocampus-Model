# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 11:41:11 2020

@author: seano
"""

import numpy as np
from sklearn.manifold import TSNE
import h5py
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
# =============================================================================
# matplotlib.use('tkagg')
# =============================================================================
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.preprocessing import minmax_scale
from sklearn.decomposition import PCA
from denoiser import kl_divergence_regularizer
import tensorflow as tf
from tensorflow.keras.models import Model

def lerp(x, a, b):
   return a + x * (b-a)

def get_color_array(xy, a, b, c, d):
    x = xy[0]
    y = xy[1]
    return np.array([lerp(y, lerp(x, a[i], b[i]),
                             lerp(x, c[i], d[i])) for i in range(3)])

def get_color(x, y, a, b, c, d):
    return np.array([lerp(y, lerp(x, a[i], b[i]),
                             lerp(x, c[i], d[i])) for i in range(3)])

def getImage(path):
    
    return OffsetImage(plt.imread(path), zoom=0.05)

def cnn_2d():
    ind=range(0,50000,50)
    im_path = "data/simulation_data_2807_50000steps.h5"
    embed_path = "data/basic_embeddings.h5"
    with h5py.File(im_path, 'r') as f:
        ims = f["visual_obs"][ind]
    
    for ind,im in enumerate(ims):
        plt.imshow(im)
        plt.axis('off')
        plt.savefig("visual_images/image_"+str(ind), bbox_inches="tight", pad_inches=0)
    
    
    with h5py.File(embed_path, 'r') as f:
            embeddings = np.array(f["embeddings"])
    
    image_paths = ["visual_images/image_"+str(i)+".png" for i in range(1000)]
            
    tsne = TSNE(n_components=3, verbose=1, perplexity=500, n_iter=5000)
    tsne_results = tsne.fit_transform(embeddings)
    a=1
    zipped = list(zip(tsne_results[:,0],tsne_results[:,1],image_paths[:1000]))
    
    fig, ax = plt.subplots()
    ax.scatter(tsne_results[:,0], tsne_results[:,1])
    print("scattered")
    for x0, y0, path in zipped:
        ab = AnnotationBbox(getImage(path), (x0, y0), xycoords='data', frameon=False)
        ax.add_artist(ab)
        print("added_artist")
        
    fig.savefig("plots/t_sne",bbox_inches="tight")
    ab = AnnotationBbox(getImage(image_paths[0]), (tsne_results[0,0], tsne_results[0,1]), frameon=False)
    ax.add_artist(ab)
    # =============================================================================
    # plt.show(block=True) 
    # =============================================================================
    
    
    for x0, y0, path in zip(x, y,paths):
        ab = AnnotationBbox(getImage(path), (x0, y0), frameon=False)
        ax.add_artist(ab)
    visual_path = "data/simulation_data_0205_1000steps.h5"
    embed_path = "data/simulation_data_0205_1000steps.h5_ladderv8_smalldata_embeddings.h5"

def hippo_3d():
    
    xy_path = "data/basic_xy.h5"
    embed_path = "data/basic_embeddings.h5"
    hippo_path = "trained_models/denoiser_hippocampus.h5"
    w = h = 55
    verts = [[0,255,0],[0,0,255],[255,0,0],[255,255,0]]
    ind=range(0,50000,10)
    
    with h5py.File(xy_path, 'r') as f:
        xy_coord = np.array(f["xy_data"][ind])

    with h5py.File(embed_path, 'r') as f:
        embeddings = np.array(f["embeddings"][ind])
        
    xy_coord = xy_coord[:,[0,1]]
        
    hippocampus = tf.keras.models.load_model(hippo_path, 
                                             custom_objects={"kl_divergence_regularizer":kl_divergence_regularizer},
                                             compile=True)
    layer_names = ["DG","CA3","CA1"]
    
    layer_outputs = [hippocampus.get_layer(layer_name).output for 
                     layer_name in layer_names]
    
    intermediate_layer_model = Model(inputs=hippocampus.input,
                                     outputs=layer_outputs)
    

    DG, CA3, CA1 = intermediate_layer_model.predict(embeddings)
    
    pca = PCA(n_components=50)
    DG_pca = pca.fit_transform(DG)
    CA3_pca = pca.fit_transform(CA3)
    CA1_pca = pca.fit_transform(CA1)
    
    scaled_xy = minmax_scale(xy_coord)
    colors = np.apply_along_axis(get_color_array, 1, scaled_xy, *verts)
    colors = colors/255.0
    
    tsne = TSNE(n_components=3, verbose=1, perplexity=25, n_iter=5000)
    tsne_dg = tsne.fit_transform(DG_pca)
    tsne_ca3 = tsne.fit_transform(CA3_pca)
    tsne_ca1 = tsne.fit_transform(CA1_pca)
# =============================================================================
#     tsne = TSNE(n_components=2, verbose=1, perplexity=150, n_iter=5000)
#     tsne_dg = tsne.fit_transform(DG_pca)
#     tsne_ca3 = tsne.fit_transform(CA3_pca)
#     tsne_ca1 = tsne.fit_transform(CA1_pca)
# =============================================================================
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(tsne_dg[:,0],tsne_dg[:,1],tsne_dg[:,2], c=colors, marker="+")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(tsne_ca3[:,0], tsne_ca3[:,1],tsne_ca3[:,2], c=colors, marker="+")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(tsne_ca1[:,0],tsne_ca1[:,1],tsne_ca1[:,2], c=colors, marker="+")
 
# =============================================================================
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.scatter(tsne_dg[:,0],tsne_dg[:,1], c=colors, marker="+")
#     ax.set_title("DG")
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.scatter(tsne_ca3[:,0], tsne_ca3[:,1], c=colors, marker="+")
#     ax.set_title("CA3")
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.scatter(tsne_ca1[:,0],tsne_ca1[:,1], c=colors, marker="+")
#     ax.set_title("CA1")
# =============================================================================
# =============================================================================
#     plt.show()   
# =============================================================================
    
def look_map():
    w = h = 200
    verts = [[0,255,0],[0,0,255],[255,0,0],[255,255,0]]
    img = np.empty((h,w,3), np.uint8)
    for y in range(h):
        for x in range(w):
            img[y,x] = get_color(x/w, y/h, *verts)
    plt.imshow(img)
    plt.show()    


if __name__ == '__main__':
# =============================================================================
#     look_map()
# =============================================================================
    hippo_3d()
# =============================================================================
#     look_map()
# =============================================================================
