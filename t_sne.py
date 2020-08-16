# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 11:41:11 2020

@author: seano
"""

import numpy as np
from sklearn.manifold import TSNE
import h5py
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def getImage(path):
    
    return OffsetImage(plt.imread(path), zoom=0.15)

embed_path = "data/simulation_data_0205_1000steps.h5_full_ladderv3_smalldata_sigmoidmiddle.hdf5"
ind=range(0,100000,100)
with h5py.File(embed_path, 'r') as f:
        embeddings = f["embeddings"][:]
emb = embeddings[ind,:]        
image_paths = ["visual_images/image_"+str(i)+".png" for i in range(10000)]
        
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=5000)
tsne_results = tsne.fit_transform(embeddings)

zipped = list(zip(tsne_results[:,0],tsne_results[:,1],image_paths))

fig, ax = plt.subplots()
ax.scatter(tsne_results[:,0], tsne_results[:,1])
print("scattered")
for x0, y0, path in zipped:
    ab = AnnotationBbox(getImage(path), (x0, y0), xycoords='data', frameon=False)
    ax.add_artist(ab)
    print("added_artist")
# =============================================================================
# ab = AnnotationBbox(getImage(image_paths[0]), (tsne_results[0,0], tsne_results[0,1]), frameon=False)
# ax.add_artist(ab)
# =============================================================================
plt.show(block=True) 

# =============================================================================
# a=1
# for x0, y0, path in zip(x, y,paths):
#     ab = AnnotationBbox(getImage(path), (x0, y0), frameon=False)
#     ax.add_artist(ab)
# visual_path = "data/simulation_data_0205_1000steps.h5"
# embed_path = "data/simulation_data_0205_1000steps.h5_ladderv8_smalldata_embeddings.h5"
# 
# =============================================================================



