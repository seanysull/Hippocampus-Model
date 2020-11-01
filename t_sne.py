# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 11:41:11 2020

@author: seano
"""

import numpy as np
from sklearn.manifold import TSNE
import h5py
import matplotlib
# =============================================================================
# matplotlib.use('tkagg')
# =============================================================================
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def getImage(path):
    
    return OffsetImage(plt.imread(path), zoom=0.05)

ind=range(0,50000,50)
im_path = "data/simulation_data_2807_50000steps.h5"
embed_path = "data/simulation_data_2807_50000steps.h5_denoiseV4_embeddings.h5"
with h5py.File(im_path, 'r') as f:
    ims = f["visual_obs"][ind]

# =============================================================================
# for ind,im in enumerate(ims):
#     plt.imshow(im)
#     plt.axis('off')
#     plt.savefig("visual_images/image_"+str(ind), bbox_inches="tight", pad_inches=0)
# 
# =============================================================================

with h5py.File(embed_path, 'r') as f:
        embeddings = f["embeddings"][ind]

image_paths = ["visual_images/image_"+str(i)+".png" for i in range(1000)]
        
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=5000)
tsne_results = tsne.fit_transform(embeddings[:1000,:])

zipped = list(zip(tsne_results[:,0],tsne_results[:,1],image_paths[:1000]))

fig, ax = plt.subplots()
ax.scatter(tsne_results[:,0], tsne_results[:,1])
print("scattered")
for x0, y0, path in zipped:
    ab = AnnotationBbox(getImage(path), (x0, y0), xycoords='data', frameon=False)
    ax.add_artist(ab)
    print("added_artist")
    
fig.savefig("plots/t_sne",bbox_inches="tight")
# =============================================================================
# ab = AnnotationBbox(getImage(image_paths[0]), (tsne_results[0,0], tsne_results[0,1]), frameon=False)
# ax.add_artist(ab)
# =============================================================================
# =============================================================================
# plt.show(block=True) 
# =============================================================================

# =============================================================================
# a=1
# for x0, y0, path in zip(x, y,paths):
#     ab = AnnotationBbox(getImage(path), (x0, y0), frameon=False)
#     ax.add_artist(ab)
# visual_path = "data/simulation_data_0205_1000steps.h5"
# embed_path = "data/simulation_data_0205_1000steps.h5_ladderv8_smalldata_embeddings.h5"
# 
# =============================================================================



