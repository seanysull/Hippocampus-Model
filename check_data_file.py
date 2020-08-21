# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 22:12:29 2020

@author: seano
"""

import h5py
import matplotlib.pyplot as plt
import time

# =============================================================================
# with h5py.File("data/simulation_data_2607_10000steps.h5_denoiseV1_medio.h5", 'r') as f:
#     vec_data = f["embeddings"][:]
# =============================================================================
with h5py.File("data/simulation_data_2807_50000steps.h5_denoiseV5dense_2807_50000.h5", 'r') as f:
    ims = f["embeddings"][range(0,50000,100)]

# =============================================================================
# for index in range(0,50000,1000):
# =============================================================================
for index in range(0,50):    
    fig=plt.imshow(ims[index],interpolation='none')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig('test_images/'+str(index)+'.png',
        bbox_inches='tight', pad_inches=0, format='png', dpi=300)
    print("image written")
