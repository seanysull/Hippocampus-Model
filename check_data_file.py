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
with h5py.File("data/simulation_data_2708_100000steps_stretched_Z.h5", 'r') as f:
    ims = f["visual_obs"][range(0,100000,4000)]

# =============================================================================
# for index in range(0,50000,1000):
# =============================================================================
for index in range(0,25):    
    fig=plt.imshow(ims[index],interpolation='none')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig('test_images/'+str(index)+'_stretch_z.png',
        bbox_inches='tight', pad_inches=0, format='png', dpi=300)
    print("image written")
