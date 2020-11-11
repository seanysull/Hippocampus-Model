# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 16:22:31 2020

@author: seano
"""
import h5py
import numpy as np

with h5py.File("data/simulation_data_100000steps_stretched.h5",'r') as f:
    vec_data = np.array(f["vector_obs"])

with h5py.File("data/stretch_xy.h5", 'w') as f:
    f.create_dataset("xy_data",data=vec_data)           
           