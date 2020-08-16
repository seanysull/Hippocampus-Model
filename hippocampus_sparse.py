# -*- coding: utf-8 -*-
"""
Created on Wed May  6 04:04:52 2020

@author: seano
"""

import tensorflow as tf
import numpy as np
from tensorflow_model_optimization.sparsity import keras as sparsity
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam
import numpy as np
import h5py
from conv_autoencoder import  DataGenerator

def train_hippocampus(DATA_PATH, DATA_NAME = "embeddings", 
                      HMODEL_NAME = "trained_models/hippocampus",
                      BATCH = 100, NUM_SAMPLES = 50000, embed_dim = 200,
                      n_DG = 160, n_CA3 = 80, n_CA1 = 160, EPOCHS=50):
    


    number_train_samples = int(np.floor(NUM_SAMPLES*0.9))
    number_val_samples = int(np.floor(NUM_SAMPLES*0.1))
    indexes = np.arange(NUM_SAMPLES)
    np.random.shuffle(indexes)
    train_indexes = indexes[:number_train_samples]
    val_indexes = indexes[number_train_samples:number_train_samples+number_val_samples]

    end_step = np.ceil(1.0 * NUM_SAMPLES / BATCH).astype(np.int32) * EPOCHS

    train_generator = DataGenerator(train_indexes, DATA_PATH, DATA_NAME,
                     to_fit=True, batch_size=BATCH, shuffle=True)
    
    val_generator = DataGenerator(val_indexes, DATA_PATH, DATA_NAME,
                     to_fit=True, batch_size=BATCH, shuffle=True)
    
    
    hippocampus = Sequential([
        Dense(n_DG, activation='relu', name='DG', input_shape=(embed_dim,)),
        Dense(n_CA3, activation='relu', name='CA3'),
        Dense(n_CA1, activation='relu', name='CA1'),
        Dense(embed_dim, activation='relu', name='EC')
    ])

    pruning_schedule = sparsity.PolynomialDecay(
                          initial_sparsity=0.5, final_sparsity=0.9,
                          begin_step=2000, end_step=end_step, frequency=100)
    
    model_for_pruning = sparsity.prune_low_magnitude(
        hippocampus, pruning_schedule=pruning_schedule)
    
    opt = Adam(lr=1e-3)
    model_for_pruning.compile(loss="mse", optimizer=opt)
    model_for_pruning.summary()

    callbacks = [sparsity.UpdatePruningStep()]
    
    model_for_pruning.fit(train_generator, epochs=EPOCHS, 
                              validation_data = val_generator, 
                              callbacks=callbacks,
                              workers=4, use_multiprocessing=False)
    
    final_model = sparsity.strip_pruning(model_for_pruning)
    final_model.summary()

    final_model.save(HMODEL_NAME, save_format= "h5")

    
if __name__ == '__main__':
    DATA_PATH = "data/simulation_data_0105_100000steps.h5"
    DATA_NAME = "visual_obs"
    embed_path = DATA_PATH+"_embeddings_mse.h5"
    MODEL_NAME = "trained_models/200hu_2epoch_3x3_mse_0205_relu.h5"
    EPOCHS = 2
    BATCH = 25
    DIM = [256, 256]
    CHANNELS = 3
    NUM_SAMPLES = 100000
    end_step = np.ceil(1.0 * NUM_SAMPLES / BATCH).astype(np.int32) * EPOCHS

    train_hippocampus(DATA_PATH=embed_path, DATA_NAME = "embeddings", 
                      HMODEL_NAME = MODEL_NAME+"_hippocampus_pruned.h5",
                      BATCH = 50, NUM_SAMPLES = 100000, embed_dim = 200,
                      n_DG = 160, n_CA3 = 80, n_CA1 = 160, EPOCHS=20)