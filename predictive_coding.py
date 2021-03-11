# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 13:45:30 2021

@author: seano
"""
import numpy as np
import h5py
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers


def load_data(embed_path,xy_path,split_ratio = 0.8, num_rows = 1000):
    
    with h5py.File(embed_path, 'r') as f:
        embeddings = pd.DataFrame(f["embeddings"][:num_rows,:])
    
    with h5py.File('data/basic_xy.h5', 'r') as f:
        xy_velocites = f["xy_data"][:num_rows,2:-1]
        orientations = f["xy_data"][:num_rows,-1:]
        
    # normalize xy features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_xy = pd.DataFrame(columns=["x_velocity","y_velocity"],
                          data=scaler.fit_transform(xy_velocites))
    scaled_orientations = pd.DataFrame(columns=["orientation"],
                          data=scaler.fit_transform(orientations))
    
    return embeddings, scaled_xy, scaled_orientations

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

def split_train_test(data, n_features, split_ratio=0.8, 
                     timesteps_in_past=5, targets = True):
    values = data.values
    n_train_samples = round(values.shape[0]*split_ratio)
    train = values[:n_train_samples, :]
    test = values[n_train_samples:, :]
    # split into input and outputs
    n_obs = timesteps_in_past * n_features
    train_X, train_y = train[:, :n_obs], train[:, -n_features:]
    test_X, test_y = test[:, :n_obs], test[:, -n_features:]
    print(train_X.shape, len(train_X), train_y.shape)
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], timesteps_in_past, n_features))
    test_X = test_X.reshape((test_X.shape[0], timesteps_in_past, n_features))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    if targets: return train_X, train_y, test_X, test_y
    else: return train_X, test_X

def vision_only_model(train_X, train_y,test_X, test_y,model_name):
    model = keras.Model.Sequential()
    model.add(layers.SimpleRNN(n_features, 
                               input_shape=(train_X.shape[1], train_X.shape[2])))
    model.compile(loss='mae', optimizer='adam')
    model.summary()
    
    history = model.fit(train_X, train_y, epochs=50, batch_size=50, 
                        validation_data=(test_X, test_y), verbose=2, 
                        shuffle=False)
    model.save(model_name)
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

def build_multimodal(): 
    embed_input = keras.Input(
        shape=(None,240), name="embed"
    )  
    velocity_input = keras.Input(shape=(None,2), name="velocity")  
    orientation_input = keras.Input(
        shape=(None,1), name="orientation"
    )  
    
    # Reduce sequence of embedded words in the title into a single 128-dimensional vector
    embed_rnn = layers.SimpleRNN(units=n_features,
                                 activation="sigmoid")(embed_input)
    # Reduce sequence of embedded words in the body into a single 32-dimensional vector
    velocity_rnn = layers.SimpleRNN(units=n_features,
                                    activation="sigmoid")(velocity_input)
    orient_rnn = layers.SimpleRNN(units=n_features,
                                  activation="sigmoid")(orientation_input)
    
    # Merge all available features into a single large vector via concatenation
    merged = layers.concatenate([embed_rnn, velocity_rnn, orient_rnn])
    dense = layers.Dense(units=n_features,activation="sigmoid")(merged)
    # Instantiate an end-to-end model predicting both priority and department
    model = keras.Model(
        inputs=[embed_input, velocity_input, orientation_input],
        outputs=[dense],
    )
    
    model.compile(
        optimizer=keras.optimizers.RMSprop(1e-3),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
    )
    model.summary()
    return model
#%% 
if __name__ == '__main__':
    # load dataset
    embed_path = 'data/basic_embeddings.h5'
    xy_path = 'data/basic_xy.h5'
    embeddings, scaled_xy, scaled_orientations = load_data(embed_path,xy_path)
    # specify the number of lag steps
    timesteps = 5
    n_features = 240
    # frame as supervised learning
    lagged_embeds = series_to_supervised(embeddings, timesteps, 1)
    lagged_velocity =  series_to_supervised(scaled_xy, timesteps, 1)
    lagged_orient =  series_to_supervised(scaled_orientations, timesteps, 1)
    train_embed_X, train_embed_y,test_embed_X , test_embed_y = split_train_test(lagged_embeds, 
                                                     n_features = 240)
    train_velocity_X, test_velocity_y = split_train_test(lagged_velocity,
                                                         n_features = 2, 
                                                     targets=False)
    train_orient_X, test_orient_y = split_train_test(lagged_orient, 
                                                     n_features = 1, 
                                                     targets=False)
    

#%%
    
    # =============================================================================
    # keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)
    # =============================================================================
    
    model = build_multimodal()
    history = model.fit(
        x = {"embed": train_embed_X, "velocity": train_velocity_X, 
             "orientation": train_orient_X},
        y = train_embed_y,
        epochs=10,
        batch_size=10,
    )
    # =============================================================================
    # model.save("trained_models/predictive_mutli_modal")
    # =============================================================================
    pyplot.plot(history.history['loss'], label='train')
# =============================================================================
#     pyplot.plot(history.history['val_loss'], label='test')
# =============================================================================
    pyplot.legend()
    pyplot.show()
#%% 


