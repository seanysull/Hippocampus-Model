# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 14:26:05 2020

@author: seano
"""

# import the necessary packages
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras import losses
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from sklearn.preprocessing import minmax_scale
import tensorflow as tf
import numpy as np
import h5py
import time
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


def build_autoencoder(width=256, height=256, depth=3,filter_number=120, filter_size=3, latentDim=300):
    # initialize the input shape to be "channels last" along with
    # the channels dimension itself

    inputShape = (height, width, depth)
    chanDim = -1
       
    # define the input to the encoder
    inputs = Input(shape=inputShape)
    x = GaussianNoise(0.2)(inputs)
    
    x = Conv2D(filter_number, filter_size, strides=2, padding="same")(x)       
    x = BatchNormalization(axis=chanDim)(x)
    x = LeakyReLU(alpha=0.2)(x)
   
    x = Conv2D(filter_number, filter_size, strides=2, padding="same")(x)    
    x = BatchNormalization(axis=chanDim)(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(filter_number, filter_size, strides=2, padding="same")(x)        
    x = BatchNormalization(axis=chanDim)(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(filter_number, filter_size, strides=2, padding="same")(x)    
    x = BatchNormalization(axis=chanDim)(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(filter_number, filter_size, strides=2, padding="same")(x)    
    x = BatchNormalization(axis=chanDim)(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(filter_number, filter_size, strides=2, padding="same")(x)    
    x = BatchNormalization(axis=chanDim)(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(int(filter_number/2), filter_size, strides=2, padding="same", 
               activation="sigmoid")(x)    
# =============================================================================
#     x = BatchNormalization(axis=chanDim)(x)
# =============================================================================

    volumeSize = K.int_shape(x)
    latent = Flatten()(x)
    # flatten the network and then construct our latent vector
       
    x = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(latent)

    x = UpSampling2D(size=2, interpolation="nearest")(x)
    x = Conv2DTranspose(filter_number, filter_size, strides=1, padding="same")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = UpSampling2D(size=2, interpolation="nearest")(x)    
    x = Conv2DTranspose(filter_number, filter_size, strides=1, padding="same")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = UpSampling2D(size=2, interpolation="nearest")(x)
    x = Conv2DTranspose(filter_number, filter_size, strides=1, padding="same")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = UpSampling2D(size=2, interpolation="nearest")(x)
    x = Conv2DTranspose(filter_number, filter_size, strides=1, padding="same")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = UpSampling2D(size=2, interpolation="nearest")(x)
    x = Conv2DTranspose(filter_number, filter_size, strides=1, padding="same")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = UpSampling2D(size=2, interpolation="nearest")(x)
    x = Conv2DTranspose(filter_number, filter_size, strides=1, padding="same")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = UpSampling2D(size=2, interpolation="nearest")(x)    
    x = Conv2DTranspose(filter_number, filter_size, strides=1, padding="same")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = LeakyReLU(alpha=0.2)(x)
    outputs = Conv2DTranspose(depth, filter_size, activation="sigmoid", padding="same")(x)
    
    autoencoder = Model(inputs, outputs, name="autoencoder")
   
    # return a 3-tuple of the encoder, decoder, and autoencoder
    return autoencoder

def inspect():
    autoencoder = build_autoencoder()
    autoencoder.summary()    

def train_autoencoder(DATA_PATH, MODEL_NAME, DATA_NAME = "visual_obs",
                      EPOCHS = 2,BATCH = 50, NUM_SAMPLES = 100000):
    # construct our convolutional autoencoder
    print("[INFO] building autoencoder...")
# =============================================================================
#     autoencoder = build_autoencoder()
# =============================================================================
    autoencoder = load_model(MODEL_NAME, compile=True)    
    autoencoder.summary()
# =============================================================================
#     logdir="logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
# =============================================================================
    logdir="logs"
    tensorboard_callback = TensorBoard(log_dir=logdir,
                                       histogram_freq=1,
                                       write_graph=True,
                                       write_images=True,
                                       update_freq=1)
    filepath = MODEL_NAME+"-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    opt = Adam(lr=1e-3)

    autoencoder.compile(loss="mse", optimizer=opt)
# =============================================================================
#     autoencoder.compile(loss="binary_crossentropy", optimizer=opt)
# =============================================================================
    
    # construct training data generator and validation generator
    number_train_samples = int(np.floor(NUM_SAMPLES*0.9))
    number_val_samples = int(np.floor(NUM_SAMPLES*0.1))
    indexes = np.arange(0,NUM_SAMPLES*5,5)
    np.random.shuffle(indexes)
    train_indexes = indexes[:number_train_samples]
    val_indexes = indexes[number_train_samples:number_train_samples+number_val_samples]

    
    train_generator = DataGenerator(train_indexes, DATA_PATH, DATA_NAME,
                     to_fit=True, batch_size=BATCH, shuffle=True)
    
    val_generator = DataGenerator(val_indexes, DATA_PATH, DATA_NAME,
                     to_fit=True, batch_size=BATCH, shuffle=True)
    
    callbacks=[checkpoint, tensorboard_callback]
    # train the convolutional autoencoder
    H = autoencoder.fit(train_generator,
        epochs=EPOCHS, validation_data = val_generator,
        workers=4, use_multiprocessing=False,
        callbacks=callbacks)
    
    ts = time.time()
    autoencoder.save(MODEL_NAME, save_format= "h5")
       
    # construct a plot that plots and saves the training history
    N = np.arange(0, EPOCHS)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, H.history["loss"], label="train_loss")
    plt.plot(N, H.history["val_loss"], label="val_loss")
    plt.yscale("log")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(MODEL_NAME+"_train_result"+".png")
    
def generate_embeddings(DATA_PATH, DATA_NAME,EMBED_NAME, MODEL_NAME, BATCH, NUM_SAMPLES):
    
    autoencoder = load_model(MODEL_NAME, compile=True)    
        
    predict_generator = DataGenerator(range(NUM_SAMPLES), DATA_PATH, DATA_NAME,
                     to_fit=False, batch_size=BATCH, shuffle=False)
    
    layer_name = autoencoder.layers[21].name
    encoder = Model(inputs=autoencoder.input,
                                 outputs=autoencoder.get_layer(layer_name).output)
    encoded = encoder.predict(predict_generator)
# =============================================================================
#     encoded = encoded.reshape((NUM_SAMPLES,2*2*60))
# =============================================================================
    
    with h5py.File(EMBED_NAME,'w') as f:
           f.create_dataset('embeddings', data=encoded)
           
def train_hippocampus(DATA_PATH, DATA_NAME = "embeddings", 
                      HMODEL_NAME = "trained_models/hippocampus_ladderv8",
                      BATCH = 100, NUM_SAMPLES = 50000, embed_dim = 240,
                      n_DG = 160, n_CA3 = 80, n_CA1 = 160, EPOCHS=50, activation="swish"):
    
    
    hippocampus = Sequential([
        Dense(embed_dim,
              activation=activation,
              name="EC_in",
              input_shape=(embed_dim,)),
        Dense(n_DG, 
              activation=activation, 
              name='DG'),
        Dense(n_CA3,
              activation=activation, 
              name='CA3'),
        Dense(n_CA1, 
              activation=activation, 
              name='CA1'),
        Dense(embed_dim, 
              activation=activation, 
              name='EC_out')
    ])
    
    opt = Adam(lr=1e-4)
    hippocampus.compile(loss="mse", optimizer=opt)
    hippocampus.summary()
    
    
    number_train_samples = int(np.floor(NUM_SAMPLES*0.9))
    number_val_samples = int(np.floor(NUM_SAMPLES*0.1))
    indexes = np.arange(NUM_SAMPLES)
    np.random.shuffle(indexes)
    train_indexes = indexes[:number_train_samples]
    val_indexes = indexes[number_train_samples:number_train_samples+number_val_samples]
    # =============================================================================
    # test_indexes = indexes[number_train_samples+number_val_samples:]
    # =============================================================================
    
    train_generator = DataGenerator(train_indexes, DATA_PATH, DATA_NAME,
                     to_fit=True, batch_size=BATCH, shuffle=True)
    
    val_generator = DataGenerator(val_indexes, DATA_PATH, DATA_NAME,
                     to_fit=True, batch_size=BATCH, shuffle=True)
    
    history = hippocampus.fit(train_generator, epochs=EPOCHS, 
                              validation_data = val_generator,
                              workers=4, use_multiprocessing=False)
    
    hippocampus.save(HMODEL_NAME, save_format= "h5")
    N = np.arange(0, EPOCHS)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, history.history["loss"], label="train_loss")
    plt.plot(N, history.history["val_loss"], label="val_loss")
    plt.yscale("log")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(HMODEL_NAME+"_train_result.png")

    
class DataGenerator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building 
    data generator for training and prediction.
    """
    def __init__(self, indexes, data_path, dataset_name,
                 to_fit=True, batch_size=50, shuffle=True, 
                 to_norm=False):
        """Initialization
        :param num_samples: number of samples in dataset
        :param data_path: path to data file location        
        :param dataset_name: name of datset in datafile
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of image channels
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.indexes = np.sort(indexes)
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        self.to_norm = to_norm

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        X = self._generate_X(indexes)
        if self.to_norm:
            X = minmax_scale(X, feature_range=(0, 1), axis=0, copy=True)
            
        if self.to_fit:
            return X, X
        else:
            return X

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _generate_X(self, indexes):
        """Generates data containing batch_size images
        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        # Generate data
        with h5py.File(self.data_path, 'r') as f:
            indexes = np.sort(indexes)
            if self.dataset_name == "embeddings":
                X = f[self.dataset_name][indexes]
            else:
                X = f[self.dataset_name][indexes, :, :, :]
        return X

if __name__ == '__main__':
    DATA_PATH = "data/simulation_data_2807_50000steps.h5"
    MODEL_NAME = "trained_models/denoiseV4.hdf5-07.hdf5"
    DATA_NAME = "visual_obs"
    EMBED_NAME = DATA_PATH+"_denoiseV4_50000.h5"
    EPOCHS = 50
    BATCH = 50
    NUM_SAMPLES = 50000

# =============================================================================
#     inspect()
# =============================================================================
# =============================================================================
#     train_autoencoder(DATA_PATH,MODEL_NAME, DATA_NAME, EPOCHS, BATCH, NUM_SAMPLES)
# =============================================================================
    generate_embeddings(DATA_PATH, DATA_NAME,EMBED_NAME, MODEL_NAME, BATCH, NUM_SAMPLES)
# =============================================================================
#     train_hippocampus(DATA_PATH=EMBED_NAME, DATA_NAME = "embeddings", 
#                       HMODEL_NAME = MODEL_NAME+"_hippocampus_V10.h5",
#                       BATCH = 50, NUM_SAMPLES = 50000, embed_dim = 240,
# =============================================================================
                      n_DG = 160, n_CA3 = 60, n_CA1 = 160, EPOCHS=200,activation="softplus" )


