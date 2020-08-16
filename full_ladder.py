# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 18:20:26 2020

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
from tensorflow.keras.layers import UpSampling2D
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


def build_autoencoder(width=256, height=256, depth=3,filter_number=64, filter_size=3, latentDim=300):
    # initialize the input shape to be "channels last" along with
    # the channels dimension itself

    inputShape = (height, width, depth)
    chanDim = -1
    noise_std = 0.2   
    # define the input to the encoder
    inputs = Input(shape=inputShape)
    # create shared layers
    l1_conv = Conv2D(int(filter_number*1.5), filter_size, strides=2, padding="same")
    l1_batch = BatchNormalization(axis=chanDim)   
    l1_nonlin = LeakyReLU(alpha=0.2)
    #clean layer 1
    clean_l1_conv = l1_conv(inputs)
    clean_l1_batch = l1_batch(clean_l1_conv)    
    clean_l1_nonlin = l1_nonlin(clean_l1_batch)
    # noisy layer 1
    noisy_image = GaussianNoise(noise_std)(inputs)
    noisy_l1_conv = l1_conv(noisy_image)
    noisy_l1_batch = l1_batch(noisy_l1_conv)
    noisy_l1_batch = GaussianNoise(noise_std)(noisy_l1_batch)
    noisy_l1_nonlin = l1_nonlin(noisy_l1_batch)   
    
    l2_conv = Conv2D(filter_number, filter_size, strides=2, padding="same")
    l2_batch = BatchNormalization(axis=chanDim)   
    l2_nonlin = LeakyReLU(alpha=0.2)
    #clean layer 2
    clean_l2_conv = l2_conv(clean_l1_nonlin)
    clean_l2_batch = l2_batch(clean_l2_conv)    
    clean_l2_nonlin = l2_nonlin(clean_l2_batch)
    # noisy layer 2
    noisy_l2_conv = l2_conv(noisy_l1_nonlin)
    noisy_l2_batch = l2_batch(noisy_l2_conv)
    noisy_l2_batch = GaussianNoise(noise_std)(noisy_l2_batch)
    noisy_l2_nonlin = l2_nonlin(noisy_l2_batch)   
    
    l3_conv = Conv2D(filter_number, filter_size, strides=2, padding="same")
    l3_batch = BatchNormalization(axis=chanDim)   
    l3_nonlin = LeakyReLU(alpha=0.2)
    #clean layer 3
    clean_l3_conv = l3_conv(clean_l2_nonlin)
    clean_l3_batch = l3_batch(clean_l3_conv)    
    clean_l3_nonlin = l3_nonlin(clean_l3_batch)
    # noisy layer 3
    noisy_l3_conv = l3_conv(noisy_l2_nonlin)
    noisy_l3_batch = l3_batch(noisy_l3_conv)
    noisy_l3_batch = GaussianNoise(noise_std)(noisy_l3_batch)
    noisy_l3_nonlin = l3_nonlin(noisy_l3_batch)  
    
    
    l4_conv = Conv2D(filter_number, filter_size, strides=2, padding="same")
    l4_batch = BatchNormalization(axis=chanDim)   
    l4_nonlin = LeakyReLU(alpha=0.2)
    #clean layer 4
    clean_l4_conv = l4_conv(clean_l3_nonlin)
    clean_l4_batch = l4_batch(clean_l4_conv)    
    clean_l4_nonlin = l4_nonlin(clean_l4_batch)
    # noisy layer 4
    noisy_l4_conv = l4_conv(noisy_l3_nonlin)
    noisy_l4_batch = l4_batch(noisy_l4_conv)
    noisy_l4_batch = GaussianNoise(noise_std)(noisy_l4_batch)
    noisy_l4_nonlin = l4_nonlin(noisy_l4_batch)  
    
    
    l5_conv = Conv2D(filter_number, filter_size, strides=2, padding="same")
    l5_batch = BatchNormalization(axis=chanDim)   
    l5_nonlin = LeakyReLU(alpha=0.2)
    #clean layer 5
    clean_l5_conv = l5_conv(clean_l4_nonlin)
    clean_l5_batch = l5_batch(clean_l5_conv)    
    clean_l5_nonlin = l5_nonlin(clean_l5_batch)
    # noisy layer 5
    noisy_l5_conv = l5_conv(noisy_l4_nonlin)
    noisy_l5_batch = l5_batch(noisy_l5_conv)
    noisy_l5_batch = GaussianNoise(noise_std)(noisy_l5_batch)
    noisy_l5_nonlin = l5_nonlin(noisy_l5_batch)  
    
    
    l6_conv = Conv2D(filter_number, filter_size, strides=2, padding="same")
    l6_batch = BatchNormalization(axis=chanDim)   
    l6_nonlin = LeakyReLU(alpha=0.2)
    #clean layer 6
    clean_l6_conv = l6_conv(clean_l5_nonlin)
    clean_l6_batch = l6_batch(clean_l6_conv)    
    clean_l6_nonlin = l6_nonlin(clean_l6_batch)
    # noisy layer 6
    noisy_l6_conv = l6_conv(noisy_l5_nonlin)
    noisy_l6_batch = l6_batch(noisy_l6_conv)
    noisy_l6_batch = GaussianNoise(noise_std)(noisy_l6_batch)
    noisy_l6_nonlin = l6_nonlin(noisy_l6_batch)  
    
    l7_conv = Conv2D(filter_number, filter_size, 
                     strides=2, padding="same",activation="sigmoid")
    l7_batch = BatchNormalization(axis=chanDim)   
# =============================================================================
#     l7_nonlin = LeakyReLU(alpha=0.2)
# =============================================================================
    #clean layer 7
    clean_l7_conv = l7_conv(clean_l6_nonlin)
    clean_l7_batch = l7_batch(clean_l7_conv)    
# =============================================================================
#     clean_l7_nonlin = l7_nonlin(clean_l7_batch)
# =============================================================================
    # noisy layer 7
    noisy_l7_conv = l7_conv(noisy_l6_nonlin)
    noisy_l7_batch = l7_batch(noisy_l7_conv)
    noisy_l7_batch = GaussianNoise(noise_std)(noisy_l7_batch)
# =============================================================================
#     noisy_l7_nonlin = l7_nonlin(noisy_l7_batch) 
# =============================================================================
    
    latent = Flatten()(clean_l7_batch)    

# =============================================================================
#     dec_l7 = UpSampling2D(size=2, interpolation="nearest")(noisy_l7_nonlin)
# =============================================================================
    dec_l7 = UpSampling2D(size=2, interpolation="nearest")(noisy_l7_batch)
    dec_l7 = Conv2DTranspose(filter_number, filter_size, strides=1, padding="same")(dec_l7)
    dec_l7 = BatchNormalization(axis=chanDim)(dec_l7)
    dec_l7 = LeakyReLU(alpha=0.2)(dec_l7)
    
    dec_l6 = Combine()([dec_l7, noisy_l6_nonlin, clean_l6_nonlin])
    dec_l6 = UpSampling2D(size=2, interpolation="nearest")(dec_l6)
    dec_l6 = Conv2DTranspose(filter_number, filter_size, strides=1, padding="same")(dec_l6)
    dec_l6 = BatchNormalization(axis=chanDim)(dec_l6)
    dec_l6 = LeakyReLU(alpha=0.2)(dec_l6)
    
    dec_l5 = Combine()([dec_l6, noisy_l5_nonlin, clean_l5_nonlin])
    dec_l5 = UpSampling2D(size=2, interpolation="nearest")(dec_l5)
    dec_l5 = Conv2DTranspose(filter_number, filter_size, strides=1, padding="same")(dec_l5)
    dec_l5 = BatchNormalization(axis=chanDim)(dec_l5)
    dec_l5 = LeakyReLU(alpha=0.2)(dec_l5)
    
    dec_l4 = Combine()([dec_l5, noisy_l4_nonlin, clean_l4_nonlin])
    dec_l4 = UpSampling2D(size=2, interpolation="nearest")(dec_l4)
    dec_l4 = Conv2DTranspose(filter_number, filter_size, strides=1, padding="same")(dec_l4)
    dec_l4 = BatchNormalization(axis=chanDim)(dec_l4)
    dec_l4 = LeakyReLU(alpha=0.2)(dec_l4)
    
    dec_l3 = Combine()([dec_l4, noisy_l3_nonlin, clean_l3_nonlin])
    dec_l3 = UpSampling2D(size=2, interpolation="nearest")(dec_l3)
    dec_l3 = Conv2DTranspose(filter_number, filter_size, strides=1, padding="same")(dec_l3)
    dec_l3 = BatchNormalization(axis=chanDim)(dec_l3)
    dec_l3 = LeakyReLU(alpha=0.2)(dec_l3)
    
    dec_l2 = Combine()([dec_l3, noisy_l2_nonlin, clean_l2_nonlin])
    dec_l2 = UpSampling2D(size=2, interpolation="nearest")(dec_l2)
    dec_l2 = Conv2DTranspose(int(filter_number*1.5), filter_size, strides=1, padding="same")(dec_l2)
    dec_l2 = BatchNormalization(axis=chanDim)(dec_l2)
    dec_l2 = LeakyReLU(alpha=0.2)(dec_l2)
    
    dec_l1 = Combine()([dec_l2, noisy_l1_nonlin, clean_l1_nonlin])
    dec_l1 = UpSampling2D(size=2, interpolation="nearest")(dec_l1)
    dec_l1 = Conv2DTranspose(int(filter_number*1.5), filter_size, strides=1, padding="same")(dec_l1)
    dec_l1 = BatchNormalization(axis=chanDim)(dec_l1)
# =============================================================================
#     dec_l1 = LeakyReLU(alpha=0.2)(dec_l1)
# =============================================================================
    outputs = Conv2DTranspose(depth, filter_size, activation="sigmoid", padding="same")(dec_l1)
    
    autoencoder = Model(inputs, outputs, name="autoencoder")
   
    # return a 3-tuple of the encoder, decoder, and autoencoder
    return autoencoder

    
def train_autoencoder(DATA_PATH, MODEL_NAME, DATA_NAME = "visual_obs",
                      EPOCHS = 2,BATCH = 50, NUM_SAMPLES = 100000):
    # construct our convolutional autoencoder
    print("[INFO] building autoencoder...")
    autoencoder = build_autoencoder()
# =============================================================================
#     autoencoder = load_model(MODEL_NAME, custom_objects={'Combine':Combine}, 
#                                              compile=True)    
# =============================================================================
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

    autoencoder.compile(loss="mse", optimizer=opt, metrics=['accuracy'])
# =============================================================================
#     autoencoder.compile(loss="binary_crossentropy", optimizer=opt)
# =============================================================================
    
    # construct training data generator and validation generator
    number_train_samples = int(np.floor(NUM_SAMPLES*0.9))
    number_val_samples = int(np.floor(NUM_SAMPLES*0.1))
    indexes = np.arange(NUM_SAMPLES)
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
    
    autoencoder = load_model(MODEL_NAME, custom_objects={'Combine':Combine}, 
                                             compile=True)    
        
    predict_generator = DataGenerator(range(NUM_SAMPLES), DATA_PATH, DATA_NAME,
                     to_fit=False, batch_size=BATCH, shuffle=False)
    
    layer_name = autoencoder.layers[26].name
    encoder = Model(inputs=autoencoder.input,
                                 outputs=autoencoder.get_layer(layer_name).output)
    encoded = encoder.predict(predict_generator)
    encoded = encoded.reshape((NUM_SAMPLES,2*2*64))
    
    with h5py.File(EMBED_NAME,'w') as f:
           f.create_dataset('embeddings', data=encoded)

def train_hippocampus(DATA_PATH, DATA_NAME = "embeddings", 
                      HMODEL_NAME = "trained_models/hippocampus_ladderv8",
                      BATCH = 100, NUM_SAMPLES = 50000, embed_dim = 300,
                      n_DG = 160, n_CA3 = 80, n_CA1 = 160, EPOCHS=50):
    
    
    hippocampus = Sequential([
        Dense(n_DG, activation='sigmoid', 
# =============================================================================
#               activity_regularizer=regularizers.l1(0.01), 
# =============================================================================
              name='DG', 
              input_shape=(embed_dim,)),
        BatchNormalization(),
# =============================================================================
#         ReLU(),
# =============================================================================
        Dense(n_CA3, activation='sigmoid', 
# =============================================================================
#               activity_regularizer=regularizers.l1(0.01), 
# =============================================================================
              name='CA3'),
        BatchNormalization(),
# =============================================================================
#         ReLU(),
# =============================================================================
        Dense(n_CA1, activation='sigmoid', 
# =============================================================================
#               activity_regularizer=regularizers.l1(0.01), 
# =============================================================================
              name='CA1'),
        BatchNormalization(),
# =============================================================================
#         ReLU(),
# =============================================================================
        Dense(embed_dim, activation='sigmoid', name='EC')
    ])
    
    opt = Adam(lr=1e-4)
    hippocampus.compile(loss="binary_crossentropy", optimizer=opt)
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

class Combine(Layer):
    def __init__(self, **kwargs):
        super(Combine, self).__init__()

    def build(self, input_shape):
        input_shape = input_shape[0][1:]
        self.w_vertical = self.add_weight(
            shape=(input_shape),
            initializer="Ones",
            trainable=True,
            name="vertical"
        )
        self.w_lateral = self.add_weight(
            shape=(input_shape),
            initializer="Zeros",
            trainable=True,
            name="lateral"
        )        
        self.w_product = self.add_weight(
            shape=(input_shape),
            initializer="Zeros",
            trainable=True,
            name="product"
        )        
        self.b = self.add_weight(
            shape=(input_shape), 
            initializer="Zeros", 
            trainable=True,
            name="bias"
        )

    def call(self, inputs):
        vertical = tf.math.multiply(inputs[0], self.w_vertical)
        lateral = tf.math.multiply(inputs[1], self.w_lateral)
        product_inner = tf.math.multiply(inputs[0], inputs[1])
        product = tf.math.multiply(product_inner, self.w_product)
# =============================================================================
#         combined_vert_lat = tf.math.add(vertical, lateral)
#         combined = tf.math.add(combined_vert_lat, product)
# =============================================================================
# =============================================================================
#         combined = tf.math.add_n([self.b, vertical, lateral, product])
# =============================================================================
        combined = self.b+vertical+lateral+product
        square_diff = tf.math.squared_difference(combined,inputs[2])
        mse = tf.math.reduce_mean(square_diff)
        self.add_loss(mse)
        return combined
    
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
    MODEL_NAME = "trained_models/ladderv7.hdf5-50.hdf5"
    DATA_NAME = "visual_obs"
    EMBED_NAME = DATA_PATH+"_ladderv7.hdf5"
    EPOCHS = 50
    BATCH = 25
    NUM_SAMPLES = 50000

# =============================================================================
#     train_autoencoder(DATA_PATH,MODEL_NAME, DATA_NAME, EPOCHS, BATCH, NUM_SAMPLES)
# =============================================================================
    generate_embeddings(DATA_PATH, DATA_NAME,EMBED_NAME, MODEL_NAME, BATCH, NUM_SAMPLES)
# =============================================================================
#     train_hippocampus(DATA_PATH=EMBED_NAME, DATA_NAME = "embeddings", 
#                       HMODEL_NAME = MODEL_NAME+"_hippocampus_sigmoid_bce.h5",
#                       BATCH = 50, NUM_SAMPLES = 10000, embed_dim = 256,
#                       n_DG = 160, n_CA3 = 80, n_CA1 = 160, EPOCHS=100)
# 
# =============================================================================
