# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 13:17:34 2020

@author: seano
"""

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
import h5py
from conv_autoencoder import DataGenerator

width=256
height=256
depth=3
inputShape = (height, width, depth)
chanDim = -1
filter_size = 3
latentDim = 512
DATA_PATH = "data/simulation_data_0105_100000steps.h5"
MODEL_NAME = "trained_models/ladderv8_alldata.h5"
DATA_NAME = "visual_obs"
EMBED_NAME = DATA_PATH+"_ladderv8_alldata.h5"
EPOCHS = 15
BATCH = 25
NUM_SAMPLES = 1000
# initialize the input shape to be "channels last" along with
# the channels dimension itself
class Combine(tf.keras.layers.Layer):
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
        return self.b + vertical + lateral + product
 
    def get_config(self):
        # Implement get_config to enable serialization. This is optional.
        base_config = super(Combine, self).get_config()

        return base_config

if __name__ == '__main__':
    inputShape = (height, width, depth)
    chanDim = -1
       
    # define the input to the encoder
    inputs = Input(shape=inputShape)
    x = GaussianNoise(0.2)(inputs)
       
    x = BatchNormalization(axis=chanDim)(x)
    x = LeakyReLU(alpha=0.2)(x)
    layer_1 = Conv2D(64, filter_size, strides=2, padding="same")(x)
    
    
    x = BatchNormalization(axis=chanDim)(layer_1)
    x = LeakyReLU(alpha=0.2)(x)
    layer_2 = Conv2D(64, filter_size, strides=2, padding="same")(x)
    
    x = BatchNormalization(axis=chanDim)(layer_2)
    x = LeakyReLU(alpha=0.2)(x)
    layer_3 = Conv2D(64, filter_size, strides=2, padding="same")(x)
    
    
    x = BatchNormalization(axis=chanDim)(layer_3)
    x = LeakyReLU(alpha=0.2)(x)
    layer_4 = Conv2D(64, filter_size, strides=2, padding="same")(x)
    
    
    x = BatchNormalization(axis=chanDim)(layer_4)
    x = LeakyReLU(alpha=0.2)(x)
    layer_5 = Conv2D(64, filter_size, strides=2, padding="same")(x)
    
    
    x = BatchNormalization(axis=chanDim)(layer_5)
    x = LeakyReLU(alpha=0.2)(x)
    layer_6 = Conv2D(64, filter_size, strides=2, padding="same")(x)
    
    x = BatchNormalization(axis=chanDim)(layer_6)
    x = LeakyReLU(alpha=0.2)(x)
    layer_7 = Conv2D(64, filter_size, strides=2, padding="same")(x)
    
    latent = Flatten()(layer_7)
    # flatten the network and then construct our latent vector
    volumeSize = K.int_shape(layer_7)
       
    x = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(latent)

    dec_layer_7 = Combine()([x, layer_7])
    dec_layer_7 = UpSampling2D(size=2, interpolation="nearest")(dec_layer_7)
    x = Conv2DTranspose(64, filter_size, strides=1, padding="same")(dec_layer_7)
    x = BatchNormalization(axis=chanDim)(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    dec_layer_6 = Combine()([x, layer_6])
    dec_layer_6 = UpSampling2D(size=2, interpolation="nearest")(dec_layer_6)
    x = Conv2DTranspose(64, filter_size, strides=1, padding="same")(dec_layer_6)
    x = BatchNormalization(axis=chanDim)(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    dec_layer_5 = Combine()([x, layer_5])
    dec_layer_5 = UpSampling2D(size=2, interpolation="nearest")(dec_layer_5)
    x = Conv2DTranspose(64, filter_size, strides=1, padding="same")(dec_layer_5)
    x = BatchNormalization(axis=chanDim)(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    dec_layer_4 = Combine()([x, layer_4])
    dec_layer_4 = UpSampling2D(size=2, interpolation="nearest")(dec_layer_4)
    x = Conv2DTranspose(64, filter_size, strides=1, padding="same")(dec_layer_4)
    x = BatchNormalization(axis=chanDim)(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    dec_layer_3 = Combine()([x, layer_3])
    dec_layer_3 = UpSampling2D(size=2, interpolation="nearest")(dec_layer_3)
    x = Conv2DTranspose(64, filter_size, strides=1, padding="same")(dec_layer_3)
    x = BatchNormalization(axis=chanDim)(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    dec_layer_2 = Combine()([x, layer_2])
    dec_layer_2 = UpSampling2D(size=2, interpolation="nearest")(dec_layer_2)
    x = Conv2DTranspose(64, filter_size, strides=1, padding="same")(dec_layer_2)
    x = BatchNormalization(axis=chanDim)(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    dec_layer_1 = Combine()([x, layer_1])
    dec_layer_1 = UpSampling2D(size=2, interpolation="nearest")(dec_layer_1)
    x = Conv2DTranspose(64, filter_size, strides=1, padding="same")(dec_layer_1)
    x = BatchNormalization(axis=chanDim)(x)
    x = LeakyReLU(alpha=0.2)(x)
    outputs = Conv2DTranspose(depth, filter_size, activation="sigmoid", padding="same")(x)
    
    autoencoder = Model(inputs, outputs, name="autoencoder")
    autoencoder.summary()   
 
    opt = Adam(lr=1e-2)
    
    autoencoder.compile(loss="mse", optimizer=opt)
    
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
    
    # train the convolutional autoencoder
    H = autoencoder.fit(train_generator,
        epochs=EPOCHS, validation_data = val_generator,
        workers=4, use_multiprocessing=False)
    
    autoencoder.save(MODEL_NAME, save_format= "h5")
    # =============================================================================
    # encoder = tf.keras.models.load_model(MODEL_NAME, 
    #                                          custom_objects={'Combine':Combine}, 
    #                                          compile=True)    
    # =============================================================================layer_name = autoencoder.layers[23].name
    encoder = Model(inputs=autoencoder.input,
                                 outputs=autoencoder.get_layer(layer_name).output)