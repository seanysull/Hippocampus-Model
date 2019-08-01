import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
from scipy import stats

import tensorflow as tf
from tensorflow import keras as keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras import backend as K


def G_rate_map(arena_size=[100, 100], theta=0., phase=[50, 50], lamb=500):

    G = np.zeros(arena_size)
    a = 0.3
    b = -3. / 2.
    lambV = (4 * np.pi) / (np.sqrt(3 * lamb))
    theta = np.radians(theta)

    for ind, val in np.ndenumerate(G):

        tmp_g = 0
        for i in np.deg2rad(np.linspace(-30, 90, 3)):
            u_f = (np.cos(i + theta), np.sin(i + theta))
            dist = (ind[0] - phase[0], ind[1] - phase[1])
            tmp_g += np.cos(lambV * np.dot(u_f, dist))

        tmp_g = np.exp(np.dot(a, (tmp_g) + b)) - 1
        G[ind] = tmp_g

    ## Normalize if for learning and LEC integration effiency
    if G.min() < 0: G += abs(G.min())
    G = (G - G.min()) / (G.max() - G.min())

    return G


def LEC_rate_map(arena_size=[100, 100], filled_perc=0.3):

    a = np.zeros(36)
    a[: int(filled_perc * 25)] = 1
    np.random.shuffle(a)
    a = a.reshape(6, 6)

    b = np.zeros((arena_size[0], arena_size[1]))

    for i in range(arena_size[0]):
        for j in range(arena_size[1]):
            idx1 = i * len(a) // arena_size[0]
            idx2 = j * len(a) // arena_size[1]
            b[i][j] = a[idx1][idx2]

    arena = scipy.ndimage.filters.gaussian_filter(b, 4)
    arena *= 0.6

    return arena


def create_arena(n_grid, n_lec):
    global G_rate_map, LEC_rate_map

    arena_size = [50, 50]

    grid_data = []
    for ii in range(n_grid):
        lamb = np.random.randint(500, 2000)
        phase = np.random.randint(0, arena_size[0], 2)  ## This is assuming arena is a square
        g = G_rate_map(arena_size=arena_size, phase=phase, lamb=lamb)
        grid_data.append(g.flatten())
    grid_data = np.array(grid_data)

    lec_1_data = []
    for ii in range(n_lec):
        l = LEC_rate_map(arena_size=arena_size, filled_perc=0.2)
        lec_1_data.append(l.flatten())
    lec_1_data = np.array(lec_1_data)

    # Make data structure combining both MEC and LEC_1
    data = np.vstack((grid_data, lec_1_data))
    data = data.T

    return data


def modify_arena(data, data2, dd, n_grid, n_lec):
    global create_arena

    #data2 = create_arena(n_grid, n_lec)
    data2[:, :n_grid] = data[:, :n_grid]

    new_data = np.zeros_like(data)

    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)

    new_data[idx[int(idx.size * dd):]] = data[idx[int(idx.size * dd):]]
    new_data[idx[:int(idx.size * dd)]] = data2[idx[:int(idx.size * dd)]]

    return new_data


def custom_softmax_DG(x):
    beta = 1.
    xx = beta*x
    e = K.exp(xx-K.max(xx, axis=-1, keepdims=True))
    s = K.sum(e, axis=-1, keepdims=True)
    return e / s

get_custom_objects().update({'custom_softmax_DG': keras.layers.Activation(custom_softmax_DG)})


def custom_softmax_CA3(x):
    beta = 1.
    xx = beta*x
    e = K.exp(xx-K.max(xx, axis=-1, keepdims=True))
    s = K.sum(e, axis=-1, keepdims=True)
    return e / s

get_custom_objects().update({'custom_softmax_CA3': keras.layers.Activation(custom_softmax_CA3)})


def create_model(n_DG, n_CA3, n_CA1, dim):
    global custom_softmax_DG, custom_softmax_CA3

    model = keras.Sequential([
        keras.layers.Dense(n_DG, activation=tf.nn.relu, input_shape=(dim,)),
        #keras.layers.Activation(custom_softmax_DG),
        keras.layers.Dense(n_CA3, activation=tf.nn.relu),
        keras.layers.Activation(custom_softmax_CA3),
        keras.layers.Dense(n_CA1, activation=tf.nn.relu),
        keras.layers.Dense(dim, activation=tf.nn.relu)
    ])

    optimizer = tf.train.RMSPropOptimizer(0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])

    return model


## Callback function is deactivated
def fit(model, data, save_weights=True):
    #f_cb = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2,
                                         #verbose=0, mode='auto', baseline=None)  # min_delta=0.01, patience=10

    EPOCHS = 1000
    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)

    history = model.fit(data[idx], data[idx], epochs=EPOCHS, validation_split=0.2, verbose=0)#, callbacks=[f_cb])

    if save_weights:
        model.save_weights('model.h5')

    return history

