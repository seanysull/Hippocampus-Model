import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
from scipy import stats

import keras
from keras.models import Model, load_model
from keras import backend as K
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam



class Arena(object):
    
    def __init__(self, arena_size=[100,100], n_mec=40, n_lec=40):
        
        self.dims = arena_size
        self.n_mec = n_mec
        self.n_lec = n_lec
        self.rateMaps = self.create_rateMaps()
        
        
    def create_rateMaps(self):

        mec_rateMap = []
        for ii in range(self.n_mec):
            lamb = np.random.randint(500, 2000)
            phase = np.random.randint(0, self.dims[0], 2)
            m = self.MEC_rateMap(phase=phase, lamb=lamb)
            mec_rateMap.append(m.flatten())
        mec_rateMap = np.array(mec_rateMap)

        lec_rateMap = []
        for ii in range(self.n_lec):
            l = self.LEC_rateMap(filled_perc=0.2)
            lec_rateMap.append(l.flatten())
        lec_rateMap = np.array(lec_rateMap)

        rateMaps = np.vstack((mec_rateMap, lec_rateMap)).T

        return rateMaps
        
        
    def MEC_rateMap(self, theta=0., phase=[50, 50], lamb=500):

        M = np.zeros(self.dims)
        a = 0.3
        b = -3. / 2.
        lambV = (4 * np.pi) / (np.sqrt(3 * lamb))
        theta = np.radians(theta)

        for ind, val in np.ndenumerate(M):

            tmp_m = 0
            for i in np.deg2rad(np.linspace(-30, 90, 3)):
                u_f = (np.cos(i + theta), np.sin(i + theta))
                dist = (ind[0] - phase[0], ind[1] - phase[1])
                tmp_m += np.cos(lambV * np.dot(u_f, dist))

            tmp_m = np.exp(np.dot(a, (tmp_m) + b)) - 1
            M[ind] = tmp_m

        if M.min() < 0: 
            M += abs(M.min())
            
        M = (M - M.min()) / (M.max() - M.min())

        return M


    def LEC_rateMap(self, filled_perc=0.3):

        a = np.zeros(36)
        a[: int(filled_perc * 25)] = 1
        np.random.shuffle(a)
        a = a.reshape(6, 6)

        b = np.zeros(self.dims)

        for i in range(self.dims[0]):
            for j in range(self.dims[1]):
                idx1 = i * len(a) // self.dims[0]
                idx2 = j * len(a) // self.dims[1]
                b[i][j] = a[idx1][idx2]

        arena = scipy.ndimage.filters.gaussian_filter(b, 4)
        arena *= 0.6

        return arena

    
    # dd goes from 0 to 1 and it's the extent by which the current LEC rate maps are substituted by new ones.
    def modify_LEC_maps(self, dd, permanent=True):
        
        rateMaps2 = self.create_rateMaps()
        rateMaps2[:, :self.n_mec] = self.rateMaps[:, :self.n_mec]

        new_rateMaps = np.zeros_like(self.rateMaps)

        idx = np.arange(self.rateMaps.shape[0])
        np.random.shuffle(idx)

        new_rateMaps[idx[int(idx.size * dd):]] = self.rateMaps[idx[int(idx.size * dd):]]
        new_rateMaps[idx[:int(idx.size * dd)]] = rateMaps2[idx[:int(idx.size * dd)]]
        
        if permanent:
            self.rateMaps = new_rateMaps
        
        return new_rateMaps
    
    
    def get_rateMaps(self):
        
        return self.rateMaps
    
    
    def plot_rateMaps(self):
        
        print('MEC maps')
        plt.figure(figsize=(15,15))
        for i, cell_n in enumerate(np.random.randint(0, self.n_mec, int(4*4))):
            plt.subplot(4,4,i+1)
            plt.imshow(self.rateMaps[:,cell_n].reshape(self.dims))
            plt.axis('off')
        plt.show()
            
        print('')
        print('LEC maps')
        plt.figure(figsize=(15,15))
        for i, cell_n in enumerate(np.random.randint(self.n_mec, self.n_mec+self.n_lec, int(4*4))):
            plt.subplot(4,4,i+1)
            plt.imshow(self.rateMaps[:,cell_n].reshape(self.dims))
            plt.axis('off')
        plt.show()


    
class HPC(object):
    
    def __init__(self, n_DG, n_CA3, n_CA1, dim):
        
        self.n_DG = n_DG
        self.n_CA3 = n_CA3
        self.n_CA1 = n_CA1
        self.dim = dim
        
        self.model = self.create_model()

        
    def create_model(self):

        model = Sequential([
            Dense(self.n_DG, activation='relu', name='DG', input_shape=(self.dim,)),
            Dense(self.n_CA3, activation='relu', name='CA3'),
            Dense(self.n_CA1, activation='relu', name='CA1'),
            Dense(self.dim, activation='relu', name='EC')
        ])

        optimizer = Adam(lr=0.001)
        model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
        
        model.summary()

        return model
    
    
    def save_model(self, name='model'):
        
        self.model.save(name+'.h5')
    
    
    def load_model(self, name='model'):
        
        self.model = load_model(name+'.h5')

        
    def train(self, data, epochs=100):

        idx = np.arange(data.shape[0])
        np.random.shuffle(idx)

        history = self.model.fit(data[idx], data[idx], epochs=epochs, validation_split=0.2, verbose=0)

        return history
    
    
    def test(self, data):
        
        return self.model.predict(data)
    
    
    def get_output(self, layer, data):
        
        model = Model(inputs=self.model.inputs, outputs=self.model.get_layer(layer).output)
        return model.predict(data)
