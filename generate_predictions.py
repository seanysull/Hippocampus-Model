import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import h5py
import tensorflow as tf
from numpy.random import default_rng
from denoiser import DataGenerator
from tensorflow.keras.models import Model
from denoiser import kl_divergence_regularizer
DATA_PATH = "data/simulation_data_2708_50000steps_morphed100.h5"
# =============================================================================
# embed_path = "data/simulation_data_2607_10000steps.h5_ladderv5_fulldata.hdf5"
# =============================================================================
DATA_NAME = "visual_obs"
MODEL_NAME = "trained_models/denoiseV9_v4_batchact_reverse.hdf5-14.hdf5"
PIC_NAME = "morph_100"
BATCH = 25
DIM = [256, 256]
CHANNELS = 3
NUM_SAMPLES = 10000

# =============================================================================
# rng = default_rng()
# test_indices = np.sort(rng.choice(NUM_SAMPLES,25,False))
# =============================================================================
indexes = range(0,50000,1000)
# =============================================================================
# predict_generator = DataGenerator(indexes, DATA_PATH, DATA_NAME, 
#                                   to_fit=False, batch_size=50, shuffle=False)
# 
# =============================================================================
# =============================================================================
# with h5py.File(DATA_PATH, "r") as f:
#     vis_data = f[DATA_NAME][image_numbers,:,:,:]
# =============================================================================
with h5py.File(DATA_PATH, "r") as f:
    vis_data = f[DATA_NAME][indexes]
    
# =============================================================================
# with h5py.File(embed_path, 'r') as f:    
#     embeddings = f["embeddings"][image_numbers]
# =============================================================================
# =============================================================================
#     
# autoencoder = tf.keras.models.load_model(MODEL_NAME,
#                                          custom_objects={"kl_divergence_regularizer":kl_divergence_regularizer},
#                                          compile=True)
# 
# =============================================================================
# =============================================================================
# autoencoder = tf.keras.models.load_model(MODEL_NAME, 
#                                          custom_objects={'Combine':Combine}, 
#                                          compile=True)
# =============================================================================
# =============================================================================
# autoencoder.summary()
# =============================================================================
# =============================================================================
# layer_name = autoencoder.layers[22].name
# =============================================================================
# =============================================================================
# layer_name = "latent"
# =============================================================================
# =============================================================================
# encoder = Model(inputs=autoencoder.input,
#                                  outputs=autoencoder.get_layer(layer_name).output)
#    
# =============================================================================
# use the convolutional autoencoder to make predictions on the
# testing images, then initialize our list of output images
print("making predictions...")
# =============================================================================
# tree_generator = DataGenerator(tree_indices, DATA_PATH, DATA_NAME,
#                  to_fit=False, batch_size=5, shuffle=False)
# carpet_generator = DataGenerator(carpet_indices, DATA_PATH, DATA_NAME,
#                  to_fit=False, batch_size=5, shuffle=False)
# temple_generator = DataGenerator(temple_indices, DATA_PATH, DATA_NAME,
#                  to_fit=False, batch_size=5, shuffle=False)
# =============================================================================
# =============================================================================
# encoded = encoder.predict(predict_generator, workers=4, use_multiprocessing=False) 
# =============================================================================
# =============================================================================
# encode = encoded.reshape((1000, 2*2*64))
# =============================================================================
# =============================================================================
# decoded = autoencoder.predict(predict_generator, workers=4, use_multiprocessing=False)
# =============================================================================
# =============================================================================
# tree_code = encoder.predict(tree_generator, workers=4, use_multiprocessing=False) 
# tree_image = autoencoder.predict(tree_generator, workers=4, use_multiprocessing=False)
# carpet_code = encoder.predict(carpet_generator, workers=4, use_multiprocessing=False) 
# carpet_image = autoencoder.predict(carpet_generator, workers=4, use_multiprocessing=False)
# temple_code = encoder.predict(temple_generator, workers=4, use_multiprocessing=False) 
# temple_image = autoencoder.predict(temple_generator, workers=4, use_multiprocessing=False)
# =============================================================================

# norm = []
# for i in range(25):
#     d1 = decoded[i]
#     norm_j = np.ones((256,256,3))
#     for j in range(3):
#         norm_j[:,:,j] = ((d1[:,:,j] - np.min(d1[:,:,j]))/np.ptp(d1[:,:,j]))
# # =============================================================================
# #     mid = np.array(norm_j).reshape(256,256,3)
# # =============================================================================
#     norm.append(norm_j)    
# norm = np.array(norm)
# d1 = decoded[0]
# n1=norm[0]    
# =============================================================================
# norm = (decoded - np.min(d1))/np.ptp(d1)
# =============================================================================
print("done predictions...")
# =============================================================================
# compare = np.hstack([vis_data, decoded])
# =============================================================================
# =============================================================================
# np.save("cae_predictions_5x5", compare)
# compare = np.load("cae_predictions_5x5.npy")
# =============================================================================
plt.axis("off")

for index in range(50):
# =============================================================================
#     fig=plt.imshow(compare[index],interpolation='none')
# =============================================================================
    fig=plt.imshow(vis_data[index],interpolation='none')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig('test_images/'+PIC_NAME+"_"+str(index)+'.png',
        bbox_inches='tight', pad_inches=0, format='png', dpi=300)
    print("image written")

# =============================================================================
# d1=decoded[0,:]
# norm = (d1 - np.min(d1))/np.ptp(d1)
# plt.switch_backend('tkagg')
# plt.imshow(norm)
# plt.savefig('predictions/'+PIC_NAME+'.png',
#          bbox_inches='tight', pad_inches=0, format='png', dpi=300)
# =============================================================================
# =============================================================================
# plt.show()
# #%%    
# =============================================================================
