from keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D
from keras.models import Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt

class Adri_Autoencoder(object):

    def __init__(self, dims=(84, 84, 3)):
        self.input_img = Input(shape=dims)

        x = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(self.input_img)
        x = MaxPooling3D((2, 2, 2), padding='same')(x)
        x = Conv3D(8, (3, 3, 3), activation='relu', padding='same')(x)
        x = MaxPooling3D((2, 2, 2), padding='same')(x)
        x = Conv3D(8, (3, 3, 3), activation='relu', padding='same')(x)
        self.encoded = MaxPooling3D((2, 2, 2), padding='same')(x)

        x = Conv3D(8, (3, 3, 3), activation='relu', padding='same')(self.encoded)
        x = UpSampling3D((2, 2, 2))(x)
        x = Conv3D(8, (3, 3, 3), activation='relu', padding='same')(x)
        x = UpSampling3D((2, 2, 2))(x)
        x = Conv3D(16, (3, 3, 3), activation='relu')(x)
        x = UpSampling3D((2, 2, 2))(x)
        self.decoded = Conv3D(1, (3, 3, 3), activation='sigmoid', padding='same')(x)

        self.autoencoder = Model(self.input_img, self.decoded)
        self.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

        self.autoencoder.summary()

    def feed_img(self, img):
        prototype = self.autoencoder.evaluate([img])
        self.autoencoder.fit([img], [img], epochs=1, batch_size=None)
        return prototype


class Conv_Autoencoder():
    def __init__(self):
        self.img_rows = 84
        self.img_cols = 84
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        optimizer = Adam(lr=0.001)

        self.autoencoder_model = self.build_model()
        self.autoencoder_model.compile(loss='mse', optimizer=optimizer)
        self.autoencoder_model.summary()

    def build_model(self):
        input_img = Input(shape=self.img_shape)

        x = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling3D((2, 2, 2), padding='same')(x)
        x = Conv3D(8, (3, 3, 3), activation='relu', padding='same')(x)
        x = MaxPooling3D((2, 2, 2), padding='same')(x)
        x = Conv3D(8, (3, 3, 3), activation='relu', padding='same')(x)
        self.encoded = MaxPooling3D((2, 2, 2), padding='same')(x)

        x = Conv3D(8, (3, 3, 3), activation='relu', padding='same')(self.encoded)
        x = UpSampling3D((2, 2, 2))(x)
        x = Conv3D(8, (3, 3, 3), activation='relu', padding='same')(x)
        x = UpSampling3D((2, 2, 2))(x)
        x = Conv3D(16, (3, 3, 3), activation='relu')(x)
        x = UpSampling3D((2, 2, 2))(x)
        self.decoded = Conv3D(1, (3, 3, 3), activation='sigmoid', padding='same')(x)

        return Model(input_img, self.decoded)

    def train_model(self, x_train, y_train, x_val, y_val, epochs, batch_size=20):
        early_stopping = EarlyStopping(monitor='val_loss',
                                       min_delta=0,
                                       patience=5,
                                       verbose=1,
                                       mode='auto')
        history = self.autoencoder_model.fit(x_train, y_train,
                                             batch_size=batch_size,
                                             epochs=epochs,
                                             validation_data=(x_val, y_val),
                                             callbacks=[early_stopping])
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

    def eval_model(self, x_test):
        preds = self.autoencoder_model.predict(x_test)
        return preds