# import the necessary packages
import numpy as np
from src.data.preprocessing import learn_dictionary
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
import tensorflow as tf

sess = tf.Session()
import keras

from keras import backend as K

K.set_session(sess)

import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import LeakyReLU,Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape,BatchNormalization, regularizers
from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt



class Deep_SVDD:
    ## Initialise static variables
    INPUT_DIM = 0
    HIDDEN_SIZE = 0
    DATASET = "MNIST"

    def __init__(self, dataset, x_trainForWtInit, inputdim, hiddenLayerSize, img_hgt, img_wdt,img_channel, modelSavePath, reportSavePath,
                 preTrainedWtPath, intValue=0, stringParam="defaultValue",
                 otherParam=None):
        """
        Called when initializing the classifier
        """
        Deep_SVDD.DATASET = dataset
        Deep_SVDD.INPUT_DIM = inputdim
        Deep_SVDD.HIDDEN_SIZE = hiddenLayerSize
        self.intValue = intValue
        self.stringParam = stringParam

        # THIS IS WRONG! Parameters should have same name as attributes
        self.differentParam = otherParam

        self.directory = modelSavePath
        self.results = reportSavePath
        self.pretrainedWts = preTrainedWtPath
        self.model = ""
        self.IMG_HGT = img_hgt
        self.IMG_WDT = img_wdt
        self.channel = img_channel
        self.h_size = Deep_SVDD.HIDDEN_SIZE
        global model
        self.r = 1.0
        self.kvar = 0.0

        self._X_train = x_trainForWtInit
        self.cae = self.build_autoencoder()



    def build_autoencoder(self):

        autoencoder = Sequential()


        def kernelConv2D_custom_init(shape, dtype=None):
            W1_init = learn_dictionary(self._X_train, 8, 5, n_sample=500)
            print("W1_init Shape .....",W1_init.shape)
            # W1_init = np.reshape(W1_init,(8,5,5,1))
            print("W1_init After reShape .....", W1_init.shape)
            return W1_init

        input_img = Input(shape=(28, 28, 1))

        x = Conv2D(32, (3, 3), padding='same', use_bias=True)(input_img)
        x = BatchNormalization(axis=-1)(x)
        x =  LeakyReLU() (x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(16, (3, 3),  padding='same', use_bias=True)(x)
        x = BatchNormalization(axis=-1)(x)
        x = LeakyReLU()(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(16, (3, 3),  padding='same', use_bias=True)(x)
        x = BatchNormalization(axis=-1)(x)
        x = LeakyReLU()(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3),  padding='same', use_bias=True)(x)
        x = BatchNormalization(axis=-1)(x)
        x = LeakyReLU()(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

        encoded = Flatten()(x)

        # Reshape the flattened layer
        x = Reshape((2, 2, 8))(encoded)

        x = Conv2D(8, (3, 3), padding='same', use_bias=True)(x)
        x = BatchNormalization(axis=-1)(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), padding='same', use_bias=True)(x)
        x = BatchNormalization(axis=-1)(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), padding='same', use_bias=True)(x)
        x = BatchNormalization(axis=-1)(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), padding='valid', use_bias=True)(x)
        x = BatchNormalization(axis=-1)(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same', use_bias=True)(x)
        #
        autoencoder = Model(input_img, decoded)

        # print("[INFO:] Autoencoder built with architecture", autoencoder.summary())
        print("[INFO:] Building Autoencoder Complete .....")

        return autoencoder

    def plot_train_history_loss(self,history):
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.savefig(self.results+"cae_")


    def pretrain_autoencoder(self,cae_loss):
        #initialize autoencoder
        autoencoder = self.cae

        if(cae_loss=="ce"):
            self.cae.compile(optimizer='adam', loss='binary_crossentropy')
        elif (cae_loss == "l2"):
            self.cae.compile(optimizer='adam', loss='mean_squared_error')
        elif (cae_loss == "rcae_loss"):
            self.cae.compile(optimizer='adam', loss='mean_squared_error')

            fpath = self.results+"weights-dcae-{epoch:02d}-{val_loss:.3f}.hdf5"
            callbacks = [ModelCheckpoint(fpath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')]
            history = autoencoder.fit(x_train, x_train,
                                      epochs=150,
                                      batch_size=200,
                                      shuffle=True,
                                      validation_data=(x_test, x_test),
                                      callbacks=callbacks)
            self.plot_train_history_loss(history)
        else:
            print("[INFO:] No valid loss function specified ....")



    def fit(self):

        return


    def predict(self):

        return


