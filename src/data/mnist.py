from src.data.base import DataLoader
from src.data.preprocessing import center_data, normalize_data, rescale_to_unit_interval, \
    global_contrast_normalization, zca_whitening, extract_norm_and_out, learn_dictionary, pca
# from src.data.modules import addConvModule
# from src.utils.visualization.mosaic_plot import plot_mosaic
from src.utils.misc import flush_last_line
from src.config import Configuration as Cfg

import gzip
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Activation,LeakyReLU,Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape,BatchNormalization, regularizers
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from keras import backend as K
from keras.optimizers import SGD,Adam
from sklearn.metrics import average_precision_score,mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from skimage import io
from numpy import linalg as LA

from keras.optimizers import RMSprop


from keras.callbacks import Callback





class RcaeParamSaver(Callback):
    def __init__(self, N,ae):
        self.N = N
        self.batch = 0
        self.ae = ae

    def on_batch_end(self, batch, logs={}):
        if self.batch % self.N == 0:
            print("Inside batch")
            # name = 'weights%08d.h5' % self.batch
            # self.model.save_weights(name)

            U = self.ae.layers[9].get_weights()
            self.ae.latent_weights[0] = U[0]

            V = self.ae.layers[11].get_weights()
            V = V[0]
            self.ae.latent_weights[1] = np.transpose(V)
            self.ae.latent_weights[2] = self.Noise
        self.batch += 1




class MNIST_DataLoader(DataLoader):

    mean_square_error_dict= {}
    def __init__(self):

        DataLoader.__init__(self)

        self.dataset_name = "mnist"
        self.n_train = 50000
        self.n_val = 10000
        self.n_test = 10000

        self.seed = Cfg.seed

        if Cfg.ad_experiment:
            self.n_classes = 2
        else:
            self.n_classes = 10

        Cfg.n_batches = int(np.ceil(self.n_train * 1. / Cfg.batch_size))

        self.data_path = "/content/drive/My Drive/2018/Colab_Deep_Learning/one_class_neural_networks//data/mnist/"

        self.on_memory = True
        Cfg.store_on_gpu = True
        # print("Inside the MNIST_DataLoader RCAE.RESULT_PATH:", RCAE_AD.RESULT_PATH)
        self.rcae_results = "/content/drive/My Drive/2018/Colab_Deep_Learning/one_class_neural_networks//reports/figures/MNIST/RCAE/"
        self.modelsave_path = "/content/drive/My Drive/2018/Colab_Deep_Learning/one_class_neural_networks/models/MNIST/RCAE/"

        print("Inside the MNIST_DataLoader RCAE.RESULT_PATH:", self.rcae_results)


        # load data from disk
        self.load_data()

        ## Rcae parameters
        self.mue = 0.1
        self.lamda = [0.01]
        self.Noise = np.zeros(len(self._X_train))
        self.anomaly_threshold= 0.0
        self.cae = self.build_autoencoder()
        self.latent_weights = [0,0,0]
        self.batchNo=0
        self.index = 0
        # Pretrain the DCAE autoencoder
        # self.pretrain_autoencoder()


    def flush_last_line(to_flush=1):
        import sys
        for _ in range(to_flush):
            sys.stdout.write("\033[F")  # back to previous line
            sys.stdout.write("\033[K")  # clear line
            sys.stdout.flush()


    def check_specific(self):

        # store primal variables on RAM
        assert Cfg.store_on_gpu

    def load_data(self, original_scale=False,):

        print("[INFO: ] Loading data...")



        X = load_mnist_images('%strain-images-idx3-ubyte.gz' %
                              self.data_path)
        y = load_mnist_labels('%strain-labels-idx1-ubyte.gz' %
                              self.data_path)
        X_test = load_mnist_images('%st10k-images-idx3-ubyte.gz' %
                                   self.data_path)
        y_test = load_mnist_labels('%st10k-labels-idx1-ubyte.gz' %
                                   self.data_path)

        if Cfg.ad_experiment:

            # set normal and anomalous class
            normal = []
            outliers = []

            if Cfg.mnist_normal == -1:
                normal = list(range(0, 10))
                normal.remove(Cfg.mnist_outlier)
            else:
                normal.append(Cfg.mnist_normal)

            if Cfg.mnist_outlier == -1:
                outliers = list(range(0, 10))
                outliers.remove(Cfg.mnist_normal)
            else:
                outliers.append(Cfg.mnist_outlier)
                print("[INFO:] The  label  of outlier  points are ", Cfg.mnist_outlier)
                print("[INFO:] The  number of outlier  points are ", len(outliers))
            
            print("[INFO:] The  label  of normal points are ", Cfg.mnist_normal)
            # extract normal and anomalous class
            
            X_norm, X_out, y_norm, y_out = extract_norm_and_out(X,y, normal=normal, outlier=outliers)

            # reduce outliers to fraction defined
            n_norm = len(y_norm)
            n_out = int(np.ceil(Cfg.out_frac * n_norm / (1 - Cfg.out_frac)))
            #
            # print("[INFO:] The number of normal data points are ", (n_norm))
            # print("[INFO:] The number of outlier data points are ", (n_out))


            # shuffle to obtain random validation splits
            print("[INFO:] Random Seed used is  ", Cfg.seed)
            np.random.seed(self.seed)
            perm_norm = np.random.permutation(len(y_norm))
            perm_out = np.random.permutation(len(y_out))

            # split into training and validation set
            n_norm_split = int(Cfg.mnist_val_frac * n_norm)
            n_out_split = int(Cfg.mnist_val_frac * n_out)


            X_norm_Training = X_norm[perm_norm[n_norm_split:]]
            X_out_Training = X_out[perm_out[:n_out][n_out_split:]]

            # print("[INFO:] The shape of Normal used in training+validation ", X_norm_Training.shape)
            # print("[INFO:] The shape of Outlier used in training+validation ", X_out_Training.shape)


            self._X_train = np.concatenate((X_norm[perm_norm[n_norm_split:]],
                                            X_out[perm_out[:n_out][n_out_split:]]))
            self._y_train = np.append(y_norm[perm_norm[n_norm_split:]],
                                      y_out[perm_out[:n_out][n_out_split:]])
            self._X_val = np.concatenate((X_norm[perm_norm[:n_norm_split]],
                                          X_out[perm_out[:n_out][:n_out_split]]))
            self._y_val = np.append(y_norm[perm_norm[:n_norm_split]],
                                    y_out[perm_out[:n_out][:n_out_split]])

            # print("[INFO:] The shape of Data used in [ Training  ] ", self._X_train.shape)
            # print("[INFO:] The shape of Data used in [ Validation ] ", self._X_val.shape)

            # shuffle data (since batches are extracted block-wise)
            self.n_train = len(self._y_train)
            self.n_val = len(self._y_val)
            perm_train = np.random.permutation(self.n_train)
            perm_val = np.random.permutation(self.n_val)
            self._X_train = self._X_train[perm_train]
            self._y_train = self._y_train[perm_train]
            self._X_val = self._X_train[perm_val]
            self._y_val = self._y_train[perm_val]


            # Subset train set such that we only get batches of the same size
            self.n_train = (self.n_train / Cfg.batch_size) * Cfg.batch_size
            subset = np.random.choice(len(self._X_train), int(self.n_train), replace=False)
            self._X_train = self._X_train[subset]
            self._y_train = self._y_train[subset]


            # Adjust number of batches
            Cfg.n_batches = int(np.ceil(self.n_train * 1. / Cfg.batch_size))

            # test set
            X_norm, X_out, y_norm, y_out = extract_norm_and_out(X_test, y_test, normal=normal, outlier=outliers)
            self._X_test = np.concatenate((X_norm, X_out))
            self._y_test = np.append(y_norm, y_out)
            perm_test = np.random.permutation(len(self._y_test))
            self._X_test = self._X_test[perm_test]
            self._y_test = self._y_test[perm_test]
            self.n_test = len(self._y_test)
            #
            # print("[INFO:] The shape of  Normal instances used in Testing ", X_norm.shape)
            # print("[INFO:] The shape of  Outlier instances used in Testing ", X_out.shape)
            # print("========================================================================")



        else:
            # split into training, validation, and test sets
            np.random.seed(self.seed)
            perm = np.random.permutation(len(X))

            self._X_train = X[perm[self.n_val:]]
            self._y_train = y[perm[self.n_val:]]
            self._X_val = X[perm[:self.n_val]]
            self._y_val = y[perm[:self.n_val]]
            self._X_test = X_test
            self._y_test = y_test

        # normalize data (if original scale should not be preserved)
        if not original_scale:

            # simple rescaling to [0,1]
            normalize_data(self._X_train, self._X_val, self._X_test, scale=np.float32(255))

            # global contrast normalization
            if Cfg.gcn:
                global_contrast_normalization(self._X_train, self._X_val, self._X_test, scale=Cfg.unit_norm_used)

            # ZCA whitening
            if Cfg.zca_whitening:
                self._X_train, self._X_val, self._X_test = zca_whitening(self._X_train, self._X_val, self._X_test)

            # rescale to [0,1] (w.r.t. min and max in train data)
            rescale_to_unit_interval(self._X_train, self._X_val, self._X_test)

            # PCA
            if Cfg.pca:
                self._X_train, self._X_val, self._X_test = pca(self._X_train, self._X_val, self._X_test, 0.95)


        print("[INFO: ] Data loaded.")


    def custom_rcae_loss(self):


        U = self.cae.layers[9].get_weights()
        U = U[0]

        V = self.cae.layers[11].get_weights()
        V = V[0]
        V = np.transpose(V)
        N = self.Noise
        lambda_val = self.lamda[0]
        mue = self.mue
        batch_size = 128
        # batch_size = 128
        # for index in range(0, N.shape[0], batch_size):
        #     batch = N[index:min(index + batch_size, N.shape[0]), :]
        # N_reshaped = N_reshaped[self.index:min(self.index + K.int_shape(y_true)[0], N_reshaped.shape[0]), :]
        # print("[INFO:] dynamic shape of batch is ", )
        # if(N.ndim >1):
        #
        #     N_reshaped = np.reshape(N,(len(N),28,28,1))
        #     symbolic_shape = K.shape(y_pred)
        #     noise_shape = [symbolic_shape[axis] if shape is None else shape
        #                    for axis, shape in enumerate(N_reshaped)]
        #     N_reshaped= N_reshaped[0:noise_shape[1]]
        #     term1 = keras.losses.mean_squared_error(y_true, (y_pred+N_reshaped ))
        #
        # else:
        #     term1 = keras.losses.mean_squared_error(y_true, (y_pred))

        def custom_rcae(y_true, y_pred):

            term1 = keras.losses.mean_squared_error(y_true, (y_pred))
            term2 = mue * 0.5 *  (LA.norm(U) +  LA.norm(V))
            term3 = lambda_val * 0.5 *  LA.norm(N)

            return (term1 + term2 + term3)

        return custom_rcae


    def encoder(self,input_img):
        # encoder
        # input = 28 x 28 x 1 (wide and thin)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)  # 28 x 28 x 32
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # 14 x 14 x 32
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)  # 14 x 14 x 64
        conv2 = BatchNormalization()(conv2)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # 7 x 7 x 64
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)  # 7 x 7 x 128 (small and thick)
        conv3 = BatchNormalization()(conv3)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        conv3 = BatchNormalization()(conv3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)  # 7 x 7 x 256 (small and thick)
        conv4 = BatchNormalization()(conv4)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        conv4 = BatchNormalization()(conv4)
        return conv4

    def decoder(self,conv4):
        # decoder
        conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)  # 7 x 7 x 128
        conv5 = BatchNormalization()(conv5)
        conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
        conv5 = BatchNormalization()(conv5)
        conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5)  # 7 x 7 x 64
        conv6 = BatchNormalization()(conv6)
        conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)
        conv6 = BatchNormalization()(conv6)
        up1 = UpSampling2D((2, 2))(conv6)  # 14 x 14 x 64
        conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1)  # 14 x 14 x 32
        conv7 = BatchNormalization()(conv7)
        conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)
        conv7 = BatchNormalization()(conv7)
        up2 = UpSampling2D((2, 2))(conv7)  # 28 x 28 x 32
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2)  # 28 x 28 x 1
        return decoded


    # def build_autoencoder(self):
    #     input_img = Input(shape=(28, 28, 1))
    #
    #     autoencoder = Model(input_img, self.decoder(self.encoder(input_img)))
    #     # autoencoder.compile(loss='mean_squared_error', optimizer=RMSprop())
    #     U = 0.0
    #     V = 0.0
    #     N = 0.0
    #     adam = keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-4, amsgrad=False)
    #     autoencoder.compile(optimizer=adam,
    #                                  loss=self.custom_rcae_loss())
    #
    #     self.cae = autoencoder
    #     return autoencoder


    # def build_autoencoder(self):
    #     input_img = Input(shape=(28, 28, 1))
    #     # encoder
    #     # input = 28 x 28 x 1 (wide and thin)
    #     conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)  # 28 x 28 x 32
    #     pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # 14 x 14 x 32
    #     conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)  # 14 x 14 x 64
    #     pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # 7 x 7 x 64
    #     conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)  # 7 x 7 x 128 (small and thick)
    #
    #     # decoder
    #     conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)  # 7 x 7 x 128
    #     up1 = UpSampling2D((2, 2))(conv4)  # 14 x 14 x 128
    #     conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)  # 14 x 14 x 64
    #     up2 = UpSampling2D((2, 2))(conv5)  # 28 x 28 x 64
    #     decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2)  # 28 x 28 x 1
    #
    #     autoencoder = Model(input_img, decoded)
    #     U = 0.0
    #     V = 0.0
    #     N = 0.0
    #     # autoencoder.compile(loss='mean_squared_error', optimizer='adam')
    #     adam = keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    #
    #     autoencoder.compile(optimizer=adam,
    #                          loss=MNIST_DataLoader.custom_rcae_loss())
    #     self.cae = autoencoder
    #     return autoencoder
    #

    # #
    # def build_autoencoder(self):
    #
    #
    #     def kernelConv2D_custom_init(shape, dtype=None):
    #         print("INFO: Shape of the self._X_train ",self._X_train.shape)
    #         W1_init = learn_dictionary(self._X_train, 8, 5, n_sample=500)
    #         print("W1_init Shape .....",W1_init.shape)
    #         kernel_size = (5, 5)
    #         input_channels = 1
    #         num_filters = 8
    #         kernel_shape = kernel_size + (input_channels, num_filters)
    #         W1_init = np.reshape(W1_init,kernel_shape)
    #
    #         print("W1_init After reShape .....", W1_init.shape)
    #         return W1_init
    #
    #     input_img = Input(shape=(28, 28, 1))
    #
    #     autoencoder = Sequential()
    #     # Encoder Layers with linear activations
    #     autoencoder.add(Conv2D(8, (5, 5), kernel_initializer=kernelConv2D_custom_init, use_bias=True ,padding='same', input_shape=(28,28,1)))
    #     autoencoder.add(LeakyReLU(alpha=0.1))
    #     autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    #     autoencoder.add(BatchNormalization())
    #
    #     autoencoder.add(Conv2D(4, (5, 5), use_bias=True, padding='same'))
    #     autoencoder.add(LeakyReLU(alpha=0.1))
    #     autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    #     autoencoder.add(BatchNormalization())
    #
    #     autoencoder.add(Flatten())
    #     #Code layer
    #
    #     autoencoder.add(Dense(units=32))
    #     autoencoder.add(LeakyReLU(alpha=0.1))
    #     autoencoder.add(BatchNormalization())
    #
    #
    #     autoencoder.add(Dense(units=196))
    #     autoencoder.add(LeakyReLU(alpha=0.1))
    #     autoencoder.add(BatchNormalization())
    #
    #     autoencoder.add(Reshape((7, 7, 4)))
    #     autoencoder.add(LeakyReLU(alpha=0.1))
    #
    #
    #
    #     # Upscale
    #     UpSampling2D((2, 2))
    #
    #     autoencoder.add(Conv2D(4, (5, 5),  use_bias=True, padding='same'))
    #     autoencoder.add(LeakyReLU(alpha=0.1))
    #     autoencoder.add(UpSampling2D((2, 2)))
    #     autoencoder.add(BatchNormalization())
    #
    #
    #
    #     autoencoder.add(Conv2D(8, (5, 5),  use_bias=True, padding='same'))
    #     autoencoder.add(LeakyReLU(alpha=0.1))
    #     autoencoder.add(UpSampling2D((2, 2)))
    #     autoencoder.add(BatchNormalization())
    #
    #     # reconstruction
    #     autoencoder.add(Conv2D(1, (5, 5),  use_bias=True, padding='same'))
    #     autoencoder.add(Activation('sigmoid'))
    #
    #     print("[INFO:] Autoencoder built with architecture", autoencoder.summary())
    #
    #     # Assign the Noise to be zero
    #     U = 0.0
    #     V = 0.0
    #     N = 0.0
    #     # print("[INFO:] Autoencoder built with architecture", autoencoder.summary())
    #     print("[INFO:] Building Autoencoder Complete .....")
    #     print("[INFO:] Adding robust cae loss.... ")
    #     # print("[INFO:] Shape of U ",U[0].shape)
    #     # print("[INFO:] Shape of V ", V[0].shape)
    #     # autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    #     autoencoder.compile(optimizer='adam', loss=MNIST_DataLoader.custom_rcae_loss(self, self.mue, self.lamda, U, V,N))
    #
    #     self.fpath = self.rcae_results+"weights-rcae-{epoch:02d}-{val_loss:.3f}.hdf5"
    #     self.callbacks = [ModelCheckpoint(self.fpath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')]
    #     self.cae = autoencoder
    #
    #     return autoencoder

    # Build lenet style autoencoder
    def build_autoencoder(self):

        # initialize the model
        autoencoder = Sequential()
        inputShape = (28, 28, 1)
        chanDim = -1 # since depth is appearing the end
        # first set of CONV => RELU => POOL layers
        autoencoder.add(Conv2D(20, (5, 5), padding="same",input_shape=inputShape))
        autoencoder.add(Activation("relu"))
        autoencoder.add(BatchNormalization(axis=chanDim))
        autoencoder.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # second set of CONV => RELU => POOL layers
        autoencoder.add(Conv2D(50, (5, 5), padding="same"))
        autoencoder.add(Activation("relu"))
        autoencoder.add(BatchNormalization(axis=chanDim))
        autoencoder.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # first (and only) set of FC => RELU layers
        autoencoder.add(Flatten())

        autoencoder.add(Dense(2450))
        autoencoder.add(Activation("relu"))
        autoencoder.add(BatchNormalization(axis=chanDim))

        autoencoder.add(Dense(32))
        autoencoder.add(Activation("relu"))
        autoencoder.add(BatchNormalization(axis=chanDim))



        autoencoder.add(Dense(2450))
        autoencoder.add(Activation("relu"))
        autoencoder.add(BatchNormalization(axis=chanDim))

        autoencoder.add(Reshape((7, 7, 50)))

        autoencoder.add(Conv2D(50, (5, 5), padding="same"))
        autoencoder.add(Activation("relu"))
        autoencoder.add(BatchNormalization(axis=chanDim))
        autoencoder.add(UpSampling2D(size=(2, 2)))

        autoencoder.add(Conv2D(20, (5, 5), padding="same",
                               input_shape=inputShape))
        autoencoder.add(Activation("relu"))
        autoencoder.add(BatchNormalization(axis=chanDim))
        autoencoder.add(UpSampling2D(size=(2, 2)))

        autoencoder.add(Conv2D(1, (5, 5), use_bias=True, padding='same'))
        autoencoder.add(Activation('sigmoid'))






        return autoencoder


    def plot_train_history_loss(self,history):
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.savefig(self.rcae_results+"rcae_")

    def compute_mse(self,Xclean, Xdecoded, lamda):
        # print len(Xdecoded)
        Xclean = np.reshape(Xclean, (len(Xclean), 784))
        m, n = Xclean.shape
        Xdecoded = np.reshape(np.asarray(Xdecoded), (m, n))
        # print Xdecoded.shape
        Xdecoded = np.reshape(Xdecoded, (len(Xdecoded), 784))

        print("[INFO:] Xclean  MSE Computed shape", Xclean.shape)

        print("[INFO:]Xdecoded  Computed shape", Xdecoded.shape)

        meanSq_error = mean_squared_error(Xclean, Xdecoded)
        print("[INFO:] MSE Computed shape", meanSq_error.shape)

        MNIST_DataLoader.mean_square_error_dict.update({lamda: meanSq_error})
        print("\n Mean square error Score ((Xclean, Xdecoded):")
        print(MNIST_DataLoader.mean_square_error_dict.values())

        return MNIST_DataLoader.mean_square_error_dict

    # Function to compute softthresholding values
    def soft_threshold(self,lamda, b):

        th = float(lamda) / 2.0
        print("(lamda,Threshold)", lamda, th)
        print("The type of b is ..., its len is ", type(b), b.shape, len(b[0]))

        if (lamda == 0):
            return b
        m, n = b.shape

        x = np.zeros((m, n))

        k = np.where(b > th)
        # print("(b > th)",k)
        # print("Number of elements -->(b > th) ",type(k))
        x[k] = b[k] - th

        k = np.where(np.absolute(b) <= th)
        # print("abs(b) <= th",k)
        # print("Number of elements -->abs(b) <= th ",len(k))
        x[k] = 0

        k = np.where(b < -th)
        # print("(b < -th )",k)
        # print("Number of elements -->(b < -th ) <= th",len(k))
        x[k] = b[k] + th
        x = x[:]

        return x

    def compute_best_worst_rank(self,testX, Xdecoded):
        # print len(Xdecoded)

        testX = np.reshape(testX, (len(testX), 784))
        m, n = testX.shape
        Xdecoded = np.reshape(np.asarray(Xdecoded), (m, n))
        # print Xdecoded.shape
        Xdecoded = np.reshape(Xdecoded, (len(Xdecoded), 784))

        # Rank the images by reconstruction error
        anamolies_dict = {}
        for i in range(0, len(testX)):
            anamolies_dict.update({i: np.linalg.norm(testX[i] - Xdecoded[i])})

        # Sort the recont error to get the best and worst 10 images
        best_top10_anamolies_dict = {}
        # Rank all the images rank them based on difference smallest  error
        best_sorted_keys = sorted(anamolies_dict, key=anamolies_dict.get, reverse=False)
        worst_top10_anamolies_dict = {}
        worst_sorted_keys = sorted(anamolies_dict, key=anamolies_dict.get, reverse=True)



        # Picking the top 10 images that were not reconstructed properly or badly reconstructed
        counter_best = 0
        # Show the top 10 most badly reconstructed images
        for b in best_sorted_keys:
            if (counter_best <= 29):
                counter_best = counter_best + 1
                best_top10_anamolies_dict.update({b: anamolies_dict[b]})
        best_top10_keys = best_top10_anamolies_dict.keys()

        # Picking the top 10 images that were not reconstructed properly or badly reconstructed
        counter_worst = 0
        # Show the top 10 most badly reconstructed images
        for w in worst_sorted_keys:
            if (counter_worst <= 29):
                counter_worst = counter_worst + 1
                worst_top10_anamolies_dict.update({w: anamolies_dict[w]})
        worst_top10_keys = worst_top10_anamolies_dict.keys()

        return [best_top10_keys, worst_top10_keys]

    def computePred_Labels(self, X_test, decoded, poslabelBoundary, negBoundary):

        y_pred = np.ones(len(X_test))
        recon_error = {}
        for i in range(0, len(X_test)):
            recon_error.update({i: np.linalg.norm(X_test[i] - decoded[i])})

        best_sorted_keys = sorted(recon_error, key=recon_error.get, reverse=False)
        worst_sorted_keys = sorted(recon_error, key=recon_error.get, reverse=True)
        anomaly_index = worst_sorted_keys[0:negBoundary]
        print("[INFO:] The anomaly index are ", anomaly_index)
        for key in anomaly_index:
            if (key >= poslabelBoundary):
                y_pred[key] = -1

        return y_pred

    def fit_auto_conv_AE(self,X_N,Xclean,lamda):

        # print("[INFO:] Lenet Style Autoencoder summary",autoencoder.summary())

        print("[INFO] compiling model...")
        # opt = SGD(lr=0.01, decay=0.01 / 150, momentum=0.9, nesterov=True)
        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        # opt =RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        self.cae.compile(loss=self.custom_rcae_loss(), optimizer=opt)

        self.lamda[0] = lamda
        X_N = np.reshape(X_N, (len(X_N), 28,28,1))
        Xclean = np.reshape(Xclean, (len(Xclean), 28, 28, 1))

        history = self.cae.fit(X_N, X_N,
                                          epochs=150,
                                          shuffle=True,
                                          validation_split=0.1,
                                          verbose=1
                                          )
        # callbacks = self.callbacks
        self.plot_train_history_loss(history)

        # model.fit(input, Xclean, n_epoch=10,
        #           run_id="auto_encoder", batch_size=128)

        ae_output = self.cae.predict(X_N)
        #Reshape it back to 784 pixels
        ae_output = np.reshape(ae_output, (len(ae_output), 784))
        Xclean = np.reshape(Xclean, (len(Xclean), 784))

        np_mean_mse =  np.mean(mean_squared_error(Xclean,ae_output))
        #Compute L2 norm during training and take the average of mse as threshold to set the label
        # norm = []
        # for i in range(0, len(input)):
        #      norm.append(np.linalg.norm(input[i] - ae_output[i]))
        # np_norm = np.asarray(norm)

        self.anomaly_threshold = np_mean_mse


        return ae_output

    def compute_softhreshold(self,Xtrue, N, lamda,Xclean):
        Xtrue = np.reshape(Xtrue, (len(Xtrue), 784))
        print
        "lamda passed ", lamda
        # inner loop for softthresholding
        for i in range(0, 1):
            X_N = Xtrue - N
            XAuto = self.fit_auto_conv_AE(X_N,Xtrue,lamda)  # XAuto is the predictions on train set of autoencoder
            XAuto = np.asarray(XAuto)
            # print "XAuto:",type(XAuto),XAuto.shape
            softThresholdIn = Xtrue - XAuto
            softThresholdIn = np.reshape(softThresholdIn, (len(softThresholdIn), 784))
            N = self.soft_threshold(lamda, softThresholdIn)
            print("Iteration NUmber is : ", i)
            print("NUmber of non zero elements  for N,lamda", np.count_nonzero(N), lamda)
            print("The shape of N", N.shape)
            print("The minimum value of N ", np.amin(N))
            print("The max value of N", np.amax(N))
        self.Noise = N
        return N

    def visualise_anamolies_detected(self,testX, noisytestX, decoded, N, best_top10_keys, worst_top10_keys, lamda):
        side = 28
        channel = 1
        N = np.reshape(N, (len(N), 28,28,1))
        # Display the decoded Original, noisy, reconstructed images
        print("side:", side)
        print("channel:", channel)
        img = np.ndarray(shape=(side * 3, side * 10, channel))
        print
        "img shape:", img.shape

        best_top10_keys = list(best_top10_keys)
        worst_top10_keys = list(worst_top10_keys)

        for i in range(10):
            row = i // 10 * 3
            col = i % 10
            img[side * row:side * (row + 1), side * col:side * (col + 1), :] = testX[best_top10_keys[i]]
            img[side * (row + 1):side * (row + 2), side * col:side * (col + 1), :] = decoded[best_top10_keys[i]]
            img[side * (row + 2):side * (row + 3), side * col:side * (col + 1), :] = N[best_top10_keys[i]]
            # img[side * (row + 3):side * (row + 4), side * col:side * (col + 1), :] = N[best_top10_keys[i]]

        img *= 255
        img = img.astype(np.uint8)
        img = np.reshape(img, (side * 3, side * 10))
        # Save the image decoded
        print("\nSaving results for best after being encoded and decoded: @")
        print(self.rcae_results + '/best/')
        io.imsave(self.rcae_results  + '/best/' + str(lamda) + 'salt_p_denoising_cae_decode.png', img)

        # Display the decoded Original, noisy, reconstructed images for worst
        img = np.ndarray(shape=(side * 3, side * 10, channel))
        for i in range(10):
            row = i // 10 * 3
            col = i % 10
            img[side * row:side * (row + 1), side * col:side * (col + 1), :] = testX[worst_top10_keys[i]]
            img[side * (row + 1):side * (row + 2), side * col:side * (col + 1), :] = decoded[worst_top10_keys[i]]
            img[side * (row + 2):side * (row + 3), side * col:side * (col + 1), :] = N[worst_top10_keys[i]]
            # img[side * (row + 3):side * (row + 4), side * col:side * (col + 1), :] = N[worst_top10_keys[i]]

        img *= 255
        img = img.astype(np.uint8)
        img = np.reshape(img, (side * 3, side * 10))
        # Save the image decoded
        print("\nSaving results for worst after being encoded and decoded: @")
        print(self.rcae_results + '/worst/')
        io.imsave(self.rcae_results + '/worst/' + str(lamda) + 'salt_p_denoising_cae_decode.png', img)

        return

    def evalPred(self,predX, trueX, trueY):

        trueX = np.reshape(trueX, (len(trueX), 784))
        predX = np.reshape(predX, (len(predX), 784))

        predY = np.ones(len(trueX))

        if predX.shape[1] > 1:
            # print("[INFO:] RecErr computed as (pred-actual)**2 ")
            # mse = []
            # for i in range(0, len(predX)):
            #     mse.append(mean_squared_error(trueX,predX))
            # np_mse = np.asarray(mse)
            # # print("[INFO:] The norm computed during eval")
            # # # if norm is greater than thereshold assign the value
            # predY[np.where(np_mse > self.anomaly_threshold)] = -1

            recErr = ((predX - trueX) ** 2).sum(axis=1)
        else:
            recErr = predX
            # predY = predX

        # print ("+++++++++++++++++++++++++++++++++++++++++++")
        # print (trueY)
        # print (predY)
        # print(predY.shape)
        # print(trueY.shape)
        # print ("+++++++++++++++++++++++++++++++++++++++++++")


        ap = average_precision_score(trueY, recErr)
        auc = roc_auc_score(trueY, recErr)

        prec = self.precAtK(recErr, trueY, K=10)

        return (ap, auc, prec)

    def precAtK(self,pred, trueY, K=None):

        if K is None:
            K = trueY.shape[0]

        # label top K largest predicted scores as +'ve
        idx = np.argsort(-pred)
        predLabel = -np.ones(trueY.shape)
        predLabel[idx[:K]] = 1

        # print(predLabel)

        prec = precision_score(trueY, predLabel)

        return prec

    def save_trained_model(self, model):

        ## save the model
        # serialize model to JSON
        model =  self.cae
        model_json = model.to_json()
        with open(self.modelsave_path + "DCAE_model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(self.modelsave_path + "DCAE_wts.h5")
        print("[INFO:] Saved model to disk @ ....",self.modelsave_path)


        return

    def pretrain_autoencoder(self):

        print("[INFO:] Pretraining Autoencoder start...")
        X_train = np.concatenate((self.data._X_train,self.data._X_val))
        y_train = np.concatenate((self.data._y_train, self.data._y_val))

        trainXPos = X_train[np.where(y_train == 1)]
        trainYPos = np.ones(len(trainXPos))
        trainXNeg = X_train[np.where(y_train == -1)]
        trainYNeg = -1*np.ones(len(trainXNeg))

        PosBoundary = len(trainXPos)
        NegBoundary = len(trainXNeg)


        # print("[INFO:]  Length of Positive data",len(trainXPos))
        # print("[INFO:]  Length of Negative data", len(trainXNeg))


        X_train = np.concatenate((trainXPos,trainXNeg))
        y_train = np.concatenate((trainYPos,trainYNeg))
        # print("[INFO:] X_test.shape", X_test.shape)
        # # print("[INFO:] y_test.shape", y_test)


        X_test = X_train
        y_test = y_train

        # X_test = self.data._X_test
        # y_test = self.data._y_test


        # print("[INFO:] X_test.shape",X_test.shape)
        # print("[INFO:] y_test.shape", y_test)
        #
        # print("[INFO:] y_train.shape", y_train.shape)
        # print("[INFO:] y_train.shape", y_train)

        # define lamda set
        lamda_set = [ 0.1]
        mue = 0.0
        TRIALS = 1
        ap = np.zeros((TRIALS,))
        auc = np.zeros((TRIALS,))
        prec = np.zeros((TRIALS,))
        # outer loop for lamda
        for l in range(0, len(lamda_set)):
            # Learn the N using softthresholding technique
            N = 0
            lamda = lamda_set[l]
            XTrue = X_train
            YTrue = y_train

            # Capture the structural Noise
            self.compute_softhreshold(XTrue, N, lamda, XTrue)
            N = self.Noise
            # Predict the conv_AE autoencoder output
            # XTrue = np.reshape(XTrue, (len(XTrue), 28, 28, 1))


            decoded = self.cae.predict(X_test)

            # compute MeanSqared error metric
            self.compute_mse(X_test, decoded, lamda)
            # print("[INFO:] The anomaly threshold computed is ", self.anomaly_threshold)

            # rank the best and worst reconstructed images
            [best_top10_keys, worst_top10_keys] = self.compute_best_worst_rank(X_test, decoded)

            # Visualise the best and worst ( image, BG-image, FG-Image)
            # XPred = np.reshape(np.asarray(decoded), (len(decoded), 28,28,1))
            self.visualise_anamolies_detected(X_test, X_test, decoded, N, best_top10_keys, worst_top10_keys, lamda)

            XPred = decoded

            y_pred = self.computePred_Labels(X_test,decoded,PosBoundary,NegBoundary)


            # (ap[l], auc[l], prec[l]) = self.nn_model.evalPred(XPred, X_test, y_test)
            auc[l] = roc_auc_score(y_test, y_pred)

            # print("AUPRC", lamda, ap[l])
            # print("AUROC", lamda, auc[l])
            # print("P@10", lamda, prec[l])
            print("=====================")
            print("AUROC", lamda, auc[l])
            print("=======================")
            print("[INFO:] Pretraining Autoencoder end saving autoencoder model @...")
            print("[INFO] serializing network and saving trained weights...")
            print("[INFO] Saving model config and layer weights...")
            self.save_trained_model()


        # print('AUPRC = %1.4f +- %1.4f' % (np.mean(ap), np.std(ap) / np.sqrt(TRIALS)))
        # print('AUROC = %1.4f +- %1.4f' % (np.mean(auc), np.std(auc) / np.sqrt(TRIALS)))
        # print('P@10  = %1.4f +- %1.4f' % (np.mean(prec), np.std(prec) / np.sqrt(TRIALS)))



        # print("\n Mean square error Score ((Xclean, Xdecoded):")
        # print(MNIST_DataLoader.mean_square_error_dict.values())
        # for k, v in MNIST_DataLoader.mean_square_error_dict.items():
        #     print(k, v)
        # # basic plot
        # data = MNIST_DataLoader.mean_square_error_dict.values()

        return





def load_mnist_images(filename):

    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)

    # reshaping and normalizing
    data = data.reshape(-1, 1, 28, 28).astype(np.float32)

    return data


def load_mnist_labels(filename):

    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)

    return data
