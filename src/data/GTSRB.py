from src.data.base import DataLoader
from src.data.preprocessing import center_data, normalize_data, rescale_to_unit_interval, \
    global_contrast_normalization, zca_whitening, extract_norm_and_out, learn_dictionary, pca
from src.utils.visualization.mosaic_plot import plot_mosaic
from src.utils.misc import flush_last_line
from src.config import Configuration as Cfg

import matplotlib
# matplotlib.use('Agg')  # or 'PS', 'PDF', 'SVG'

import matplotlib.pyplot as plt
import numpy as np
import cv2
import csv
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


# matplotlib.use('Agg')

class GTSRB_DataLoader(DataLoader):
    mean_square_error_dict= {}
    def __init__(self):

        DataLoader.__init__(self)

        self.dataset_name = "gtsrb"

        # GTSRB stop sign images (class 14)
        self.n_train = 780
        self.n_val = 0
        self.n_test = 290  # 270 normal examples and 20 adversarial examples
        self.n_test_adv = 20
        self.seed = Cfg.seed

        self.n_classes = 2
        self.prj_path = "/content/drive/My Drive/2018/Colab_Deep_Learning/one_class_neural_networks/"
        self.data_path = self.prj_path+"/data/data_gtsrb/"

        self.on_memory = True
        Cfg.store_on_gpu = True

        # load data from disk
        self.load_data()
        # print("Inside the MNIST_DataLoader RCAE.RESULT_PATH:", RCAE_AD.RESULT_PATH)
        self.rcae_results = self.prj_path+"/reports/figures/gtsrb/RCAE/"
        self.results = self.prj_path+"/reports/figures/gtsrb/Inputs/"
        self.modelsave_path = self.prj_path+"/models/gtsrb/RCAE/"
        ## Rcae parameters
        self.mue = 0.1
        self.lamda = [0.01]
        self.Noise = np.zeros(len(self._X_train))
        self.anomaly_threshold = 0.0
        self.cae = self.build_autoencoder()
        self.latent_weights = [0, 0, 0]
        self.batchNo = 0
        self.index = 0
        self.IMG_HGT=32
        self.IMG_WDT=32
        self.channel=3
        
       

    def check_specific(self):

        # store primal variables on RAM
        assert Cfg.store_on_gpu

    def save_reconstructed_image(self, Xtest, X_decoded):

        # use Matplotlib (don't ask)
        import matplotlib.pyplot as plt

        n = 5  # how many digits we will display
        plt.figure(figsize=(20, 4))
        for i in range(n):
            # display original
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(Xtest[i].reshape(32, 32,3))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(X_decoded[i].reshape(32, 32,3))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()
        plt.savefig(self.results + "/cae_input_images.png")

        return

    def load_data(self, original_scale=True):

        print("[INFO:] Loading data...")

        # get train data
        X = readTrafficSigns(rootpath=self.data_path, which_set="train", label=14)

        # X = readTrafficSigns_asnparray(rootpath=self.data_path, which_set="train", label=14)
        # X_test_norm = readTrafficSigns(rootpath=self.data_path, which_set="test", label=14)
        # X_test_adv = np.load(self.data_path + "/Images_150.npy")
        # print("Shape of input",X.shape)

        # X_test_adv = np.moveaxis(X_test_adv, 1, 3)



        # print("Shape of X_test_adv, ", X_test_adv.shape)

        # debug_visualise_anamolies_detected(X_test_adv,X_test_adv,X_test_adv,X_test_adv)
        #
        # plot_cifar(X_test_adv,10,10)
        # exit()

        # get (normal) test data
        # X_test_norm = readTrafficSigns(rootpath=self.data_path, which_set="test", label=14)
        # sub-sample test set data of size
        print("The random seed used in the experiment is ",self.seed )
        self.seed = Cfg.seed
        np.random.seed(self.seed)
        perm = np.random.permutation(len(X))
        X_test_norm = X[perm[:100], ...]
        
        self._X_train = X[perm[100:], ...]
        self.n_train = len(self._X_train)
        self._y_train = np.zeros(self.n_train, dtype=np.uint8)



        # load (adversarial) test data
        print("[INFO:] Loading adversarial data...")
        X_test_adv =  np.load(self.data_path + "/Images_150.npy")
        labels_adv = np.load(self.data_path + "/Labels_150.npy")
        # print("[INFO:] The number of Loading adversarial data...",len(X_test_adv))

        # X_test_adv , labels_adv = self.generate_AdversarialSigns(X_test_norm)

        self._X_test = np.concatenate((X_test_norm, X_test_adv[labels_adv == 1]), axis=0).astype(np.float32)
        self._y_test = np.concatenate((np.zeros(len(X_test_norm), dtype=np.uint8),
                                       1*np.ones(int(np.sum(labels_adv)), dtype=np.uint8)), axis=0)
        self.n_test = len(self._X_test)

        # since val set is referenced at some points initialize empty np arrays
        self._X_val = np.empty(shape=(0, 3, 32, 32), dtype=np.float32)
        self._y_val = np.empty(shape=(0), dtype=np.uint8)

        # Adjust number of batches
        Cfg.n_batches = int(np.ceil(self.n_train * 1. / Cfg.batch_size))

        # shuffle
        np.random.seed(self.seed)
        perm_train = np.random.permutation(self.n_train)
        perm_test = np.random.permutation(self.n_test)
        self._X_train = self._X_train[perm_train, ...]
        self._y_train = self._y_train[perm_train]
        self._X_test = self._X_test[perm_test, ...]
        self._y_test = self._y_test[perm_test]



        positiveSamples_test = self._X_test[np.where(self._y_test == 0)]
        positiveSamples_test = positiveSamples_test[0:100]
        positiveSamples_test = np.concatenate((self._X_train[0:170],positiveSamples_test))

        negativeSamples_test = self._X_test[np.where(self._y_test == 1)]
        #negativeSamples_test = negativeSamples_test[0:50]
        #negativeSamples_test = negativeSamples_test[0:40]
        negativeSamples_test = negativeSamples_test[0:100]

        self._X_train = np.concatenate((self._X_train,positiveSamples_test))

        self._X_train = self._X_train[0:780]
        self._y_train = np.zeros(len(self._X_train))


        
        y_positiveSamples_test = np.zeros(len(positiveSamples_test))
        y_negativeSamples_test = 1*np.ones(len(negativeSamples_test))

        self._X_test = np.concatenate((positiveSamples_test,negativeSamples_test))
        self._y_test = np.concatenate((y_positiveSamples_test,y_negativeSamples_test))


        # print("[INFO:] The number of train samples", len(self._X_train))
        # print("[INFO:] The number of test samples", len(self._X_test))
        print("[INFO:] Negative Y_test labels", len(self._y_test[np.where(self._y_test == 1)]))
        print("[INFO:] Positive Y_test labels", len(self._y_test[np.where(self._y_test == 0)]))


        # Adjust number of batches
        Cfg.n_batches = int(np.ceil(self.n_train * 1. / Cfg.batch_size))

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

        flush_last_line()
        
        # self._X_train = np.concatenate((self._X_train, self._X_test))
        # self._y_train = np.concatenate((self._y_train, self._y_test))
        # Make sure the axis dimensions are aligned for training convolutional autoencoders
        self._X_train = np.moveaxis(self._X_train, 1, 3)
        self._X_test = np.moveaxis(self._X_test, 1, 3)
       
              
        self._X_train = self._X_train/255.0
        self._X_test = self._X_test / 255.0
        
        X_train = self._X_train
        X_test = self._X_test
        y_train = self._y_train
        y_test = self._y_test
        
        print("X_train,X_test====>",X_train.shape, X_test.shape)
        
        ## Combine the positive data
        trainXPos = X_train[np.where(y_train == 0)]
        trainYPos = np.zeros(len(trainXPos))
        testXPos = X_test[np.where(y_test == 0)]
        testYPos = np.zeros(len(testXPos))
        
        
        # Combine the negative data
        trainXNeg = X_train[np.where(y_train == 1)]
        trainYNeg = np.ones(len(trainXNeg))
        testXNeg = X_test[np.where(y_test == 1)]
        testYNeg = np.ones(len(testXNeg))

        print("trainXPos,testXPos",trainXPos.shape, testXPos.shape)
        X_trainPOS = np.concatenate((trainXPos, testXPos))
        y_trainPOS = np.concatenate((trainYPos, testYPos))
        
        X_trainNEG = np.concatenate((trainXNeg, testXNeg))
        y_trainNEG = np.concatenate((trainYNeg, testYNeg))
        
        # Just 0.01 points are the number of anomalies.
        num_of_anomalies = int(0.1 * len(X_trainPOS))
        
        X_trainNEG = X_trainNEG[0:num_of_anomalies]
        y_trainNEG = y_trainNEG[0:num_of_anomalies]
        
        
        X_train = np.concatenate((X_trainPOS, X_trainNEG))
        y_train = np.concatenate((y_trainPOS, y_trainNEG))
        
        
        self._X_train = X_train
        self._y_train = y_train
        
        self._X_test = X_train
        self._y_test = y_train
        
        self._X_test_beforegcn = X_train
        self._y_test_beforegcn = y_train
        
        self._X_test_beforegcn = np.reshape(self._X_test_beforegcn,(len(self._X_test_beforegcn),32,32,3))
        
        X_test_sample = self._X_test[-5:]
        import random
        random_list = random.sample(range(1, 700), 5)
        
        
        X_train_sample = self._X_train[random_list]
        
        print("[INFO:] The shape of self.data._X_train", self._X_train.shape)
        print("[INFO:] The shape of self.data._X_test", self._X_test.shape)
         
        X_test = np.concatenate((X_train_sample, X_test_sample))
        
        # X_train_sample = np.moveaxis(X_train_sample, 1, 3)
        # X_test_sample = np.moveaxis(X_test_sample, 1, 3)
        # X_train_sample = X_train_sample/255.0
        # X_test_sample = X_test_sample / 255.0
  
        # self.save_reconstructed_image(X_train_sample, X_train_sample)
        
      
        
        

        
        # global contrast normalization
       
        # if Cfg.gcn:
        #       [self._X_train,self._X_val,self._X_test] = global_contrast_normalization(self._X_train, self._X_val, self._X_test, scale=Cfg.unit_norm_used)
        #       self._X_test = self._X_train            
        
        print("Data loaded.")
       
        
        
        


    def custom_rcae_loss(self):

        U = self.cae.layers[16].get_weights()
        U = U[0]

        V = self.cae.layers[19].get_weights()
        V = V[0]
        V = np.transpose(V)

        print("[INFO:] Shape of U, V",U.shape,V.shape)
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
        #     N_reshaped = np.reshape(N,(len(N),32,32,3))
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
            term2 = mue * 0.5 * (LA.norm(U) + LA.norm(V))
            term3 = lambda_val * 0.5 * LA.norm(N)

            return (term1 + term2 + term3)

        return custom_rcae


    def build_autoencoder1(self):
        input_img = Input(shape=(32, 32, 3))
        x = Conv2D(64, (3, 3), padding='same')(input_img)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(16, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(16, (3, 3), padding='same')(encoded)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(3, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        decoded = Activation('sigmoid')(x)
        model = Model(input_img, decoded)


        return model



    def build_autoencoder(self):

        # initialize the model
        autoencoder = Sequential()
        inputShape = (32,32,3)
        chanDim = -1 # since depth is appearing the end
        # first set of CONV => RELU => POOL layers
        autoencoder.add(Conv2D(64, (3, 3), padding="same",input_shape=inputShape))
        autoencoder.add(Activation("relu"))
        autoencoder.add(BatchNormalization(axis=chanDim))
        autoencoder.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # second set of CONV => RELU => POOL layers
        autoencoder.add(Conv2D(32, (3, 3), padding="same"))
        autoencoder.add(Activation("relu"))
        autoencoder.add(BatchNormalization(axis=chanDim))
        autoencoder.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        autoencoder.add(Conv2D(16, (3, 3), padding="same"))
        autoencoder.add(Activation("relu"))
        autoencoder.add(BatchNormalization(axis=chanDim))
        autoencoder.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # first (and only) set of FC => RELU layers
        autoencoder.add(Flatten())


        autoencoder.add(Dense(256))
        autoencoder.add(Activation("relu"))
        autoencoder.add(BatchNormalization(axis=chanDim))

        autoencoder.add(Dense(32))
        autoencoder.add(Activation("relu"))
        autoencoder.add(BatchNormalization(axis=chanDim))



        autoencoder.add(Dense(256))
        autoencoder.add(Activation("relu"))
        autoencoder.add(BatchNormalization(axis=chanDim))

        autoencoder.add(Reshape((4, 4, 16)))

        autoencoder.add(Conv2D(16, (3, 3), padding="same"))
        autoencoder.add(Activation("relu"))
        autoencoder.add(BatchNormalization(axis=chanDim))
        autoencoder.add(UpSampling2D(size=(2, 2)))

        autoencoder.add(Conv2D(32, (3, 3), padding="same"))
        autoencoder.add(Activation("relu"))
        autoencoder.add(BatchNormalization(axis=chanDim))
        autoencoder.add(UpSampling2D(size=(2, 2)))

        autoencoder.add(Conv2D(64, (3, 3), padding="same",
                               input_shape=inputShape))
        autoencoder.add(Activation("relu"))
        autoencoder.add(BatchNormalization(axis=chanDim))
        autoencoder.add(UpSampling2D(size=(2, 2)))

        autoencoder.add(Conv2D(3, (3, 3), use_bias=True, padding='same'))
        autoencoder.add(Activation('sigmoid'))

        # print("[INFO:] Autoencoder summary ", autoencoder.summary())

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
        Xclean = np.reshape(Xclean, (len(Xclean), 3072))
        m, n = Xclean.shape
        Xdecoded = np.reshape(np.asarray(Xdecoded), (m, n))
        # print Xdecoded.shape
        Xdecoded = np.reshape(Xdecoded, (len(Xdecoded), 3072))

        print("[INFO:] Xclean  MSE Computed shape", Xclean.shape)

        print("[INFO:]Xdecoded  Computed shape", Xdecoded.shape)

        meanSq_error = mean_squared_error(Xclean, Xdecoded)
        print("[INFO:] MSE Computed shape", meanSq_error.shape)

        GTSRB_DataLoader.mean_square_error_dict.update({lamda: meanSq_error})
        print("\n Mean square error Score ((Xclean, Xdecoded):")
        print(GTSRB_DataLoader.mean_square_error_dict.values())

        return GTSRB_DataLoader.mean_square_error_dict

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

        testX = np.reshape(testX, (len(testX), 3072))
        m, n = testX.shape
        Xdecoded = np.reshape(np.asarray(Xdecoded), (m, n))
        # print Xdecoded.shape
        Xdecoded = np.reshape(Xdecoded, (len(Xdecoded), 3072))

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
        # opt = SGD(lr=0.01, decay=0.01 / 350, momentum=0.9, nesterov=True)
        # opt =RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.cae.compile(loss=self.custom_rcae_loss(), optimizer=opt)
        # self.cae.compile(optimizer='adam', loss='mean_squared_error')


        self.lamda[0] = lamda
        X_N = np.reshape(X_N, (len(X_N), 32,32,3))
        Xclean = np.reshape(Xclean, (len(Xclean), 32,32,3))

        history = self.cae.fit(X_N, X_N,
                                          epochs=500,
                                          batch_size=64,
                                          shuffle=True,
                                          validation_split=0.1,
                                          verbose=0
                                          )
        # callbacks = self.callbacks
        self.plot_train_history_loss(history)

        # model.fit(input, Xclean, n_epoch=10,
        #           run_id="auto_encoder", batch_size=128)

        ae_output = self.cae.predict(X_N)
        #Reshape it back to 3072 pixels
        ae_output = np.reshape(ae_output, (len(ae_output), 3072))
        Xclean = np.reshape(Xclean, (len(Xclean), 3072))

        np_mean_mse =  np.mean(mean_squared_error(Xclean,ae_output))
        #Compute L2 norm during training and take the average of mse as threshold to set the label
        # norm = []
        # for i in range(0, len(input)):
        #      norm.append(np.linalg.norm(input[i] - ae_output[i]))
        # np_norm = np.asarray(norm)

        self.anomaly_threshold = np_mean_mse


        return ae_output

    def compute_softhreshold(self,Xtrue, N, lamda,Xclean):
        Xtrue = np.reshape(Xtrue, (len(Xtrue), 3072))
        print
        "lamda passed ", lamda
        # inner loop for softthresholding
        for i in range(0, 1):
            X_N = Xtrue - N
            XAuto = self.fit_auto_conv_AE(X_N,Xtrue,lamda)  # XAuto is the predictions on train set of autoencoder
            XAuto = np.asarray(XAuto)
            # print "XAuto:",type(XAuto),XAuto.shape
            softThresholdIn = Xtrue - XAuto
            softThresholdIn = np.reshape(softThresholdIn, (len(softThresholdIn), 3072))
            N = self.soft_threshold(lamda, softThresholdIn)
            print("Iteration NUmber is : ", i)
            print("NUmber of non zero elements  for N,lamda", np.count_nonzero(N), lamda)
            print("The shape of N", N.shape)
            print("The minimum value of N ", np.amin(N))
            print("The max value of N", np.amax(N))
        self.Noise = N
        return N

    def visualise_anamolies_detected(self,testX, noisytestX, decoded, N, best_top10_keys, worst_top10_keys, lamda):


        print("[INFO:] The shape of input data  ",testX.shape)
        print("[INFO:] The shape of decoded  data  ", decoded.shape)


        side =32
        channel = 3
        N = np.reshape(N, (len(N), 32, 32, 3))
        # Display the decoded Original, noisy, reconstructed images
        print("[INFO:] The shape of N  data  ", N.shape)

        img = np.ndarray(shape=(side * 4, side * 10, channel))
        print("img shape:", img.shape)


        best_top10_keys = list(best_top10_keys)
        worst_top10_keys = list(worst_top10_keys)

        for i in range(10):
            row = i // 10 * 4
            col = i % 10
            img[side * row:side * (row + 1), side * col:side * (col + 1), :] = testX[best_top10_keys[i]]
            img[side * (row + 1):side * (row + 2), side * col:side * (col + 1), :] = noisytestX[best_top10_keys[i]]
            img[side * (row + 2):side * (row + 3), side * col:side * (col + 1), :] = decoded[best_top10_keys[i]]
            img[side * (row + 3):side * (row + 4), side * col:side * (col + 1), :] = N[best_top10_keys[i]]

        img *= 255
        img = img.astype(np.uint8)

        # Save the image decoded
        print("\nSaving results for best after being encoded and decoded: @")
        print(self.rcae_results + '/best/')
        io.imsave(self.rcae_results + '/best/' + str(lamda) + '_RCAE.png', img)

        # Display the decoded Original, noisy, reconstructed images for worst
        img = np.ndarray(shape=(side * 4, side * 10, channel))
        for i in range(10):
            row = i // 10 * 4
            col = i % 10
            img[side * row:side * (row + 1), side * col:side * (col + 1), :] = testX[worst_top10_keys[i]]
            img[side * (row + 1):side * (row + 2), side * col:side * (col + 1), :] = noisytestX[worst_top10_keys[i]]
            img[side * (row + 2):side * (row + 3), side * col:side * (col + 1), :] = decoded[worst_top10_keys[i]]
            img[side * (row + 3):side * (row + 4), side * col:side * (col + 1), :] = N[worst_top10_keys[i]]

        img *= 255
        img = img.astype(np.uint8)

        # Save the image decoded
        print("\nSaving results for worst after being encoded and decoded: @")
        print(self.rcae_results + '/worst/')
        io.imsave(self.rcae_results + '/worst/' + str(lamda) + '_RCAE.png', img)

        return

    def build_architecture(self, nnet):

        if Cfg.weight_dict_init & (not nnet.pretrained):
            # initialize first layer filters by atoms of a dictionary
            W1_init = learn_dictionary(nnet.data._X_train, n_filters=16, filter_size=5, n_sample=500)
            plot_mosaic(W1_init, title="First layer filters initialization", canvas="black",
                        export_pdf=(Cfg.xp_path + "/filters_init"))
        else:
            W1_init = None

        # Build LeNet 5 type architecture

        nnet.addInputLayer(shape=(None, 3, 32, 32))

        if Cfg.weight_dict_init & (not nnet.pretrained):
            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=16, filter_size=(5, 5), pad='same',
                              W=W1_init, b=None)
        else:
            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=16, filter_size=(5, 5), pad='same',
                              b=None)
        if Cfg.leaky_relu:
            nnet.addLeakyReLU()
        else:
            nnet.addReLU()
        nnet.addMaxPool(pool_size=(2, 2))

        nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=32, filter_size=(5, 5), pad='same', b=None)
        if Cfg.leaky_relu:
            nnet.addLeakyReLU()
        else:
            nnet.addReLU()
        nnet.addMaxPool(pool_size=(2, 2))

        nnet.addDenseLayer(num_units=64, b=None)
        if Cfg.leaky_relu:
            nnet.addLeakyReLU()
        else:
            nnet.addReLU()

        nnet.addDenseLayer(num_units=Cfg.gtsrb_rep_dim, b=None)

        if Cfg.softmax_loss:
            nnet.addDenseLayer(num_units=1)
            nnet.addSigmoidLayer()
        elif Cfg.svdd_loss:
            nnet.setFeatureLayer()  # set the currently highest layer to be the SVDD feature layer
        else:
            raise ValueError("No valid choice of loss for dataset " + self.dataset_name)

    def build_autoencoder_oldgtsrb(self, nnet):

        if Cfg.weight_dict_init & (not nnet.pretrained):
            # initialize first layer filters by atoms of a dictionary
            W1_init = learn_dictionary(nnet.data._X_train, n_filters=16, filter_size=5, n_sample=500)
            plot_mosaic(W1_init, title="First layer filters initialization", canvas="black",
                        export_pdf=(Cfg.xp_path + "/filters_init"))
        else:
            W1_init = None

        # Build autoencoder

        nnet.addInputLayer(shape=(None, 3, 32, 32))

        if Cfg.weight_dict_init & (not nnet.pretrained):
            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=16, filter_size=(5, 5), pad='same',
                              W=W1_init, b=None)
        else:
            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=16, filter_size=(5, 5), pad='same', b=None)
        if Cfg.leaky_relu:
            nnet.addLeakyReLU()
        else:
            nnet.addReLU()
        nnet.addMaxPool(pool_size=(2, 2))

        nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=32, filter_size=(5, 5), pad='same', b=None)
        if Cfg.leaky_relu:
            nnet.addLeakyReLU()
        else:
            nnet.addReLU()
        nnet.addMaxPool(pool_size=(2, 2))

        nnet.addDenseLayer(num_units=64, b=None)
        if Cfg.leaky_relu:
            nnet.addLeakyReLU()
        else:
            nnet.addReLU()

        # Code Layer
        nnet.addDenseLayer(num_units=Cfg.gtsrb_rep_dim, b=None)
        nnet.setFeatureLayer()  # set the currently highest layer to be the SVDD feature layer

        nnet.addDenseLayer(num_units=64, b=None)
        nnet.addReshapeLayer(shape=([0], 1, 8, 8))
        if Cfg.leaky_relu:
            nnet.addLeakyReLU()
        else:
            nnet.addReLU()

        nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=32, filter_size=(5, 5), pad='same', b=None)
        if Cfg.leaky_relu:
            nnet.addLeakyReLU()
        else:
            nnet.addReLU()
        nnet.addUpscale(scale_factor=(2, 2))

        nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=16, filter_size=(5, 5), pad='same', b=None)
        if Cfg.leaky_relu:
            nnet.addLeakyReLU()
        else:
            nnet.addReLU()
        nnet.addUpscale(scale_factor=(2, 2))

        nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=3, filter_size=(5, 5), pad='same', b=None)
        nnet.addSigmoidLayer()

    def apply_BoundaryAttack(self,images,fmodel):

        adversarial = []
        labels = np.ones(len(images))
        import foolbox
        # apply attack on source image
        # ::-1 reverses the color channels, because Keras ResNet50 expects BGR instead of RGB
        # attack = foolbox.attacks.BoundaryAttack(fmodel)
        attack = foolbox.attacks.FGSM(fmodel)

        for image in images:
            # adversarial.append(attack(image, -1))
            adversarial.append(attack(image[:, :, ::-1], -1))

        # if the attack fails, adversarial will be None and a warning will be printed
        adv_labels = -1* np.ones(len(adversarial))
        adversarial = np.asarray(adversarial)
        adversarial = adversarial[0:self.n_test_adv] ## pick 20 Boundary attack samples
        adv_labels = adv_labels[0:self.n_test_adv] ## assign -1 as their labels

        return adversarial,adv_labels


    def generate_AdversarialSigns(self,X_normal):
        import foolbox
        import keras
        import numpy as np
        from keras.applications.resnet50 import ResNet50

        # instantiate model
        keras.backend.set_learning_phase(0)
        kmodel = ResNet50(weights='imagenet')
        preprocessing = (np.array([104, 116, 123]), 1)
        fmodel = foolbox.models.KerasModel(kmodel, bounds=(0, 255), preprocessing=preprocessing)

        attack_images, attack_labels = self.apply_BoundaryAttack(X_normal,fmodel)


        return attack_images,attack_labels


def readTrafficSigns(rootpath, which_set="train", label=14):
    '''
    Reads traffic sign data for German Traffic Sign Recognition Benchmark.
    '''

    images = [] # images
    labels = [] # corresponding labels

    if which_set == "train":
        dir_path = rootpath + "Final_Training/Images"
        prefix = dir_path + '/' + format(label, '05d') + '/'  # subdirectory for class
        gtFile = open(prefix + 'GT-' + format(label, '05d') + '.csv')  # annotations file
    if which_set == "test":
        dir_path = rootpath + "Final_Test/Images"
        prefix = dir_path + '/'
        gtFile = open(prefix + '/' + 'GT-final_test.csv')  # annotations file

    gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
    # gtReader.next() # skip header
    next(gtReader)
    # loop over all images in current annotations file
    for row in gtReader:
        x1 = int(row[3])
        y1 = int(row[4])
        x2 = int(row[5])
        y2 = int(row[6])
        img = plt.imread(prefix + row[0])  # the 1th column is the filename
        img = img[x1:x2, y1:y2, :]  # remove border of 10% around sign
        img = cv2.resize(img, (32, 32))  # resize to 32x32
        img = np.rollaxis(img, 2)  # img.shape = (3, 32, 32)
        images.append(img)
        labels.append(int(row[7]))  # the 8th column is the label
    gtFile.close()

    # convert to numpy arrays
    idx = (np.array(labels) == label)
    n = np.sum(idx)
    X = np.zeros((n, 3, 32, 32), np.float32)
    i = 0
    for img in range(len(images)):
        if idx[img]:
            X[i, :] = images[img]
            i += 1
        else:
            pass

    return X



import numpy
from PIL import Image
import PIL


def PIL2array(img):
    return numpy.array(img.getdata(),
                       numpy.uint8).reshape(img.size[1], img.size[0], 3)

def readTrafficSigns_asnparray(rootpath, which_set="train", label=14):
    '''
    Reads traffic sign data for German Traffic Sign Recognition Benchmark.
    '''

    images = [] # images
    labels = [] # corresponding labels

    if which_set == "train":
        dir_path = rootpath + "Final_Training/Images"
        prefix = dir_path + '/' + format(label, '05d') + '/'  # subdirectory for class
        gtFile = open(prefix + 'GT-' + format(label, '05d') + '.csv')  # annotations file
    if which_set == "test":
        dir_path = rootpath + "Final_Test/Images"
        prefix = dir_path + '/'
        gtFile = open(prefix + '/' + 'GT-final_test.csv')  # annotations file

    gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
    # gtReader.next() # skip header
    next(gtReader)
    result = []
    # loop over all images in current annotations file
    for row in gtReader:
        x1 = int(row[3])
        y1 = int(row[4])
        x2 = int(row[5])
        y2 = int(row[6])
        # img = plt.imread(prefix + row[0])  # the 1th column is the filename
        img = Image.open(prefix + row[0])
        # img = img[x1:x2, y1:y2, :]  # remove border of 10% around sign
        img = img.resize((32, 32), PIL.Image.ANTIALIAS)
        result.append(PIL2array(img))

        # img = cv2.resize(img, (32, 32))  # resize to 32x32
        # img = np.rollaxis(img, 2)  # img.shape = (3, 32, 32)
        # images.append(img)
        labels.append(int(row[7]))  # the 8th column is the label
    gtFile.close()

    # convert to numpy arrays
    X = np.asarray(result)

    # idx = (np.array(labels) == label)
    # n = np.sum(idx)
    # X = np.zeros((n, 3, 32, 32), np.float32)
    # i = 0
    # for img in range(len(images)):
    #     if idx[img]:
    #         X[i, :] = images[img]
    #         i += 1
    #     else:
    #         pass

    return X

def plot_cifar(data, row, col, scale=3., label_list=None):
        fig_width = data.shape[0] / 80 * row * scale
        fig_height = data.shape[1] / 80 * col * scale
        fig, axes = plt.subplots(row,
                                 col,
                                 figsize=(fig_height, fig_width))
        for i in range(row * col):
            # train[i][0] is i-th image data with size 32x32
            image = data[i]
            # image = image.transpose(1, 2, 0)
            r, c = divmod(i, col)
            # axes[r][c].imshow(image)  # cmap='gray' is for black and white picture.
            axes[r][c].imshow((image * 255).astype(np.uint8))

            axes[r][c].axis('off')  # do not show axis value
        plt.tight_layout()  # automatic padding between subplots
        plt.savefig("/Users/raghav/envPython3/experiments/one_class_neural_networks/reports/figures/gtsrb/RCAE/" + "_Xtrain.png")




def debug_visualise_anamolies_detected(testX, noisytestX, decoded, N, best_top10_keys, worst_top10_keys, lamda):

        #
        # print("[INFO:] The shape of input data  ",testX.shape)
        # print("[INFO:] The shape of decoded  data  ", decoded.shape)


        side =32
        channel = 3
        N = np.reshape(N, (len(N), 32, 32, 3))
        # Display the decoded Original, noisy, reconstructed images
        print("[INFO:] The shape of N  data  ", N.shape)

        img = np.ndarray(shape=(side * 4, side * 10, channel))
        print("img shape:", img.shape)


        best_top10_keys = list(best_top10_keys)
        worst_top10_keys = list(worst_top10_keys)

        for i in range(10):
            row = i // 10 * 4
            col = i % 10
            img[side * row:side * (row + 1), side * col:side * (col + 1), :] = testX[best_top10_keys[i]]
            img[side * (row + 1):side * (row + 2), side * col:side * (col + 1), :] = noisytestX[best_top10_keys[i]]
            img[side * (row + 2):side * (row + 3), side * col:side * (col + 1), :] = decoded[best_top10_keys[i]]
            img[side * (row + 3):side * (row + 4), side * col:side * (col + 1), :] = N[best_top10_keys[i]]

        # img *= 255
        img = img.astype(np.uint8)

        # Save the image decoded
        print("\nSaving results for best after being encoded and decoded: @")
        save_results = "/Users/raghav/envPython3/experiments/one_class_neural_networks/reports/figures/cifar10/RCAE/"
        print(save_results+"/best/")
        io.imsave(save_results + '/best/' + str(lamda) + '_RCAE.png', img)

        # Display the decoded Original, noisy, reconstructed images for worst
        img = np.ndarray(shape=(side * 4, side * 10, channel))
        for i in range(10):
            row = i // 10 * 4
            col = i % 10
            img[side * row:side * (row + 1), side * col:side * (col + 1), :] = testX[worst_top10_keys[i]]
            img[side * (row + 1):side * (row + 2), side * col:side * (col + 1), :] = noisytestX[worst_top10_keys[i]]
            img[side * (row + 2):side * (row + 3), side * col:side * (col + 1), :] = decoded[worst_top10_keys[i]]
            img[side * (row + 3):side * (row + 4), side * col:side * (col + 1), :] = N[worst_top10_keys[i]]

        # img *= 255
        img = img.astype(np.uint8)

        # Save the image decoded
        print("\nSaving results for worst after being encoded and decoded: @")
        print(save_results + '/worst/')
        io.imsave(save_results + '/worst/' + str(lamda) + '_RCAE.png', img)

        return



