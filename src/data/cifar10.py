from src.data.base import DataLoader
from src.data.preprocessing import center_data, normalize_data, rescale_to_unit_interval, \
    global_contrast_normalization, zca_whitening, extract_norm_and_out, learn_dictionary, pca
from src.utils.visualization.mosaic_plot import plot_mosaic
from src.utils.misc import flush_last_line
from src.config import Configuration as Cfg

import os
import numpy as np
import pickle
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

PROJECT_DIR = "/content/drive/My Drive/2019/testing/oc-nn/"


from keras.callbacks import Callback

class CIFAR_10_DataLoader(DataLoader):
    mean_square_error_dict ={}
    def __init__(self):

        DataLoader.__init__(self)

        self.dataset_name = "cifar10"

        self.n_train = 45000
        self.n_val = 5000
        self.n_test = 10000
        self.num_outliers = 500

        self.seed = Cfg.seed

        if Cfg.ad_experiment:
            self.n_classes = 2
        else:
            self.n_classes = 10

        Cfg.n_batches = int(np.ceil(self.n_train * 1. / Cfg.batch_size))

        self.data_path = PROJECT_DIR+"/data/cifar-10/cifar-10-batches-py/"

        self.on_memory = True
        Cfg.store_on_gpu = True

        # load data from disk
        self.load_data()

        # print("Inside the MNIST_DataLoader RCAE.RESULT_PATH:", RCAE_AD.RESULT_PATH)
        self.rcae_results = PROJECT_DIR+"/reports/figures/cifar10/RCAE/"
        self.modelsave_path = PROJECT_DIR+"/models/cifar10/DCAE/"

        print("Inside the CIFAR10_DataLoader RCAE.RESULT_PATH:", self.rcae_results)

        # load data from disk
        self.load_data()

        ## Rcae parameters
        self.mue = 0.1
        self.lamda = [0.01]
        self.Noise = np.zeros(len(self._X_train))
        self.anomaly_threshold = 0.0
        self.cae = self.build_autoencoder()
        self.latent_weights = [0, 0, 0]
        self.batchNo = 0
        self.index = 0


    def check_specific(self):

        # store primal variables on RAM
        assert Cfg.store_on_gpu

    def load_data(self):
        print("[INFO:] Loading data...")
        print("The normal label used in experiment,",Cfg.cifar10_normal)
        # normalize data
        from keras.datasets import cifar10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255.0
        x_test /= 255.0
        
      
        ## Added newly
        x_train = np.concatenate((x_train, x_test))
        y_train = np.concatenate((y_train, y_test))
        
        
        # if(Cfg.cifar10_normal == 1 or Cfg.cifar10_normal == 7 or Cfg.cifar10_normal == 9 ):
        #         x_train = x_train
        #         y_train = y_train
        # else:
        #     x_train = np.concatenate((x_train, x_test))
        #     y_train = np.concatenate((y_train, y_test))
         
        ##Added newly
        
        
        
       
        y_train = np.reshape(y_train, len(y_train))
        x_norm = x_train[np.where(y_train == Cfg.cifar10_normal)]
        y_norm = np.zeros(len(x_norm))
        
        outliers = list(range(0, 10))
        outliers.remove(Cfg.cifar10_normal)
        idx_outlier = np.any(y_train[..., None] == np.array(outliers)[None, ...], axis=1)
        x_outlier = x_train[idx_outlier]
        
        print("INFO: Random Seed set is ",self.seed)
        np.random.seed(self.seed)
        x_outlier = np.random.permutation(x_outlier)[:self.num_outliers]
        
        # x_outlier = x_outlier[0:self.num_outliers]
        
        
        y_outlier =  np.ones(len(x_outlier))
        
        # x_outlier shape 
        print('x_outlier shape:', x_outlier.shape)
       
        
        x_train = np.concatenate((x_norm, x_outlier))
        y_train = np.concatenate((y_norm, y_outlier))

        # print(x_train.shape)
        self._X_train = x_train
        self._y_train = y_train
        self._X_val = np.empty(x_train.shape)
        self._y_val = np.empty(y_train.shape)

        self._X_test = x_train
        self._y_test = y_train
        
        # print("INFO Saving images befoer gcn ....")
        self._X_test_beforegcn = x_train
        self._y_test_beforegcn = y_train
        
        print("_X_test_beforegcn,",self._X_test_beforegcn.shape,np.max(self._X_test_beforegcn),np.min(self._X_test_beforegcn))
        
        # Xtest = Xtest/255.0
      
        
        gcn_required_for_classes = [ 1,3,5,6,7,9]
        # global contrast normalization
        if(Cfg.cifar10_normal in gcn_required_for_classes):
            if Cfg.gcn:
              [self._X_train,self._X_val,self._X_test] = global_contrast_normalization(self._X_train, self._X_val, self._X_test, scale=Cfg.unit_norm_used)
              self._X_test = self._X_train 
        
        print("Data loaded.")
        return



    def load_data1(self, original_scale=False):

        print("Loading data...")

        # load training data
        X, y = [], []
        count = 1
        filename = '%s/data_batch_%i' % (self.data_path, count)
        while os.path.exists(filename):
            with open(filename, 'rb') as f:
                batch = pickle.load(f,encoding='latin1')
            X.append(batch['data'])
            y.append(batch['labels'])
            count += 1
            filename = '%s/data_batch_%i' % (self.data_path, count)
        # X = np.asarray(X)
        # best_top10_keys = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # worst_top10_keys = [5001, 5002, 5003, 5004, 5005, 5006, 5007, 5008, 5009, 5010]
        #
        # debug_visualise_anamolies_detected(X,X,X,X,best_top10_keys,worst_top10_keys,0.0)
        #
        from keras.datasets import cifar10

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()


        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        X = x_train
        y = y_train

        # print("[INFO:] The shape of X before reshape is ", X.shape)

        # reshape data and cast them properly
        # X = np.concatenate(X).reshape(-1, 32, 32, 3).astype(np.float32)
        # y = np.concatenate(y).astype(np.int32)
        # print("[INFO:] The shape of X after reshape is ", X)



        # best_top10_keys = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # worst_top10_keys = [5001, 5002, 5003, 5004, 5005, 5006, 5007, 5008, 5009, 5010]
        # debug_visualise_anamolies_detected(X, X, X, X, best_top10_keys, worst_top10_keys,
        #                                            0.0)
        # exit()

        #load test set
        # path = '%s/test_batch' % self.data_path
        # with open(path, 'rb') as f:
        #     batch = pickle.load(f,encoding='latin1')



        #reshaping and casting for test data
        # X_test = batch['data'].reshape(-1, 32, 32, 3).astype(np.float32)
        # y_test = np.array(batch['labels'], dtype=np.int32)
        X_test = x_test
        y_test = y_test


        if Cfg.ad_experiment:

            # set normal and anomalous class
            normal = []
            outliers = []

            if Cfg.cifar10_normal == -1:
                normal = list(range(0, 10))
                normal.remove(Cfg.cifar10_outlier)
            else:
                normal.append(Cfg.cifar10_normal)

            if Cfg.cifar10_outlier == -1:
                outliers = list(range(0, 10))
                outliers.remove(Cfg.cifar10_normal)
            else:
                outliers.append(Cfg.cifar10_outlier)
            
            print("The normal label used in experiment,",Cfg.cifar10_normal)
            # extract normal and anomalous class
            X_norm, X_out, y_norm, y_out = extract_norm_and_out(X, y, normal=normal, outlier=outliers)

            # best_top10_keys = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            # worst_top10_keys = [3001, 3002, 3003, 3004, 3005, 3006, 3007, 3008, 3009, 3010]
            # debug_visualise_anamolies_detected(X_norm,X_out,X_norm,X_out,best_top10_keys,worst_top10_keys,0.0)
            # exit()

            # reduce outliers to fraction defined
            n_norm = len(y_norm)
            n_out = int(np.ceil(Cfg.out_frac * n_norm / (1 - Cfg.out_frac)))

            # shuffle to obtain random validation splits
            np.random.seed(self.seed)
            perm_norm = np.random.permutation(len(y_norm))
            perm_out = np.random.permutation(len(y_out))

            # split into training and validation set
            n_norm_split = int(Cfg.cifar10_val_frac * n_norm)
            n_out_split = int(Cfg.cifar10_val_frac * n_out)
            self._X_train = np.concatenate((X_norm[perm_norm[n_norm_split:]],
                                            X_out[perm_out[:n_out][n_out_split:]]))
            self._y_train = np.append(y_norm[perm_norm[n_norm_split:]],
                                      y_out[perm_out[:n_out][n_out_split:]])
            self._X_val = np.concatenate((X_norm[perm_norm[:n_norm_split]],
                                          X_out[perm_out[:n_out][:n_out_split]]))
            self._y_val = np.append(y_norm[perm_norm[:n_norm_split]],
                                    y_out[perm_out[:n_out][:n_out_split]])

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

        else:
            # split into training and validation sets with stored seed
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

        flush_last_line()


        print("Data loaded.")

    def build_architecture(self, nnet):

        # implementation of different network architectures
        assert Cfg.cifar10_architecture in (1, 2, 3)

        if Cfg.cifar10_architecture == 1:

            if Cfg.weight_dict_init & (not nnet.pretrained):
                # initialize first layer filters by atoms of a dictionary
                W1_init = learn_dictionary(nnet.data._X_train, n_filters=16, filter_size=5, n_sample=500)
                plot_mosaic(W1_init, title="First layer filters initialization",
                            canvas="black",
                            export_pdf=(Cfg.xp_path + "/filters_init"))
            else:
                W1_init = None

            # build architecture 1
            nnet.addInputLayer(shape=(None, 3, 32, 32))

            if Cfg.cifar10_bias:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=16, filter_size=(5, 5), pad='same')
            else:
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
            if Cfg.cifar10_bias:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=16, filter_size=(5, 5), pad='same')
            else:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=16, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            nnet.addMaxPool(pool_size=(2, 2))

            if Cfg.cifar10_bias:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=32, filter_size=(5, 5), pad='same')
            else:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=32, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            if Cfg.cifar10_bias:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=32, filter_size=(5, 5), pad='same')
            else:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=32, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            nnet.addMaxPool(pool_size=(2, 2))

            if Cfg.cifar10_bias:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=64, filter_size=(5, 5), pad='same')
            else:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=64, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            if Cfg.cifar10_bias:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=64, filter_size=(5, 5), pad='same')
            else:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=64, filter_size=(5, 5),  pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            nnet.addMaxPool(pool_size=(2, 2))

            if Cfg.cifar10_bias:
                nnet.addDenseLayer(num_units=Cfg.cifar10_rep_dim)
            else:
                nnet.addDenseLayer(num_units=Cfg.cifar10_rep_dim, b=None)

            if Cfg.softmax_loss:
                nnet.addDenseLayer(num_units=1)
                nnet.addSigmoidLayer()
            elif Cfg.svdd_loss:
                nnet.setFeatureLayer()  # set the currently highest layer to be the SVDD feature layer
            else:
                raise ValueError("No valid choice of loss for dataset " + self.dataset_name)

        if Cfg.cifar10_architecture == 2:

            if Cfg.weight_dict_init & (not nnet.pretrained):
                # initialize first layer filters by atoms of a dictionary
                W1_init = learn_dictionary(nnet.data._X_train, n_filters=16, filter_size=5, n_sample=500)
                plot_mosaic(W1_init, title="First layer filters initialization",
                            canvas="black",
                            export_pdf=(Cfg.xp_path + "/filters_init"))
            else:
                W1_init = None

            # build architecture 2
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

            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=32, filter_size=(5, 5), pad='same',
                              b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            nnet.addMaxPool(pool_size=(2, 2))

            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=64, filter_size=(5, 5), pad='same',
                              b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            nnet.addMaxPool(pool_size=(2, 2))

            nnet.addDenseLayer(num_units=Cfg.cifar10_rep_dim, b=None)

            if Cfg.softmax_loss:
                nnet.addDenseLayer(num_units=1)
                nnet.addSigmoidLayer()
            elif Cfg.svdd_loss:
                nnet.setFeatureLayer()  # set the currently highest layer to be the SVDD feature layer
            else:
                raise ValueError("No valid choice of loss for dataset " + self.dataset_name)

        if Cfg.cifar10_architecture == 3:

            if Cfg.weight_dict_init & (not nnet.pretrained):
                # initialize first layer filters by atoms of a dictionary
                W1_init = learn_dictionary(nnet.data._X_train, n_filters=32, filter_size=5, n_sample=500)
                plot_mosaic(W1_init, title="First layer filters initialization",
                            canvas="black",
                            export_pdf=(Cfg.xp_path + "/filters_init"))
            else:
                W1_init = None

            # build architecture 3
            nnet.addInputLayer(shape=(None, 3, 32, 32))

            if Cfg.weight_dict_init & (not nnet.pretrained):
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=32, filter_size=(5, 5), pad='same',
                                  W=W1_init, b=None)
            else:
                nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=32, filter_size=(5, 5), pad='same',
                                  b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            nnet.addMaxPool(pool_size=(2, 2))

            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=64, filter_size=(5, 5), pad='same', b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            nnet.addMaxPool(pool_size=(2, 2))

            nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=128, filter_size=(5, 5), pad='same',
                              b=None)
            if Cfg.leaky_relu:
                nnet.addLeakyReLU()
            else:
                nnet.addReLU()
            nnet.addMaxPool(pool_size=(2, 2))

            nnet.addDenseLayer(num_units=Cfg.cifar10_rep_dim, b=None)

            if Cfg.softmax_loss:
                nnet.addDenseLayer(num_units=1)
                nnet.addSigmoidLayer()
            elif Cfg.svdd_loss:
                nnet.setFeatureLayer()  # set the currently highest layer to be the SVDD feature layer
            else:
                raise ValueError("No valid choice of loss for dataset " + self.dataset_name)

    # def build_autoencoder(self, nnet):
    #
    #     # implementation of different network architectures
    #     assert Cfg.cifar10_architecture in (1, 2, 3)
    #
    #     if Cfg.cifar10_architecture == 1:
    #
    #         if Cfg.weight_dict_init & (not nnet.pretrained):
    #             # initialize first layer filters by atoms of a dictionary
    #             W1_init = learn_dictionary(nnet.data._X_train, 16, 5, n_sample=500)
    #             plot_mosaic(W1_init, title="First layer filters initialization",
    #                         canvas="black",
    #                         export_pdf=(Cfg.xp_path + "/filters_init"))
    #
    #         nnet.addInputLayer(shape=(None, 3, 32, 32))
    #
    #         if Cfg.weight_dict_init & (not nnet.pretrained):
    #             nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=16, filter_size=(5, 5), pad='same',
    #                               W=W1_init, b=None)
    #         else:
    #             nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=16, filter_size=(5, 5), pad='same', b=None)
    #         if Cfg.leaky_relu:
    #             nnet.addLeakyReLU()
    #         else:
    #             nnet.addReLU()
    #         nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=16, filter_size=(5, 5), pad='same', b=None)
    #         if Cfg.leaky_relu:
    #             nnet.addLeakyReLU()
    #         else:
    #             nnet.addReLU()
    #         nnet.addMaxPool(pool_size=(2, 2))
    #
    #         nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=32, filter_size=(5, 5), pad='same', b=None)
    #         if Cfg.leaky_relu:
    #             nnet.addLeakyReLU()
    #         else:
    #             nnet.addReLU()
    #         nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=32, filter_size=(5, 5), pad='same', b=None)
    #         if Cfg.leaky_relu:
    #             nnet.addLeakyReLU()
    #         else:
    #             nnet.addReLU()
    #         nnet.addMaxPool(pool_size=(2, 2))
    #
    #         nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=64, filter_size=(5, 5), pad='same', b=None)
    #         if Cfg.leaky_relu:
    #             nnet.addLeakyReLU()
    #         else:
    #             nnet.addReLU()
    #         nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=64, filter_size=(5, 5), pad='same', b=None)
    #         if Cfg.leaky_relu:
    #             nnet.addLeakyReLU()
    #         else:
    #             nnet.addReLU()
    #         nnet.addMaxPool(pool_size=(2, 2))
    #
    #         # Code Layer
    #         nnet.addDenseLayer(num_units=Cfg.cifar10_rep_dim, b=None)
    #         nnet.setFeatureLayer()  # set the currently highest layer to be the SVDD feature layer
    #         nnet.addReshapeLayer(shape=([0], (Cfg.cifar10_rep_dim / 4), 2, 2))
    #         if Cfg.leaky_relu:
    #             nnet.addLeakyReLU()
    #         else:
    #             nnet.addReLU()
    #         nnet.addUpscale(scale_factor=(2, 2))
    #
    #         nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=64, filter_size=(5, 5), pad='same', b=None)
    #         if Cfg.leaky_relu:
    #             nnet.addLeakyReLU()
    #         else:
    #             nnet.addReLU()
    #         nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=64, filter_size=(5, 5), pad='same', b=None)
    #         if Cfg.leaky_relu:
    #             nnet.addLeakyReLU()
    #         else:
    #             nnet.addReLU()
    #         nnet.addUpscale(scale_factor=(2, 2))
    #
    #         nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=32, filter_size=(5, 5), pad='same', b=None)
    #         if Cfg.leaky_relu:
    #             nnet.addLeakyReLU()
    #         else:
    #             nnet.addReLU()
    #         nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=32, filter_size=(5, 5), pad='same', b=None)
    #         if Cfg.leaky_relu:
    #             nnet.addLeakyReLU()
    #         else:
    #             nnet.addReLU()
    #         nnet.addUpscale(scale_factor=(2, 2))
    #
    #         nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=16, filter_size=(5, 5), pad='same', b=None)
    #         if Cfg.leaky_relu:
    #             nnet.addLeakyReLU()
    #         else:
    #             nnet.addReLU()
    #         nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=16, filter_size=(5, 5), pad='same', b=None)
    #         if Cfg.leaky_relu:
    #             nnet.addLeakyReLU()
    #         else:
    #             nnet.addReLU()
    #         nnet.addUpscale(scale_factor=(2, 2))
    #
    #         nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=3, filter_size=(5, 5), pad='same', b=None)
    #         nnet.addSigmoidLayer()
    #
    #     if Cfg.cifar10_architecture == 2:
    #
    #         if Cfg.weight_dict_init & (not nnet.pretrained):
    #             # initialize first layer filters by atoms of a dictionary
    #             W1_init = learn_dictionary(nnet.data._X_train, 16, 5, n_sample=500)
    #             plot_mosaic(W1_init, title="First layer filters initialization",
    #                         canvas="black",
    #                         export_pdf=(Cfg.xp_path + "/filters_init"))
    #
    #         nnet.addInputLayer(shape=(None, 3, 32, 32))
    #
    #         if Cfg.weight_dict_init & (not nnet.pretrained):
    #             nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=16, filter_size=(5, 5), pad='same',
    #                               W=W1_init, b=None)
    #         else:
    #             nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=16, filter_size=(5, 5), pad='same',
    #                               b=None)
    #         if Cfg.leaky_relu:
    #             nnet.addLeakyReLU()
    #         else:
    #             nnet.addReLU()
    #         nnet.addMaxPool(pool_size=(2, 2))
    #
    #         nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=32, filter_size=(5, 5), pad='same', b=None)
    #         if Cfg.leaky_relu:
    #             nnet.addLeakyReLU()
    #         else:
    #             nnet.addReLU()
    #         nnet.addMaxPool(pool_size=(2, 2))
    #
    #         nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=64, filter_size=(5, 5), pad='same', b=None)
    #         if Cfg.leaky_relu:
    #             nnet.addLeakyReLU()
    #         else:
    #             nnet.addReLU()
    #         nnet.addMaxPool(pool_size=(2, 2))
    #
    #         # Code Layer
    #         nnet.addDenseLayer(num_units=Cfg.cifar10_rep_dim, b=None)
    #         nnet.setFeatureLayer()  # set the currently highest layer to be the SVDD feature layer
    #         nnet.addReshapeLayer(shape=([0], (Cfg.cifar10_rep_dim / 16), 4, 4))
    #         if Cfg.leaky_relu:
    #             nnet.addLeakyReLU()
    #         else:
    #             nnet.addReLU()
    #
    #         nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=64, filter_size=(5, 5), pad='same', b=None)
    #         if Cfg.leaky_relu:
    #             nnet.addLeakyReLU()
    #         else:
    #             nnet.addReLU()
    #         nnet.addUpscale(scale_factor=(2, 2))
    #
    #         nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=32, filter_size=(5, 5), pad='same', b=None)
    #         if Cfg.leaky_relu:
    #             nnet.addLeakyReLU()
    #         else:
    #             nnet.addReLU()
    #         nnet.addUpscale(scale_factor=(2, 2))
    #
    #         nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=16, filter_size=(5, 5), pad='same', b=None)
    #         if Cfg.leaky_relu:
    #             nnet.addLeakyReLU()
    #         else:
    #             nnet.addReLU()
    #         nnet.addUpscale(scale_factor=(2, 2))
    #
    #         nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=3, filter_size=(5, 5), pad='same', b=None)
    #         nnet.addSigmoidLayer()
    #
    #     if Cfg.cifar10_architecture == 3:
    #
    #         if Cfg.weight_dict_init & (not nnet.pretrained):
    #             # initialize first layer filters by atoms of a dictionary
    #             W1_init = learn_dictionary(nnet.data._X_train, 32, 5, n_sample=500)
    #             plot_mosaic(W1_init, title="First layer filters initialization",
    #                         canvas="black",
    #                         export_pdf=(Cfg.xp_path + "/filters_init"))
    #
    #         nnet.addInputLayer(shape=(None, 3, 32, 32))
    #
    #         if Cfg.weight_dict_init & (not nnet.pretrained):
    #             nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=32, filter_size=(5, 5), pad='same',
    #                               W=W1_init, b=None)
    #         else:
    #             nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=32, filter_size=(5, 5), pad='same',
    #                               b=None)
    #         if Cfg.leaky_relu:
    #             nnet.addLeakyReLU()
    #         else:
    #             nnet.addReLU()
    #         nnet.addMaxPool(pool_size=(2, 2))
    #
    #         nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=64, filter_size=(5, 5), pad='same', b=None)
    #         if Cfg.leaky_relu:
    #             nnet.addLeakyReLU()
    #         else:
    #             nnet.addReLU()
    #         nnet.addMaxPool(pool_size=(2, 2))
    #
    #         nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=128, filter_size=(5, 5), pad='same',
    #                           b=None)
    #         if Cfg.leaky_relu:
    #             nnet.addLeakyReLU()
    #         else:
    #             nnet.addReLU()
    #         nnet.addMaxPool(pool_size=(2, 2))
    #
    #         # Code Layer
    #         nnet.addDenseLayer(num_units=Cfg.cifar10_rep_dim, b=None)
    #         nnet.setFeatureLayer()  # set the currently highest layer to be the SVDD feature layer
    #         nnet.addReshapeLayer(shape=([0], (Cfg.cifar10_rep_dim / 16), 4, 4))
    #         if Cfg.leaky_relu:
    #             nnet.addLeakyReLU()
    #         else:
    #             nnet.addReLU()
    #
    #         nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=128, filter_size=(5, 5), pad='same', b=None)
    #         if Cfg.leaky_relu:
    #             nnet.addLeakyReLU()
    #         else:
    #             nnet.addReLU()
    #         nnet.addUpscale(scale_factor=(2, 2))
    #
    #         nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=64, filter_size=(5, 5), pad='same', b=None)
    #         if Cfg.leaky_relu:
    #             nnet.addLeakyReLU()
    #         else:
    #             nnet.addReLU()
    #         nnet.addUpscale(scale_factor=(2, 2))
    #
    #         nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=32, filter_size=(5, 5), pad='same', b=None)
    #         if Cfg.leaky_relu:
    #             nnet.addLeakyReLU()
    #         else:
    #             nnet.addReLU()
    #         nnet.addUpscale(scale_factor=(2, 2))
    #
    #         nnet.addConvLayer(use_batch_norm=Cfg.use_batch_norm, num_filters=3, filter_size=(5, 5), pad='same', b=None)
    #         nnet.addSigmoidLayer()


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

        autoencoder.add(Dense(128))
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

        CIFAR_10_DataLoader.mean_square_error_dict.update({lamda: meanSq_error})
        print("\n Mean square error Score ((Xclean, Xdecoded):")
        print(CIFAR_10_DataLoader.mean_square_error_dict.values())

        return CIFAR_10_DataLoader.mean_square_error_dict

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
        # opt = SGD(lr=0.01, decay=0.01 / 150, momentum=0.9, nesterov=True)
        # opt =RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        opt =Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.cae.compile(loss=self.custom_rcae_loss(), optimizer=opt)
        # self.cae.compile(optimizer='adam', loss='mean_squared_error')


        self.lamda[0] = lamda
        X_N = np.reshape(X_N, (len(X_N), 32,32,3))
        Xclean = np.reshape(Xclean, (len(Xclean), 32,32,3))

        history = self.cae.fit(X_N, X_N,
                                          epochs=500,
                                          batch_size=200,
                                          shuffle=True,
                                          validation_split=0.1,
                                          verbose=1
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
        io.imsave(self.rcae_results + '/best/' + str(lamda) + "_"+ str(Cfg.cifar10_normal)+'_RCAE.png', img)

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
        io.imsave(self.rcae_results + '/worst/' + str(lamda) + "_" +str(Cfg.cifar10_normal)+'_RCAE.png', img)

        return

    # def visualise_anamolies_detected(self,testX, noisytestX, decoded, N, best_top10_keys, worst_top10_keys, lamda):
    #     side = 32
    #     channel = 3
    #     N = np.reshape(N, (len(N), 32,32,3))
    #     # Display the decoded Original, noisy, reconstructed images
    #     print("side:", side)
    #     print("channel:", channel)
    #     img = np.ndarray(shape=(side * 3, side * 10, channel))
    #     print
    #     "img shape:", img.shape
    #
    #     best_top10_keys = list(best_top10_keys)
    #     worst_top10_keys = list(worst_top10_keys)
    #
    #     for i in range(10):
    #         row = i // 10 * 3
    #         col = i % 10
    #         img[side * row:side * (row + 1), side * col:side * (col + 1), :] = testX[best_top10_keys[i]]
    #         img[side * (row + 1):side * (row + 2), side * col:side * (col + 1), :] = decoded[best_top10_keys[i]]
    #         img[side * (row + 2):side * (row + 3), side * col:side * (col + 1), :] = N[best_top10_keys[i]]
    #         # img[side * (row + 3):side * (row + 4), side * col:side * (col + 1), :] = N[best_top10_keys[i]]
    #
    #     img *= 255
    #     img = img.astype(np.uint8)
    #     img = np.reshape(img, (side * 3, side * 10))
    #     # Save the image decoded
    #     print("\nSaving results for best after being encoded and decoded: @")
    #     print(self.rcae_results + '/best/')
    #     io.imsave(self.rcae_results  + '/best/' + str(lamda) + 'salt_p_denoising_cae_decode.png', img)
    #
    #     # Display the decoded Original, noisy, reconstructed images for worst
    #     img = np.ndarray(shape=(side * 3, side * 10, channel))
    #     for i in range(10):
    #         row = i // 10 * 3
    #         col = i % 10
    #         img[side * row:side * (row + 1), side * col:side * (col + 1), :] = testX[worst_top10_keys[i]]
    #         img[side * (row + 1):side * (row + 2), side * col:side * (col + 1), :] = decoded[worst_top10_keys[i]]
    #         img[side * (row + 2):side * (row + 3), side * col:side * (col + 1), :] = N[worst_top10_keys[i]]
    #         # img[side * (row + 3):side * (row + 4), side * col:side * (col + 1), :] = N[worst_top10_keys[i]]
    #
    #     img *= 255
    #     img = img.astype(np.uint8)
    #     img = np.reshape(img, (side * 3, side * 10))
    #     # Save the image decoded
    #     print("\nSaving results for worst after being encoded and decoded: @")
    #     print(self.rcae_results + '/worst/')
    #     io.imsave(self.rcae_results + '/worst/' + str(lamda) + 'salt_p_denoising_cae_decode.png', img)
    #
    #     return

    def evalPred(self,predX, trueX, trueY):

        trueX = np.reshape(trueX, (len(trueX), 3072))
        predX = np.reshape(predX, (len(predX), 3072))

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
            # XTrue = np.reshape(XTrue, (len(XTrue), 32,32,3))


            decoded = self.cae.predict(X_test)

            # compute MeanSqared error metric
            self.compute_mse(X_test, decoded, lamda)
            # print("[INFO:] The anomaly threshold computed is ", self.anomaly_threshold)

            # rank the best and worst reconstructed images
            [best_top10_keys, worst_top10_keys] = self.compute_best_worst_rank(X_test, decoded)

            # Visualise the best and worst ( image, BG-image, FG-Image)
            # XPred = np.reshape(np.asarray(decoded), (len(decoded), 32,32,3))
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

        img *= 255
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

        img *= 255
        img = img.astype(np.uint8)

        # Save the image decoded
        print("\nSaving results for worst after being encoded and decoded: @")
        print(save_results + '/worst/')
        io.imsave(save_results + '/worst/' + str(lamda) + '_RCAE.png', img)

        return
