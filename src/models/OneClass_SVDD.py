

# import the necessary packages
import numpy as np
from src.data.preprocessing import learn_dictionary, global_contrast_normalization
from sklearn.metrics import average_precision_score, mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.feature_extraction.image import PatchExtractor
from PIL import Image
from skimage import io
from sklearn.decomposition import MiniBatchDictionaryLearning, PCA



RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
import tensorflow as tf

sess = tf.Session()

from keras.regularizers import L1L2

customL2_regularizer = L1L2(l2=1e-6)
from keras import backend as K

K.set_session(sess)
import matplotlib.pyplot as plt
import numpy as np

from src.data.main import load_dataset
from src.config import Configuration as Cfg

import tensorflow as tf
from keras.models import model_from_json
from keras.models import Model
from keras.models import model_from_json
from keras.models import Sequential
from keras.callbacks import LambdaCallback
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD, Adagrad, Adadelta, RMSprop, Adam

from keras.models import Model, Sequential
from keras.layers import Activation, LeakyReLU, Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, \
    BatchNormalization, regularizers

from sklearn.metrics import roc_auc_score

from keras.optimizers import RMSprop

from keras.callbacks import Callback
from keras.callbacks import Callback

center = None
R_updated = None
rvalue = 0.1


class Adjust_svdd_Radius(Callback):
    def __init__(self, model, cvar, radius, X_train, modeltype,rep_dim):
        '''
        display: Number of batches to wait before outputting loss
        '''
        self.seen = 0
        self.radius = radius
        self.n_train = len(X_train)
        self.model = model
        self.inputs = X_train
        self.cvar = cvar
        self.y_reps = np.zeros((len(X_train), rep_dim))
        self.model_type = modeltype
        self.rep_dim = rep_dim

    def on_epoch_end(self, batch, logs={}):

        if (self.model_type == "OC_SVDD"):
            reps = self.model.predict(self.inputs[:len(self.inputs), :])
            reps = np.reshape(reps, (len(reps), self.rep_dim))
            self.reps = reps
            self.y_reps = self.reps
            center = self.cvar
            dist = np.sum((reps - self.cvar) ** 2, axis=1)
            scores = dist
            val = np.sort(scores)
            R_new = np.percentile(val, Cfg.nu * 100)  # qth quantile of the radius.
            R_updated = R_new
            # print("[INFO:] Center (c)  Used.", center)
            # print("[INFO:] Updated Radius (R) .", R_updated)
            self.radius = R_new
            # print("[INFO:] \n Updated Radius Value...", R_new)
            # print("[INFO:] \n Updated Rreps value..", self.y_reps)
            return self.radius

        elif (self.model_type == "OC_NN"):
            reps = self.model.predict(self.inputs[:len(self.inputs), :])
            # reps = np.reshape(reps, (len(reps), 32))
            # print("[INFO:] The OCNN - reps shape is ", reps.shape)
            self.reps = reps
            # self.y_reps = self.reps
            center = self.cvar
            dist = np.sum((reps - self.cvar) ** 2, axis=1)
            scores = dist
            val = np.sort(scores)
            R_new = np.percentile(val, Cfg.nu * 100)  # qth quantile of the radius.
            R_updated = R_new

            # R_new = np.percentile(reps, Cfg.nu * 100)  # qth quantile of the radius.
            self.rvalue = R_new
            self.radius = R_new
            # print("[INFO:] \n Updated R Value for OCNN...", self.rvalue)
            # print("[INFO:] \n Center Value used  for OCNN...", self.cvar)


class OneClass_SVDD:
    ## Initialise static variables
    INPUT_DIM = 0
    HIDDEN_SIZE = 0
    DATASET = "mnist"
    mean_square_error_dict = {}
    RESULT_PATH = ""

    def __init__(self, dataset, lossfuncType, inputdim, hiddenLayerSize, img_hgt, img_wdt, img_channel, modelSavePath,
                 reportSavePath,
                 preTrainedWtPath, seed, intValue=0, stringParam="defaultValue",
                 otherParam=None):
        """
        Called when initializing the classifier
        """
        OneClass_SVDD.DATASET = dataset
        OneClass_SVDD.INPUT_DIM = inputdim
        OneClass_SVDD.HIDDEN_SIZE = hiddenLayerSize
        OneClass_SVDD.RESULT_PATH = reportSavePath
        Cfg.seed = seed
        print("OneClass_SVDD.RESULT_PATH:",OneClass_SVDD.RESULT_PATH)
        self.intValue = intValue
        self.stringParam = stringParam

        # THIS IS WRONG! Parameters should have same name as attributes
        self.differentParam = otherParam

        self.directory = modelSavePath
        self.results = reportSavePath
        self.pretrainedWts = preTrainedWtPath
        self.model = ""
        self.lossfunctype = lossfuncType

        self.IMG_HGT = img_hgt
        self.IMG_WDT = img_wdt
        self.channel = img_channel
        self.h_size = OneClass_SVDD.HIDDEN_SIZE
        global model
        self.r = 1.0
        self.kvar = 0.0
        self.pretrain = True
        self.load_dcae_path = ""
        # self.num_decoder_layers = 14
        self.num_decoder_layers = 9
        self.Rvar = 0.0  # Radius
        self.cvar = 0.0  # center which represents the mean of the representations
        self.l2_penalty_wts = None

        # load dataset
        load_dataset(self, dataset.lower(), self.pretrain)

        # Depending on the dataset build respective autoencoder architecture
        if (dataset.lower() == "mnist"):
            from src.data.mnist import MNIST_DataLoader
            # Create the robust cae for the dataset passed
            self.prj_path = "/content/drive/My Drive/2018/Colab_Deep_Learning/one_class_neural_networks/"
            # self.prj_path = "/Users/raghav/envPython3/experiments/one_class_neural_networks/"
            self.nn_model = MNIST_DataLoader()
            self.load_dcae_path = self.prj_path + "/models/MNIST/RCAE/MNIST/"
            self.lossfuncType = lossfuncType
            self.n_train = len(self.data._X_train)
            self.val = np.ones(Cfg.mnist_rep_dim) * 0.5
        # Depending on the dataset build respective autoencoder architecture
        
        if (dataset.lower() == "cifar10"):
            from src.data.cifar10 import CIFAR_10_DataLoader
            # Create the robust cae for the dataset passed
            self.prj_path = "/content/drive/My Drive/2018/Colab_Deep_Learning/one_class_neural_networks/"
            # self.prj_path = "/Users/raghav/envPython3/experiments/one_class_neural_networks/"
            self.nn_model = CIFAR_10_DataLoader()
            self.load_dcae_path = self.prj_path + "/models/CIFAR10/OC_NN/"
            self.lossfuncType = lossfuncType
            self.n_train = len(self.data._X_train)
            # self.val = np.ones(Cfg.mnist_rep_dim) * 0.5
            
        if (dataset.lower() == "gtsrb"):
            from src.data.GTSRB import GTSRB_DataLoader
            # Create the robust cae for the dataset passed
            self.prj_path = "/content/drive/My Drive/2018/Colab_Deep_Learning/one_class_neural_networks/"
            # self.prj_path = "/Users/raghav/envPython3/experiments/one_class_neural_networks/"
            self.nn_model = GTSRB_DataLoader()
            self.load_dcae_path = self.prj_path + "/models/GTSRB/OC_NN/"
            self.lossfuncType = lossfuncType
            self.n_train = len(self.data._X_train)
            # self.val = np.ones(Cfg.mnist_rep_dim) * 0.5
        
        if (dataset.lower() == "lhc"):
            from src.data.cifar10 import LHC_DataLoader
            # Create the robust cae for the dataset passed
            self.prj_path = "/content/drive/My Drive/2018/Colab_Deep_Learning/one_class_neural_networks/"
            # self.prj_path = "/Users/raghav/envPython3/experiments/one_class_neural_networks/"
            self.nn_model = LHC_DataLoader()
            self.load_dcae_path = self.prj_path + "/models/LHC/OC_NN/"
            self.lossfuncType = lossfuncType
            self.n_train = len(self.data._X_train)
            # self.val = np.ones(Cfg.mnist_rep_dim) * 0.5

    def load_data(self, data_loader=None, pretrain=False):
        self.data = data_loader()
        return

    def learn_dictionary(X, n_filters, filter_size, n_sample=1000,
                         n_sample_patches=0, **kwargs):
        """
        learn a dictionary of n_filters atoms from n_sample images from X
        """
        print("[INFO] : learn a dictionary of n_filters atoms from n_sample images from X.")
        n_channels = X.shape[1]
        print("[INFO] : Dimension of  X.", X.shape)
        print("[INFO] : Number of channels X.", n_channels)
        # subsample n_sample images randomly
        rand_idx = np.random.choice(len(X), n_sample, replace=False)

        # extract patches
        patch_size = (filter_size, filter_size)
        patches = PatchExtractor(patch_size).transform(
            X[rand_idx, ...].reshape(n_sample, X.shape[2], X.shape[3], X.shape[1]))
        patches = patches.reshape(patches.shape[0], -1)
        patches -= np.mean(patches, axis=0)
        patches /= np.std(patches, axis=0)

        if n_sample_patches > 0 and (n_sample_patches < len(patches)):
            np.random.shuffle(patches)
            patches = patches[:n_sample_patches, ...]

        # learn dictionary
        print('Learning dictionary for weight initialization...')

        dico = MiniBatchDictionaryLearning(n_components=n_filters, alpha=1, n_iter=1000, batch_size=10, shuffle=True,
                                           verbose=True, **kwargs)
        W = dico.fit(patches).components_
        W = W.reshape(n_filters, n_channels, filter_size, filter_size)

        # print('Dictionary learned.')
        print('[INFO:] Dictionary Learned', W.shape)

        return W.astype(np.float32)

    def save_model(self, model, path):

        ## save the model
        # serialize model to JSON
        if (OneClass_SVDD.DATASET == "mnist"):
            model_json = model.to_json()
            with open(path + "DCAE_DIGIT__" + str(Cfg.mnist_normal) + "__model.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights(path + "DCAE_DIGIT__" + str(Cfg.mnist_normal) + "__model.h5")
            # print("Saved model to disk....")

        return

    def get_pretrainedModelName(self):

        if (Cfg.mnist_normal == 5):
            return "DCAE_DIGIT__5__model"

        return

    def get_pretrainedEncoderWts(self):
        model_name = self.get_pretrainedModelName()

        # load json and create model

        json_file = open(self.load_dcae_path + model_name + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)

        # load weights into new model
        loaded_model.load_weights(self.load_dcae_path + model_name + '.h5')
        # print("[INFO:] Loaded pretrained DCAE model weights from disk.....")
        # loaded_model.summary(line_length=100)

        encoder = self.remove_decoder_layers(loaded_model)
        # loaded_model.summary(line_length=100)

        saved_encoderPath = self.load_dcae_path + "/encoder/"

        self.save_model(encoder, saved_encoderPath)

        return saved_encoderPath

    def get_pretrained_OC_Encoder(self):
        from keras.models import load_model
        # load weights into new model
        # load json and create model
        json_file = open(self.results + '/cae.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(self.results + "/cae.h5")
        print("[INFO:] Loaded model from disk")

        # loaded_model = load_model(self.results + '/cae.h5')
        # print("[INFO:] Loaded pretrained DCAE model weights from disk.....")
        # loaded_model.summary(line_length=100)

        encoder = self.remove_decoder_layers(loaded_model)
        # loaded_model.summary(line_length=100)

        # saved_encoderPath = self.load_dcae_path + "/encoder/"
        #
        # self.save_model(encoder, saved_encoderPath)

        return encoder

    def load_pretrained_model(self):

        base_model = self.build_svdd_network()

        saved_encoder = self.get_pretrained_OC_Encoder()

        return saved_encoder

    # copy weights from one model to another model
    # tested in Keras 1.x
    def copyModel2Model(self, model_source, model_target, certain_layer=""):
        # print("[INFO:] Length of model layers",len(model_source.layers))
        # print("[INFO:] Length of model layers", (model_source.layers))


        for l_tg, l_sr in zip(model_target.layers, model_source.layers):
            wk0 = l_sr.get_weights()
            l_tg.set_weights(wk0)
            if l_tg.name == certain_layer:
                break
        print("model source was copied into model target")

    def load_pre_trained_CAE_model(self, encoder):

        base_model = self.build_oc_svdd_network()

        # trained_encoder = self.get_pretrained_OC_Encoder()

        # Copy weights from one model to another keras model
        self.copyModel2Model(encoder, base_model)

        return base_model

    def load_pretrained_model_full(self):

        model_name = self.get_pretrainedModelName()

        json_file = open(self.load_dcae_path + model_name + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)

        # load weights into new model
        loaded_model.load_weights(self.load_dcae_path + model_name + '.h5')

        return loaded_model

    def remove_decoder_layers(self, model):

        for count in range(0, self.num_decoder_layers):
            model.layers.pop()
        encoder = model

        return encoder

    def build_OneClass_SVDD_network(self):

        # load the pretrained model and its weights
        print(
            "[INFO:]  Before removing : This might take a while Please wait .... Loading pre-trained autoencoder weights")
        model = self.load_pretrained_model()
        self.encoder = model
        # self.encoder.summary(line_length=100)

        ## Remove the decoder portion and obtain a Flattened latent representation
        # self.encoder = self.remove_decoder_layers(model)

        print("[INFO:] After removal Loaded pretrained DCAE Encoder model weights from disk.....")
        # self.encoder.summary(line_length=100)

        ## Removal of decoder layer complete
        # import scipy.io
        # from sklearn.preprocessing import MinMaxScaler

        inp = self.encoder.input
        out = self.encoder.layers[-1].output
        # from keras.models import Model
        # enoder_model_svdd = Model(inp, out)  # create a new model which doesn't have the last two layers in VGG16
        # mode_ocnn_svdd = Sequential()  # new model
        #
        # for layer in enoder_model_svdd.layers:
        #     mode_ocnn_svdd.add(layer)

        # self.encoder.summary(line_length=100)

        model_svdd = Model(inputs=inp,
                           outputs=out)

        print("[INFO:] The one class svdd model build complete... ")

        # ## Add L2 regulizers to each of the layer weights
        # from keras.regularizers import L1L2
        # customL2_regularizer = L1L2(l2=0.5)
        # for layer in model_svdd.layers:
        #     if hasattr(layer, 'kernel_regularizer'):
        #         layer.kernel_regularizer = customL2_regularizer

        # https: // github.com / keras - team / keras / issues / 2717

        return model_svdd

    def build_OC_SVDD_network(self, cae):

        # load the pretrained model and its weights
        print(
            "[INFO:]  Before removing : This might take a while Please wait .... Loading pre-trained autoencoder weights")
        encoder = self.remove_decoder_layers(cae)
        # model = self.load_pre_trained_CAE_model(encoder)
        self.encoder = encoder
        # self.encoder.summary(line_length=100)

        ## Remove the decoder portion and obtain a Flattened latent representation


        print("[INFO:] After removal Loaded pretrained DCAE Encoder model weights from disk.....")
        # self.encoder.summary(line_length=100)

        ## Removal of decoder layer complete
        # import scipy.io
        # from sklearn.preprocessing import MinMaxScaler

        # inp = self.encoder.input
        # out = self.encoder.layers[-1].output
        # from keras.models import Model
        # enoder_model_svdd = Model(inp, out)  # create a new model which doesn't have the last two layers in VGG16
        # mode_ocnn_svdd = Sequential()  # new model
        #
        # for layer in enoder_model_svdd.layers:
        #     mode_ocnn_svdd.add(layer)

        # self.encoder.summary(line_length=100)

        # model_svdd = Model(inputs=inp,
        #                    outputs=out)

        print("[INFO:] The one class svdd model build complete... ")

        # ## Add L2 regulizers to each of the layer weights
        # from keras.regularizers import L1L2
        # customL2_regularizer = L1L2(l2=0.5)
        # for layer in model_svdd.layers:
        #     if hasattr(layer, 'kernel_regularizer'):
        #         layer.kernel_regularizer = customL2_regularizer

        # https: // github.com / keras - team / keras / issues / 2717
        model_svdd = self.encoder
        return model_svdd

    # use the pretrained model saved to create the network with SVDD loss
    def get_OneClass_SVDD_network_reps(self, inputs):

        # Build the SVDD one class model
        self.model_svdd = self.build_OneClass_SVDD_network()

        self.reps = self.model_svdd.predict(inputs[:len(inputs), :])

        print("[INFO:] Obtained the initial representations of input using pretrained weights ")
        return self.reps

    def get_OC_SVDD_network_reps(self, inputs, encoder):

        # Build the SVDD one class model
        # self.model_svdd = self.build_OC_SVDD_network(cae)
        self.model_svdd = encoder

        self.reps = self.model_svdd.predict(inputs[:len(inputs), :])

        print("[INFO:] Obtained the initial representations of input using pretrained weights ")
        return self.reps

    # Compute the center as mean embeddings and initialize Radius-R
    def initialize_c_as_mean(self, inputs, n_batches, eps=0.1):
        """
        initialize c as the mean of the final layer representations from all samples propagated in n_batches
        """

        reps = self.get_OneClass_SVDD_network_reps(inputs)
        self.reps = reps

        print("[INFO:] Initializing c and Radius R value...")

        # consider the value all the number of batches (and thereby samples) to initialize from
        c = np.mean(reps, axis=0)

        # If c_i is too close to 0 in dimension i, set to +-eps.
        # Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        self.cvar = c  # Initialize the center

        # initialize R at the (1-nu)-th quantile of distances
        dist_init = np.sum((reps - c) ** 2, axis=1)
        out_idx = int(np.floor(len(reps) * Cfg.nu))
        sort_idx = dist_init.argsort()
        self.Rvar = Cfg.floatX(dist_init[sort_idx][-out_idx])

        # print("[INFO:] Center (c)  initialized.", c)
        # print("[INFO:] Shape of Center (c)  initialized.", c.shape)
        # print("[INFO:] Distances (D)  initialized.", dist_init)
        # print("[INFO:] Shape of Distances (D)  initialized.", dist_init.shape)
        # print("[INFO:] out_idx (D)  out_idx.", out_idx)
        # print("[INFO:] sort_idx (D)  sort_idx.", sort_idx)
        # print("[INFO:] Radius (R)  initialized.", Cfg.floatX(dist_init[sort_idx][-out_idx]))

    # set learning rate decay for training SVDD
    def decay_learning_rate(epoch):
        """
        decay the learning rate after epoch specified in Cfg.lr_decay_after_epoch
        """

        # only allow decay for non-adaptive solvers
        assert Cfg.nn_solver in ("sgd", "momentum", "adam")

        if epoch >= Cfg.lr_decay_after_epoch:
            lr_new = (Cfg.lr_decay_after_epoch / Cfg.floatX(epoch)) * Cfg.learning_rate_init
            return Cfg.floatX(lr_new)
        else:
            return Cfg.floatX(Cfg.learning_rate_init)

    # adjust learning_rate drop and finetune
    def adjust_learning_rate_finetune(epoch):

        if Cfg.lr_drop and (epoch == Cfg.lr_drop_in_epoch):
            # Drop the learning rate in epoch specified in Cfg.lr_drop_after_epoch by factor Cfg.lr_drop_factor
            # Thus, a simple separation of learning into a "region search" and "finetuning" stage.
            lr_new = Cfg.floatX((1.0 / Cfg.lr_drop_factor) * Cfg.learning_rate)
            print("")
            print("Learning rate drop in epoch {} from {:.6f} to {:.6f}".format(
                epoch, Cfg.floatX(Cfg.learning_rate), lr_new))
            print("")
            Cfg.learning_rate = lr_new

        return lr_new

    def get_oneClass_trainData(self):
        # X_train = np.concatenate((self.data._X_train, self.data._X_val))
        # y_train = np.concatenate((self.data._y_train, self.data._y_val))
        if(OneClass_SVDD.DATASET== "mnist"):
            X_train = self.data._X_train
            y_train = self.data._y_train
    
            X_test = self.data._X_test
            y_test = self.data._y_test
    
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
    
            X_trainPOS = np.concatenate((trainXPos, testXPos))
            y_trainPOS = np.concatenate((trainYPos, testYPos))
    
            X_trainNEG = np.concatenate((trainXNeg, testXNeg))
            y_trainNEG = np.concatenate((trainYNeg, testYNeg))
    
            # Just 0.01 points are the number of anomalies.
            num_of_anomalies = int(0.01 * len(X_trainPOS))
            X_trainNEG = X_trainNEG[0:num_of_anomalies]
            y_trainNEG = y_trainNEG[0:num_of_anomalies]
    
            X_train = np.concatenate((X_trainPOS, X_trainNEG))
            y_train = np.concatenate((y_trainPOS, y_trainNEG))
    
            print("[INFO: ] Shape of One Class Input Data used in training", X_train.shape)
            print("[INFO: ] Shape of (Positive) One Class Input Data used in training", X_trainPOS.shape)
            print("[INFO: ] Shape of (Negative) One Class Input Data used in training", X_trainNEG.shape)
            
            return X_train

            
        elif(OneClass_SVDD.DATASET == "cifar10"):
            X_train = self.data._X_train
            y_train = self.data._y_train
            
            X_test = X_train
            y_test = y_train
            
            
            return X_train
            
        
        elif(OneClass_SVDD.DATASET == "gtsrb"):
                
                # X_train = np.concatenate((self.data._X_train, self.data._X_test))
                # y_train = np.concatenate((self.data._y_train, self.data._y_test))
                
                # X_train = self.data._X_train
                # y_train = self.data._y_train
            
                X_train = self.data._X_train
                y_train = self.data._y_train
        
                # X_test = self.data._X_test
                # y_test = self.data._y_test
                
                # # Make sure the axis dimensions are aligned for training convolutional autoencoders
               
                
                # print("X_train,X_test====>",X_train.shape, X_test.shape)
              
                # X_train = X_train/255.0
                # X_test = X_test / 255.0

        
                # ## Combine the positive data
                # trainXPos = X_train[np.where(y_train == 0)]
                # trainYPos = np.zeros(len(trainXPos))
                # testXPos = X_test[np.where(y_test == 0)]
                # testYPos = np.zeros(len(testXPos))
        
        
                # # Combine the negative data
                # trainXNeg = X_train[np.where(y_train == 1)]
                # trainYNeg = np.ones(len(trainXNeg))
                # testXNeg = X_test[np.where(y_test == 1)]
                # testYNeg = np.ones(len(testXNeg))

                # print("trainXPos,testXPos",trainXPos.shape, testXPos.shape)
                # X_trainPOS = np.concatenate((trainXPos, testXPos))
                # y_trainPOS = np.concatenate((trainYPos, testYPos))
        
                # X_trainNEG = np.concatenate((trainXNeg, testXNeg))
                # y_trainNEG = np.concatenate((trainYNeg, testYNeg))
        
                # # Just 0.01 points are the number of anomalies.
                # num_of_anomalies = int(0.1 * len(X_trainPOS))
        
                # X_trainNEG = X_trainNEG[0:num_of_anomalies]
                # y_trainNEG = y_trainNEG[0:num_of_anomalies]
        
        
                # X_train = np.concatenate((X_trainPOS, X_trainNEG))
                # y_train = np.concatenate((y_trainPOS, y_trainNEG))
        
                
                # self.data._X_test = X_train
                # self.data._y_test = y_train
                
                print(" [INFO:]  The shape  of  training data ----",X_train.shape)
                return X_train

        elif(OneClass_SVDD.DATASET == "lhc"):
            X_train = self.data._X_train
            y_train = self.data._y_train
            
            X_test = X_train
            y_test = y_train
            
            return X_train
        
        
        if (self.lossfuncType == "ONE_CLASS_NEURAL_NETWORK"):
            X_train = X_train[np.where(y_train == 0)]
            return X_train

        return 

    def get_oneClass_testData(self):

        if(OneClass_SVDD.DATASET == "mnist"):
            
            X_train = self.data._X_train
            y_train = self.data._y_train
    
            X_test = self.data._X_test
            y_test = self.data._y_test
    
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
            
            
    
            X_testPOS = np.concatenate((trainXPos, testXPos))
            y_testPOS = np.concatenate((trainYPos, testYPos))
    
            X_testNEG = np.concatenate((trainXNeg, testXNeg))
            y_testNEG = np.concatenate((trainYNeg, testYNeg))
    
            # Just 0.01 points are the number of anomalies.
            num_of_anomalies = int(0.01 * len(X_testPOS))
            X_testNEG = X_testNEG[0:num_of_anomalies]
            y_testNEG = y_testNEG[0:num_of_anomalies]
    
            X_test = np.concatenate((X_testPOS, X_testNEG))
            y_test = np.concatenate((y_testPOS, y_testNEG))
    
            PosBoundary = len(X_testPOS)
            NegBoundary = len(X_testNEG)
    
            print("[INFO: ] Shape of One Class Input Data used in testing", X_test.shape)
            print("[INFO: ] Shape of (Positive) One Class Input Data used in testing", X_testPOS.shape)
            print("[INFO: ] Shape of (Negative) One Class Input Data used in testing", X_testNEG.shape)
            
            return [X_test, y_test, PosBoundary, NegBoundary,X_test]
            
        elif(OneClass_SVDD.DATASET == "cifar10"):
            X_train = self.data._X_train
            y_train = self.data._y_train
            
            X_test = X_train
            y_test = y_train
            
            
            X_test_beforegcn = self.data._X_test_beforegcn
            
            testXNeg = X_test[np.where(y_test == 1)]
            testXPos = X_test[np.where(y_test == 0)]
            
            PosBoundary = len(testXPos)
            NegBoundary = len(testXNeg)
            print("[INFO: ] Shape of One Class Input Data used in testing", X_test.shape)
            print("[INFO: ] Shape of (Positive) One Class Input Data used in testing", testXPos.shape)
            print("[INFO: ] Shape of (Negative) One Class Input Data used in testing", testXNeg.shape)
            
            return [X_test, y_test, PosBoundary, NegBoundary,X_test_beforegcn]
            
        
        elif(OneClass_SVDD.DATASET == "gtsrb"):
                
                # X_train = np.concatenate((self.data._X_train, self.data._X_test))
                # y_train = np.concatenate((self.data._y_train, self.data._y_test))
                
                # # X_train = self.data._X_train
                # # y_train = self.data._y_train
            
                # X_test = X_train
                # y_test = y_train
                # # X_test = self.data._X_test
                # # y_test = self.data._y_test
    
                
                # X_train = np.concatenate((trainXPos, trainXNeg))
                # y_train = np.concatenate((trainYPos, trainYNeg))
                
                # X_test = X_train
                
                
                # # Make sure the axis dimensions are aligned for training convolutional autoencoders
                # # X_train = np.moveaxis(X_train, 1, 3)
                # # X_test = np.moveaxis(X_test, 1, 3)
              
                # X_train = X_train/255.0
                # X_test = X_test / 255.0
                
                
                
                # print("INFO: The self.data._X_val ",self.data._X_val.shape)
                # X_test = self.data._X_val
                # y_test = y_train
                
                X_test =  self.data._X_test 
                y_test = self.data._y_test 
                
                testXPos = X_test[np.where(y_test == 0)]
                testYPos = np.zeros(len(testXPos))
                testXNeg = X_test[np.where(y_test == 1)]
                testYNeg = 1 * np.ones(len(testXNeg))
    
                PosBoundary = len(testXPos)
                NegBoundary = len(testXNeg)
                
    
                print("[INFO:]  Length of Positive data", len(testXPos))
                print("[INFO:]  Length of Negative data", len(testXNeg))
    
                
                return [X_test, y_test, PosBoundary, NegBoundary,X_test]

        elif(OneClass_SVDD.DATASET == "lhc"):
            X_train = self.data._X_train
            y_train = self.data._y_train
            
            X_test = X_train
            y_test = y_train
            
            PosBoundary = len(testXPos)
            NegBoundary = len(testXNeg)
            print("[INFO: ] Shape of One Class Input Data used in testing", X_test.shape)
            print("[INFO: ] Shape of (Positive) One Class Input Data used in testing", testXPos.shape)
            print("[INFO: ] Shape of (Negative) One Class Input Data used in testing", testXNeg.shape)
            
            return [X_test, y_test, PosBoundary, NegBoundary]

        return 

    def adjust_svdd_radius(self):

        if Cfg.svdd_loss and (Cfg.hard_margin or (Cfg.block_coordinate)):
            # set R to be the (1-nu)-th quantile of distances

            out_idx = int(np.floor(self.n_train * Cfg.nu))
            dist = K.sum(((self.reps - self.cvar) ** 2), axis=1)
            scores = dist - self.Rvar
            scores = scores.argsort()
            R_new = scores[-out_idx] + self.Rvar
            self.Rvar = R_new

        print("[INFO:] The shape of scores", scores)
        radius = self.Rvar

        return radius

    def get_l2_penalty_wts(self, model, pow=2):
        """
        returns the l2 penalty on (trainable) network parameters combined as sum
        """

        l2_penalty = 0

        for layer in model.layers:
            l2_penalty = l2_penalty + K.sum(K.abs(layer.get_weights()) ** pow)

        C = Cfg.C

        # Network weight decay
        if Cfg.weight_decay:
            self.l2_penalty_wts = (1 / C) * l2_penalty
            l2_penalty = self.l2_penalty_wts
        else:
            l2_penalty = K.cast(0, dtype='floatX')

        return l2_penalty

    # Custom loss SVDD_loss ball interpretation
    def custom_ocnn_hypershere_loss(self):

        center = self.cvar

        # val = np.ones(Cfg.mnist_rep_dim) * 0.5
        # center = K.variable(value=val)


        # define custom_obj_ball
        def custom_obj_ball(y_true, y_pred):
            # compute the distance from center of the circle to the

            dist = (K.sum(K.square(y_pred - center), axis=1))
            avg_distance_to_c = K.mean(dist)

            return (avg_distance_to_c)

        return custom_obj_ball

    def custom_ocnn_hyperplane_loss(self):

        r = rvalue
        center = self.cvar
        # w = self.oc_nn_model.layers[-2].get_weights()[0]
        # V = self.oc_nn_model.layers[-1].get_weights()[0]
        # print("Shape of w",w.shape)
        # print("Shape of V",V.shape)
        nu = Cfg.nu

        def custom_hinge(y_true, y_pred):
            # term1 = 0.5 * tf.reduce_sum(w ** 2)
            # term2 = 0.5 * tf.reduce_sum(V ** 2)

            term3 =   K.square(r) + K.sum( K.maximum(0.0,    K.square(y_pred -center) - K.square(r)  ) , axis=1 )
            # term3 = K.square(r) + K.sum(K.maximum(0.0, K.square(r) - K.square(y_pred - center)), axis=1)
            term3 = 1 / nu * K.mean(term3)

            loss = term3

            return (loss)

        return custom_hinge

    def build_oc_svdd_network(self):

        # initialize the model
        svdd_network = Sequential()
        inputShape = (28, 28, 1)
        chanDim = -1  # since depth is appearing the end
        # first set of CONV => RELU => POOL layers

        svdd_network.add(Conv2D(16, (3, 3), padding="same", input_shape=inputShape))
        svdd_network.add(Activation("relu"))
        svdd_network.add(BatchNormalization(axis=chanDim))
        svdd_network.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        svdd_network.add(Conv2D(8, (3, 3), padding="same", input_shape=inputShape))
        svdd_network.add(Activation("relu"))
        svdd_network.add(BatchNormalization(axis=chanDim))
        svdd_network.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        svdd_network.add(Conv2D(8, (3, 3), padding="same", input_shape=inputShape))
        svdd_network.add(Activation("relu"))
        svdd_network.add(BatchNormalization(axis=chanDim))
        svdd_network.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        svdd_network.add(Conv2D(8, (3, 3), padding="same", input_shape=inputShape))
        svdd_network.add(Activation("relu"))
        svdd_network.add(BatchNormalization(axis=chanDim))
        svdd_network.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        return svdd_network

    def build_svdd_network(self):

        def my_init(shape, dtype=None):
            W1_init = learn_dictionary(self.data._X_train, 20, 5, n_sample=500)
            # print("self.data._X_train",self.data._X_train.shape)
            # print("W1_init.shape",W1_init.shape)
            W1_init = np.reshape(W1_init, (5, 5, 1, 20))
            # print("Reshaped W1_init.shape", W1_init.shape)
            return W1_init
            # return W1_init

        # initialize the model
        autoencoder = Sequential()
        inputShape = (28, 28, 1)
        chanDim = -1  # since depth is appearing the end
        # first set of CONV => RELU => POOL layers

        autoencoder.add(Conv2D(20, (5, 5), padding="same", kernel_initializer=my_init, input_shape=inputShape))
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

        return autoencoder

    def build_ocnn_network(self, encoder):

        inp = encoder.input
        out = encoder.layers[-1].output
        from keras.models import Model
        enoder_model_svdd = Model(inp, out)  # create a new model which doesn't have the last two layers in VGG16

        ocnn_model = Sequential()  # new model

        for layer in enoder_model_svdd.layers:
            ocnn_model.add(layer)

        ## Set the layers of OC-NN encoders as non-trainable

        # for layer in ocnn_model.layers:
        #     layer.trainable = False
        # print("Set the layers to be non trainable....")



        ## Add one class neural network
        # ocnn_model.add(Flatten())
        # ocnn_model.add(Dense(32,activation='linear'))
        # ocnn_model.add(Dense(16,activation='linear'))
        # ocnn_model.add(Dense(1,activation='linear'))
        # print("Set Linear activation.....")

        # ocnn_model.summary(line_length=100)

        return ocnn_model

        # Call fit function for each of the batches with svdd_loss

    def encoder(self, input_img):
        # encoder
        # input = 28 x 28 x 1 (wide and thin)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', use_bias=True)(input_img)  # 28 x 28 x 32
        conv1 = BatchNormalization(axis=-1)(conv1)

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

    def decoder(self, conv4):
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

    def compile_autoencoder(self):

        def my_init(shape, dtype=None):
            W1_init = learn_dictionary(self.data._X_train, 16, 3, n_sample=500)
            # print("self.data._X_train",self.data._X_train.shape)
            # print("W1_init.shape",W1_init.shape)
            W1_init = np.reshape(W1_init, (3, 3, 1, 16))
            # print("Reshaped W1_init.shape", W1_init.shape)
            return W1_init
            # return W1_init

        chanDim = -1  # since depth is appearing the end

        input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

        x = Conv2D(16, (3, 3), kernel_initializer=my_init, use_bias=False, padding='same')(input_img)
        x = BatchNormalization(axis=chanDim)(x)
        x = LeakyReLU(0.1)(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(8, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = LeakyReLU(0.1)(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(8, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = LeakyReLU(0.1)(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(8, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = LeakyReLU(0.1)(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)

        # at this point the representation is (4, 4, 8) i.e. 128-dimensional
        x = Conv2D(4, (3, 3), padding='same', use_bias=False)(encoded)
        x = BatchNormalization(axis=chanDim)(x)
        x = LeakyReLU(0.1)(x)
        x = UpSampling2D((2, 2))(x)

        x = Conv2D(8, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = LeakyReLU(0.1)(x)
        x = UpSampling2D((2, 2))(x)

        x = Conv2D(8, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = LeakyReLU(0.1)(x)
        x = UpSampling2D((2, 2))(x)

        x = Conv2D(16, (3, 3), use_bias=False)(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = LeakyReLU(0.1)(x)
        x = UpSampling2D((2, 2))(x)

        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same', use_bias=False)(x)

        # this model maps an input to its encoded representation
        encoder = Model(input_img, encoded)

        autoencoder = Model(input_img, decoded)
        # Compile the autoencoder with the mean squared error
        autoencoder.compile(loss='mean_squared_error', optimizer='adam')
        # print("Autoencoder Architecture", autoencoder.summary())

        return [autoencoder, encoder]

    def compile_autoencoder_cifar(self):

        def my_init(shape, dtype=None):
            W1_init = learn_dictionary(self.data._X_train, 64, 3, n_sample=500)
            # print("self.data._X_train",self.data._X_train.shape)
            # print("W1_init.shape",W1_init.shape)
            W1_init = np.reshape(W1_init, (3, 3, 3, 64))
            # print("Reshaped W1_init.shape", W1_init.shape)
            return W1_init
            # return W1_init

        chanDim = -1  # since depth is appearing the end

        input_img = Input(shape=(32, 32, 3))  # adapt this if using `channels_first` image data format
       
        x = Conv2D(128, (3, 3),  use_bias=False, padding='same')(input_img)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        
        x = Conv2D(64, (3, 3), padding='same',use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        
        x = Conv2D(32, (3, 3), padding='same',use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        
        x = Conv2D(32, (3, 3), padding='same',use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)
        
        x = Conv2D(32, (3, 3), padding='same',use_bias=False)(encoded)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        x = UpSampling2D((2, 2))(x)
        
        x = Conv2D(32, (3, 3), padding='same',use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        x = UpSampling2D((2, 2))(x)
        
        x = Conv2D(64, (3, 3), padding='same',use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        x = UpSampling2D((2, 2))(x)
        
        x = Conv2D(128, (3, 3), padding='same',use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        x = UpSampling2D((2, 2))(x)
        
        x = Conv2D(3, (3, 3), padding='same',use_bias=False)(x)
        x = BatchNormalization()(x)
        decoded = Activation('sigmoid') (x)
        
        # this model maps an input to its encoded representation
        encoder = Model(input_img, encoded)

        autoencoder = Model(input_img, decoded)
        # Compile the autoencoder with the mean squared error
        autoencoder.compile(loss='binary_crossentropy', optimizer='adam')
        # print("[INFO] : Autoencoder Architecture", autoencoder.summary())

        return [autoencoder, encoder]

    def compile_autoencoder_gtsrb(self):

       
        def my_init(shape, dtype=None):
            W1_init = learn_dictionary(self.data._X_train, 64, 3, n_sample=500)
            # print("self.data._X_train",self.data._X_train.shape)
            # print("W1_init.shape",W1_init.shape)
            W1_init = np.reshape(W1_init, (3, 3, 3, 64))
            # print("Reshaped W1_init.shape", W1_init.shape)
            return W1_init
            # return W1_init

        chanDim = -1  # since depth is appearing the end

        input_img = Input(shape=(32, 32, 3))  # adapt this if using `channels_first` image data format
       
        x = Conv2D(128, (3, 3),  use_bias=False, padding='same')(input_img)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        
        x = Conv2D(64, (3, 3), padding='same',use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        
        x = Conv2D(32, (3, 3), padding='same',use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        
        x = Conv2D(32, (3, 3), padding='same',use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)
        
        x = Conv2D(32, (3, 3), padding='same',use_bias=False)(encoded)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        x = UpSampling2D((2, 2))(x)
        
        x = Conv2D(32, (3, 3), padding='same',use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        x = UpSampling2D((2, 2))(x)
        
        x = Conv2D(64, (3, 3), padding='same',use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        x = UpSampling2D((2, 2))(x)
        
        x = Conv2D(128, (3, 3), padding='same',use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        x = UpSampling2D((2, 2))(x)
        
        x = Conv2D(3, (3, 3), padding='same',use_bias=False)(x)
        x = BatchNormalization()(x)
        decoded = Activation('sigmoid') (x)
        
        # this model maps an input to its encoded representation
        encoder = Model(input_img, encoded)

        autoencoder = Model(input_img, decoded)
        # Compile the autoencoder with the mean squared error
        autoencoder.compile(loss='binary_crossentropy', optimizer='adam')
        # print("[INFO] : Autoencoder Architecture", autoencoder.summary())

        return [autoencoder, encoder]
    
    def save_reconstructed_image(self, Xtest, X_decoded):

        # use Matplotlib (don't ask)
        import matplotlib.pyplot as plt
        # print("Xtest,",Xtest.shape,np.max(Xtest),np.min(Xtest))
        # print("X_decoded,",X_decoded.shape,np.max(X_decoded),np.min(X_decoded))
        # # Xtest = Xtest/255.0
        # X_decoded = X_decoded/255.0
       
        
        n = 10  # how many digits we will display
        plt.figure(figsize=(20, 4))
        for i in range(n):
            # display original
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(Xtest[i].reshape(self.IMG_HGT, self.IMG_WDT,self.channel))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(X_decoded[i].reshape(self.IMG_HGT, self.IMG_WDT,self.channel))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()
        plt.savefig(self.results + "/cae_reconstructed_images.png")
        

        return

    def pretrain_cae(self, solver, lr, n_epochs):
        
        # Compile the autoencoder with l2 loss and train for specified number of epochs
        if(OneClass_SVDD.DATASET == "mnist"):
            [cae, encoder] = self.compile_autoencoder()
            X = self.data._X_train
        elif(OneClass_SVDD.DATASET == "cifar10"):
            [cae, encoder] = self.compile_autoencoder_cifar()
            X = self.data._X_train
            self.data._X_test = self.data._X_test_beforegcn
            print("Inside cifar10: self.data._X_test,",self.data._X_test_beforegcn.shape,np.max(self.data._X_test_beforegcn),np.min(self.data._X_test_beforegcn))
        
            
        elif(OneClass_SVDD.DATASET == "gtsrb"):
            [cae, encoder] = self.compile_autoencoder_gtsrb()
            X = self.get_oneClass_trainData()
            self.data._X_test = X
            self.data._X_train = X
            # X = np.reshape(X,(len(X),32,32,3))
            # self.data._X_train = np.reshape(self.data._X_train,(len(self.data._X_train),32,32,3))
            # self.data._X_test = np.reshape(self.data._X_test,(len(self.data._X_test),32,32,3))
           
            
        
        # train_autoencoder(self)
        print("[INFO:] The shape of X used to train CAE",X.shape)
       
        # print("[INFO:] Training Convolutional Autoencoder with: ", solver, "optimizer")
        cae.fit(X, X,
                batch_size=200,
                epochs=n_epochs,
                verbose=0)

        # Save model and its weights
        # from keras.models import model_from_json
        # # serialize model to JSON
        # model_json = cae.to_json()
        # with open(self.results + "/cae.json", "w") as json_file:
        #     json_file.write(model_json)
        # # serialize weights to HDF5
        # cae.save_weights(self.results + "/cae.h5")
        # print("[INFO:] Saved model to disk @", self.results)
        
        # Reshape the Xtest data
        # self.data._X_test = np.reshape(self.data._X_test,(len(self.data._X_test),32,32,3))
        # print("[INFo:] data._X_test_beforegcn,",self.data._X_test_beforegcn.shape,np.max(self.data._X_test_beforegcn),np.min(self.data._X_test_beforegcn))
        if(OneClass_SVDD.DATASET == "cifar10"):
            X_test_sample = self.data._X_test_beforegcn[-5:]
            import random
            random_list = random.sample(range(1, 700), 5)
            
            
            X_train_sample = self.data._X_test_beforegcn[random_list]
            print("[INFO:] The shape of self.data._X_train", self.data._X_train.shape)
            print("[INFO:] The shape of self.data._X_test", self.data._X_test.shape)
             
            X_test = np.concatenate((X_train_sample, X_test_sample))
            
            X_decoded = cae.predict(X_test)
            #self.save_reconstructed_image(X_test, X_decoded)
            print("INFO: Autoencoder training completed....")
            
        elif(OneClass_SVDD.DATASET == "mnist"):
            X_test_sample = self.data._X_test[-5:]
            import random
            random_list = random.sample(range(1, 700), 5)
            
            
            X_train_sample = self.data._X_test[random_list]
            print("[INFO:] The shape of self.data._X_train", self.data._X_train.shape)
            print("[INFO:] The shape of self.data._X_test", self.data._X_test.shape)
             
            X_test = np.concatenate((X_train_sample, X_test_sample))
            
            X_decoded = cae.predict(X_test)
            # self.save_reconstructed_image(X_test, X_decoded)
            print("INFO: Autoencoder training completed for mnist....")
            
        # self.save_reconstructed_image(X_test, X_decoded)
        
        
        return [cae, encoder]

    def remove_layers(self, model, n_layers):

        for i in range(0, n_layers):
            model.layers.pop()

        return model

    def compile_svdd_network(self, cae):

        # Remove the decoder layers and preserve only the encoder
        model_svdd = self.remove_layers(cae, n_layers=11)
        # print("[INFO:] The compiled model is ", model_svdd.summary())
        return model_svdd

    def initialize_c_and_R(self, inputs):
        ## Initialize  c and R

        reps = self.get_OC_SVDD_network_reps(inputs)
        self.reps = reps

        print("[INFO:] The shape of the reps obtained are", reps.shape)

        reps = np.reshape(reps, (len(reps), (32)))
        self.reps = reps
        print("[INFO:] The shape of the reps obtained are", reps.shape)

        print("[INFO:] Initializing c and Radius R value...")
        eps = 0.1
        # consider the value all the number of batches (and thereby samples) to initialize from
        c = np.mean(reps, axis=0)

        # If c_i is too close to 0 in dimension i, set to +-eps.
        # Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        self.cvar = c  # Initialize the center

        # initialize R at the (1-nu)-th quantile of distances
        print("[INFO:] Center (c)  initialized.", c.shape)
        dist_init = np.sum((reps - c) ** 2, axis=1)
        out_idx = int(np.floor(len(reps) * Cfg.nu))
        sort_idx = dist_init.argsort()
        self.Rvar = Cfg.floatX(dist_init[sort_idx][-out_idx])

        print("[INFO:] Center (c)  initialized.", c)
        # print("[INFO:] Shape of Center (c)  initialized.", c.shape)
        # print("[INFO:] Distances (D)  initialized.", dist_init)
        # print("[INFO:] Shape of Distances (D)  initialized.", dist_init.shape)
        # print("[INFO:] out_idx (D)  out_idx.", out_idx)
        # print("[INFO:] sort_idx (D)  sort_idx.", sort_idx)
        print("[INFO:] Radius (R)  initialized.", Cfg.floatX(dist_init[sort_idx][-out_idx]))

        return

    def initialize_c_with_mean(self, inputs, encoder):

        reps = self.get_OC_SVDD_network_reps(inputs, encoder)

        # print("[INFO:] The shape of the reps obtained are", reps.shape)

        reps = np.reshape(reps, (len(reps), self.h_size))
        
        self.reps = reps
        # print("[INFO:] The shape of the reps obtained are", reps.shape)

        print("[INFO:] Initializing c and Radius R value...")
        eps = 0.1
        # consider the value all the number of batches (and thereby samples) to initialize from
        c = np.mean(reps, axis=0)

        # If c_i is too close to 0 in dimension i, set to +-eps.
        # Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        self.cvar = c  # Initialize the center
        self.center = c
        # initialize R at the (1-nu)-th quantile of distances
        # print("[INFO:] Center (c)  initialized.", c.shape)

        center = self.cvar
        dist = np.sum((reps - c) ** 2, axis=1)
        scores = dist
        val = np.sort(scores)
        # Cfg.nu = 0.01
        self.Rvar = np.percentile(val, Cfg.nu * 100)  # qth quantile of the radius.
        # print("[INFO:] Center (c)  initialized.", c)
        print("[INFO:] Radius (R)  initialized.", self.Rvar)

        return

    def fit(self):

        from keras.callbacks import LearningRateScheduler

        epochs = 150
        lr_power = 0.9

        def lr_scheduler(epoch, mode='adam'):
            # '''if lr_dict.has_key(epoch):
            #     lr = lr_dict[epoch]
            #     print 'lr: %f' % lr'''
            lr = 1e-4
            if mode is 'power_decay':
                # original lr scheduler
                lr = lr_base * ((1 - float(epoch) / epochs) ** lr_power)
            if mode is 'exp_decay':
                # exponential decay
                lr = (float(lr_base) ** float(lr_power)) ** float(epoch + 1)
            # adam default lr
            if mode is 'adam' and epoch > 50:
                lr = 1e-5
                if(epoch== 51):
                    print('lr: rate adjusted for fine tuning %f' % lr)

            if mode is 'progressive_drops':
                # drops as progression proceeds, good for sgd
                if epoch > 0.9 * epochs:
                    lr = 0.0001
                elif epoch > 0.75 * epochs:
                    lr = 0.001
                elif epoch > 0.5 * epochs:
                    lr = 0.01
                else:
                    lr = 0.1

            # print('lr: %f' % lr)
            return lr

        scheduler = LearningRateScheduler(lr_scheduler)

        if (self.lossfuncType == "SOFT_BOUND_DEEP_SVDD" or self.lossfuncType == "ONE_CLASS_DEEP_SVDD"):
            # initialize_c_as_mean
            n_batches = -1  # -1 refers to considering all the data in one batch
            inputs = self.get_oneClass_trainData()
           
            trainX = inputs
            print(" [INFO:]  The shape  of  trainX data ----",trainX.shape)
            [cae, encoder] = self.pretrain_cae(solver="adam", lr=1.0, n_epochs=150)
            # Create the SVDD network architecture and load pre-trained ae network weights
            # Initialize center c as the mean
            self.initialize_c_with_mean(inputs, encoder)
            inp = encoder.input
            out = encoder.layers[-1].output
            from keras.models import Model
            enoder_model_svdd = Model(inp, out)  # create a new model which doesn't have the last two layers in VGG16
            model_ocnn_svdd = Sequential()  # new model

            for layer in enoder_model_svdd.layers:
                model_ocnn_svdd.add(layer)

            model_ocnn_svdd.add(Flatten())
            if(OneClass_SVDD.DATASET == "mnist"):
                model_ocnn_svdd.add(Dense(self.h_size))
                X_train_to_Adjust_R = self.data._X_train
                trainX = X_train_to_Adjust_R
                
            elif(OneClass_SVDD.DATASET == "cifar10"):
                model_ocnn_svdd.add(Dense(self.h_size))
                X_train_to_Adjust_R = self.data._X_train
            
            elif(OneClass_SVDD.DATASET == "gtsrb"):
                model_ocnn_svdd.add(Dense(self.h_size))
                X_train_to_Adjust_R = trainX
                
            self.model_svdd = model_ocnn_svdd
            
            
            modeltype = "OC_SVDD"
            out_batch = Adjust_svdd_Radius(self.model_svdd, self.cvar, self.Rvar, X_train_to_Adjust_R, modeltype,self.h_size)
            callbacks = [out_batch, scheduler]

            # define SGD optimizer
            # opt = SGD(lr=0.01, decay=0.01 / 50, momentum=0.9, nesterov=True)
            opt = Adam(lr=1e-4)
            
            print("[INFO:] Hypersphere Loss function.....")
            self.model_svdd.compile(loss=self.custom_ocnn_hypershere_loss(),
                                    optimizer=opt)
            y_reps = out_batch.y_reps
            y_reps = y_reps[0:len(trainX)]

            # fit the  model_svdd by defining a custom loss function
            H = self.model_svdd.fit(trainX, y_reps, shuffle=True,
                                    batch_size=200,
                                    epochs=150,
                                    validation_split=0.01,
                                    verbose=0,
                                    callbacks=callbacks
                                    )

            self.Rvar = out_batch.radius
            self.cvar = out_batch.cvar
            print("[INFO:] \n Model compiled and fit Initial Radius Value...", self.Rvar)
            # print("[INFO:] Model compiled and fit with custom ocnn_hypershere_loss")


        elif (self.lossfuncType == "ONE_CLASS_NEURAL_NETWORK"):
            # Build OC-NN network
            n_batches = -1  # -1 refers to considering all the data in one batch
            inputs = self.get_oneClass_trainData()
            trainX = inputs
            [cae, encoder] = self.pretrain_cae(solver="adam", lr=1.0, n_epochs=150)
            # Create the SVDD network architecture and load pre-trained ae network weights
            # Initialize center c as the mean
            self.initialize_c_with_mean(inputs, encoder)
            inp = encoder.input
            out = encoder.layers[-1].output
            from keras.models import Model
            enoder_model_svdd = Model(inp, out)  # create a new model which doesn't have the last two layers in VGG16
            mode_ocnn_svdd = Sequential()  # new model
            
           
            if(OneClass_SVDD.DATASET == "gtsrb"):
                X_train_to_Adjust_R = trainX
                # n_epochs=500 
                
            elif(OneClass_SVDD.DATASET == "cifar10" ):
                X_train_to_Adjust_R = self.data._X_train
            
            elif( OneClass_SVDD.DATASET == "mnist"):
                X_train_to_Adjust_R = trainX
           
            
            
            for layer in enoder_model_svdd.layers:
                mode_ocnn_svdd.add(layer)

            mode_ocnn_svdd.add(Flatten())
            
            if(OneClass_SVDD.DATASET == "mnist"):
                
                mode_ocnn_svdd.add(Dense(32,activation='linear',use_bias=False))

                # mode_ocnn_svdd.add(Dense(16,activation='sigmoid',use_bias=False))
                
                # mode_ocnn_svdd.add(Dense(1,activation='linear',use_bias=False))
            
            if(OneClass_SVDD.DATASET == "cifar10" ):
                
                mode_ocnn_svdd.add(Dense(32,activation='linear',use_bias=False))

             
            if(OneClass_SVDD.DATASET == "gtsrb"):
                
                mode_ocnn_svdd.add(Dense(32,activation='linear',use_bias=False))

                mode_ocnn_svdd.add(Dense(16,activation='sigmoid',use_bias=False))
            

            # print("[INFO:] Model SVDD Summary", mode_ocnn_svdd.summary())
            self.model_svdd = mode_ocnn_svdd
            self.reps = self.model_svdd.predict(inputs[:len(inputs), :])
            # consider the value all the number of batches (and thereby samples) to initialize from
            c = np.mean(self.reps, axis=0)
            
            
            eps= 0.1
            c[(abs(c) < eps) & (c < 0)] = -eps
            c[(abs(c) < eps) & (c > 0)] = eps

            # If c_i is too close to 0 in dimension i, set to +-eps.
            self.cvar = c  # Initialize the center


            # print("[INFO:] Model SVDD Summary", self.model_svdd.summary())
            modeltype = "OC_NN"
            out_batch = Adjust_svdd_Radius(self.model_svdd, self.cvar, self.Rvar, X_train_to_Adjust_R, modeltype,self.h_size)
            callbacks = [out_batch, scheduler]

            # define SGD optimizer
            # opt = SGD(lr=0.01, decay=0.01 / 50, momentum=0.9, nesterov=True)
            opt = Adam(lr=1e-4)
            print("[INFO:] Hyperplane Loss function.....")
            self.model_svdd.compile(loss=self.custom_ocnn_hyperplane_loss(),
                                    optimizer=opt)
            y_reps = out_batch.y_reps
            y_reps = y_reps[0:len(trainX)]

            # fit the  model_svdd by defining a custom loss function
            H = self.model_svdd.fit(trainX, y_reps, shuffle=True,
                                    batch_size=200,
                                    epochs=100,
                                    validation_split=0.01,
                                    verbose=0,
                                    callbacks=callbacks
                                    )

            self.Rvar = out_batch.radius
            self.cvar = out_batch.cvar
            print("[INFO:] \n Model compiled  Outside CB: Updated Radius Value...", self.Rvar)
            # print("[INFO:] Model compiled and fit with custom ocnn_hypershere_loss")

        return

    def compute_predictedLabels_soft_margin(self, scores, poslabelBoundary, negBoundary, testX):

        # Normal datapoints have positive scores, whereas inlier have negative scores.
        result = np.sign(scores)
        y_pred = -1 * result  # Anomalies have postive scores and normal points have negative scores

        # print("result",result)
        np.savetxt(self.prj_path + "/y_pred.csv", y_pred,
                   delimiter=",")
        pos_decisionScore = y_pred[0:poslabelBoundary]
        neg_decisionScore = y_pred[poslabelBoundary:]

        self.plot_anomaly_score_histogram(pos_decisionScore, neg_decisionScore)

        # top_100_anomalies = np.asarray(top_100_anomalies)
        top_100_anomalies = testX[np.where(y_pred == -1)]
        print("The number of anomalies found are:", len(top_100_anomalies))

        result = self.tile_raster_images(top_100_anomalies, [28, 28], [30, 30])
        print("[INFO:] Saving Anomalies Found at ..", self.results)
        io.imsave(self.results + self.lossfuncType + "_Top100_anomalies.png", result)

        return y_pred

    def plot_anomaly_score_histogram(self, pos_decisionScore, neg_decisionScore):

        plt.hist(pos_decisionScore, label='Normal')
        plt.hist(neg_decisionScore, label='Anomaly')
        plt.legend(loc='upper left')
        plt.title(self.lossfuncType)
        # plt.show()
        plt.savefig(self.results + self.lossfuncType + ".png")

        return

    def scale_to_unit_interval(self, ndar, eps=1e-8):
        """ Scales all values in the ndarray ndar to be between 0 and 1 """
        ndar = ndar.copy()
        ndar -= ndar.min()
        ndar *= 1.0 / (ndar.max() + eps)
        return ndar

    def tile_raster_images(self, X, img_shape, tile_shape, tile_spacing=(0, 0),
                           scale_rows_to_unit_interval=True,
                           output_pixel_vals=True):
        """
        Source : http://deeplearning.net/tutorial/utilities.html#how-to-plot
        Transform an array with one flattened image per row, into an array in
        which images are reshaped and layed out like tiles on a floor.
        """

        assert len(img_shape) == 2
        assert len(tile_shape) == 2
        assert len(tile_spacing) == 2

        # The expression below can be re-written in a more C style as
        # follows :
        #
        # out_shape = [0,0]
        # out_shape[0] = (img_shape[0] + tile_spacing[0]) * tile_shape[0] -
        #                tile_spacing[0]
        # out_shape[1] = (img_shape[1] + tile_spacing[1]) * tile_shape[1] -
        #                tile_spacing[1]
        out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                     in zip(img_shape, tile_shape, tile_spacing)]

        if isinstance(X, tuple):
            assert len(X) == 4
            # Create an output numpy ndarray to store the image
            if output_pixel_vals:
                out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype='uint8')
            else:
                out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype=X.dtype)

            # colors default to 0, alpha defaults to 1 (opaque)
            if output_pixel_vals:
                channel_defaults = [0, 0, 0, 255]
            else:
                channel_defaults = [0., 0., 0., 1.]

            for i in range(4):
                if X[i] is None:
                    # if channel is None, fill it with zeros of the correct
                    # dtype
                    out_array[:, :, i] = np.zeros(out_shape,
                                                  dtype='uint8' if output_pixel_vals else out_array.dtype
                                                  ) + channel_defaults[i]
                else:
                    # use a recurrent call to compute the channel and store it
                    # in the output
                    out_array[:, :, i] = self.tile_raster_images(X[i], img_shape, tile_shape,
                                                                 tile_spacing, scale_rows_to_unit_interval,
                                                                 output_pixel_vals)
            return out_array

        else:
            # if we are dealing with only one channel
            H, W = img_shape
            Hs, Ws = tile_spacing

            # generate a matrix to store the output
            out_array = np.zeros(out_shape, dtype='uint8' if output_pixel_vals else X.dtype)

            for tile_row in range(tile_shape[0]):
                for tile_col in range(tile_shape[1]):
                    if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                        if scale_rows_to_unit_interval:
                            # if we should scale values to be between 0 and 1
                            # do this by calling the `scale_to_unit_interval`
                            # function
                            this_img = self.scale_to_unit_interval(
                                X[tile_row * tile_shape[1] + tile_col].reshape(img_shape))
                        else:
                            this_img = X[tile_row * tile_shape[1] + tile_col].reshape(img_shape)
                            # add the slice to the corresponding position in the
                            # output array
                        out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                        ] \
                            = this_img * (255 if output_pixel_vals else 1)
            return out_array

    def compute_predictedLabels_OneclassSVDD(self, scores, poslabelBoundary, negBoundary, testX):

        y_pred = np.ones(len(scores))
        scores_dict = {}
        # prepare the dict
        for index in range(0, len(scores)):
            scores_dict.update({index: scores[index]})

        worst_sorted_keys = sorted(scores_dict, key=scores_dict.get, reverse=True)
        anomaly_index = worst_sorted_keys[0:negBoundary]

        print("[INFO:] The anomaly index are ", anomaly_index)
        for key in anomaly_index:
            if (key >= poslabelBoundary):
                y_pred[key] = -1

        pos_decisionScore = y_pred[0:poslabelBoundary]
        neg_decisionScore = y_pred[0:negBoundary]

        self.plot_anomaly_score_histogram(pos_decisionScore, neg_decisionScore)

        most_anomalous_index = worst_sorted_keys[0:100]
        top_100_anomalies = []

        for i in most_anomalous_index:
            top_100_anomalies.append(testX[i])

        top_100_anomalies = np.asarray(top_100_anomalies)

        top_100_anomalies = np.reshape(top_100_anomalies, (-1, 28, 28))

        result = self.tile_raster_images(top_100_anomalies, [28, 28], [10, 10])
        print("[INFO:] Saving Anomalies Found at ..", self.results)
        io.imsave(self.results + self.lossfuncType + "_Top100_anomalies.png", result)

        return y_pred

    def tile_raster_visualise_anamolies_detected(self,testX, worst_top10_keys, lamda, nrows=10, ncols=10):
        #
        # print("[INFO:] The shape of input data  ",testX.shape)
        # print("[INFO:] The shape of decoded  data  ", decoded.shape)


        side = self.IMG_HGT
        channel = self.channel
        # side = 28
        # channel = 1
        # Display the decoded Original, noisy, reconstructed images


        img = np.ndarray(shape=(side * nrows, side * ncols, channel))
        print("img shape:", img.shape)

        worst_top10_keys = list(worst_top10_keys)

        # Display the decoded Original, noisy, reconstructed images for worst
        img = np.ndarray(shape=(side * nrows, side * ncols, channel))
        for i in range(ncols):
            row = i // ncols * nrows
            col = i % ncols
            img[side * row:side * (row + 1), side * col:side * (col + 1), :] = testX[worst_top10_keys[i]]
            img[side * (row + 1):side * (row + 2), side * col:side * (col + 1), :] = testX[worst_top10_keys[i+2*ncols]]
            img[side * (row + 2):side * (row + 3), side * col:side * (col + 1), :] = testX[worst_top10_keys[i+3*ncols]]
            img[side * (row + 3):side * (row + 4), side * col:side * (col + 1), :] = testX[worst_top10_keys[i+4*ncols]]
            img[side * (row + 4):side * (row + 5), side * col:side * (col + 1), :] = testX[worst_top10_keys[i+5*ncols]]
            img[side * (row + 5):side * (row + 6), side * col:side * (col + 1), :] = testX[worst_top10_keys[i+6*ncols]]
            img[side * (row + 6):side * (row + 7), side * col:side * (col + 1), :] = testX[worst_top10_keys[i+7*ncols]]
            img[side * (row + 7):side * (row + 8), side * col:side * (col + 1), :] = testX[worst_top10_keys[i+8*ncols]]
            img[side * (row + 8):side * (row + 9), side * col:side * (col + 1), :] = testX[worst_top10_keys[i+9*ncols]]
            img[side * (row + 9):side * (row + 10), side * col:side * (col + 1), :] = testX[worst_top10_keys[i+10*ncols]]

        img *= 255
        img = img.astype(np.uint8)

        # Save the image decoded
        if(OneClass_SVDD.DATASET == "cifar10"):
            print('Saving Top-100'+ str(lamda) + "most anomalous digit: @",self.results)
            io.imsave(self.results + '___' +str(lamda) + self.lossfunctype + 'Class_'+ str(Cfg.cifar10_normal) + "_Top100.png", img)
        
        if(OneClass_SVDD.DATASET == "mnist"):
            print('Saving Top-100'+ str(lamda) + "most anomalous digit: @",self.results)
            io.imsave(self.results + '___' +str(lamda) + self.lossfunctype + 'Class_'+ str(Cfg.mnist_normal) + "_Top100.png", img)

        
        return
    
    
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
        io.imsave(self.rcae_results + '/worst/' + str(lamda) + '_'+ '' +'.png', img)

        return
    
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


    def save_Most_Normal(self,X_test,scores):

        worst_sorted_keys= np.argsort(scores)
        most_anomalous_index = worst_sorted_keys[0:100]
        top_100_anomalies = []

        for i in most_anomalous_index:
            top_100_anomalies.append(X_test[i])

        top_100_anomalies = np.asarray(top_100_anomalies)
        
        if(OneClass_SVDD.DATASET == "cifar10"):
            print("[INFO:] The  top_100_anomalies",top_100_anomalies.shape)
            self.tile_raster_visualise_anamolies_detected(X_test,worst_sorted_keys,"most_normal")
            
        elif(OneClass_SVDD.DATASET == "mnist"):
            top_100_anomalies = np.reshape(top_100_anomalies, (-1, 28, 28))
            result = self.tile_raster_images(top_100_anomalies, [28, 28], [10, 10])
            print("[INFO:] Saving Anomalies Found at ..", self.results)
            io.imsave(self.results + self.lossfuncType + "__" + str(Cfg.mnist_normal) +"most_normal_Top100.png", result)
            

        return 
    
    def save_Most_Anomalous(self,X_test,scores):

        worst_sorted_keys= np.argsort(-scores)
        most_anomalous_index = worst_sorted_keys[0:100]
        top_100_anomalies = []

        for i in most_anomalous_index:
            top_100_anomalies.append(X_test[i])

        top_100_anomalies = np.asarray(top_100_anomalies)
        
        if(OneClass_SVDD.DATASET == "cifar10"):
            print("[INFO:] The  top_100_anomalies",top_100_anomalies.shape)
            self.tile_raster_visualise_anamolies_detected(X_test,worst_sorted_keys,"most_anomalous")
            
        elif(OneClass_SVDD.DATASET == "mnist"):
            top_100_anomalies = np.reshape(top_100_anomalies, (-1, 28, 28))
            result = self.tile_raster_images(top_100_anomalies, [28, 28], [10, 10])
            print("[INFO:] Saving Anomalies Found at ..", self.results)
            io.imsave(self.results + self.lossfuncType + "__" + str(Cfg.mnist_normal) +"most_anomalous_Top100.png", result)
           
        return 

    def predict(self):

        if (self.lossfuncType == "SOFT_BOUND_DEEP_SVDD"):
            testX, testY, PosBoundary, NegBoundary,X_testForPlotting = self.get_oneClass_testData()
            # testX = self._X_for_testingData
            # X_testForPlotting = testX
            center = self.cvar
            # Compute the predicted reps
            predicted_reps = self.model_svdd.predict(testX[:len(testX), :])

            # compute the score
            dist = np.sum(((predicted_reps - center) ** 2), axis=1)
            scores = dist - self.Rvar
            
            # Sort the scores and pick the inclass anomalies
            self.save_Most_Normal(X_testForPlotting,scores)
            
            self.save_Most_Anomalous(X_testForPlotting,scores)
            
            print("[INFO:] SOFT BOUNDARY SVDD Algorithm")
            auc = roc_auc_score(testY, scores)
            print("=" * 35)
            print("[INFO:]  AUROC Oneclass SVDD (Hypersphere)....", auc)
            print("=" * 35)

        elif (self.lossfuncType == "ONE_CLASS_DEEP_SVDD"):

            testX, testY, PosBoundary, NegBoundary,X_testForPlotting = self.get_oneClass_testData()
            center = self.cvar
            # Compute the predicted reps
            predicted_reps = self.model_svdd.predict(testX[:len(testX), :])

            # compute the score
            dist = np.sum(((predicted_reps - center) ** 2), axis=1)
            # scores = dist - self.Rvar SOft Bound Deep SVDD
            scores = dist  # One class Deep SVDD
            
            # Sort the scores and pick the inclass anomalies
            self.save_Most_Normal(X_testForPlotting,scores)
            
            self.save_Most_Anomalous(X_testForPlotting,scores)
            
            
            print("[INFO:] One Class Deep SVDD Algorithm")
            auc = roc_auc_score(testY, scores)
            print("=" * 35)
            print("[INFO:]  AUROC Oneclass SVDD (Hypersphere)....", auc)
            print("=" * 35)


        elif (self.lossfuncType == "ONE_CLASS_NEURAL_NETWORK"):

            testX, testY, PosBoundary, NegBoundary,X_testForPlotting = self.get_oneClass_testData()
            # Compute the predicted reps
            scores = self.model_svdd.predict(testX[:len(testX), :])
            center = self.cvar
            # compute the score
            dist = np.sum(((scores - center) ** 2), axis=1)
            # scores = dist - self.Rvar #SOft Bound Deep SVDD
            scores = dist  
            
            # Sort the scores and pick the inclass anomalies
            self.save_Most_Normal(X_testForPlotting,scores)
            
            self.save_Most_Anomalous(X_testForPlotting,scores)
            
            
            # compute the score
            print("[INFO:] Final Rvar used is ", self.Rvar)
            # print("[INFO:] Final Center value is ", self.cvar)

            # scores = predicted_reps - 0.0
            print("[INFO:] One Class Neural Network Algorithm")
            auc = roc_auc_score(testY, scores)
            print("=" * 35)
            print("[INFO:]  AUROC: Oneclass Neural Network (OCNN) ....", auc)
            print("=" * 35)

        return auc

