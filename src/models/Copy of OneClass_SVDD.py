# import the necessary packages
import numpy as np
from src.data.preprocessing import learn_dictionary
from sklearn.metrics import average_precision_score, mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from skimage import io

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
import tensorflow as tf

sess = tf.Session()

from keras.regularizers import L1L2

customL2_regularizer = L1L2(l2=1e-6)
from keras import backend as K

K.set_session(sess)

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


class Adjust_svdd_Radius(Callback):
    def __init__(self, model, cvar, radius, X_train):
        '''
        display: Number of batches to wait before outputting loss
        '''
        self.seen = 0
        self.radius = radius
        self.n_train = len(X_train)
        self.model = model
        self.inputs = X_train
        self.cvar = cvar

    def on_epoch_end(self, batch, logs={}):
        if Cfg.svdd_loss and (Cfg.hard_margin or (Cfg.block_coordinate)):
            # set R to be the (1-nu)-th quantile of distances

            # Initialize a fixed center
            # val = np.ones(Cfg.mnist_rep_dim) * 0.5
            # center = K.variable(value=val)
            center = self.cvar

            reps = self.model.predict(self.inputs[:len(self.inputs), :])
            out_idx = int(np.floor(self.n_train * Cfg.nu))
            # dist = K.sum(((reps - center) ** 2), axis=1)
            dist =(K.sum(K.square(reps - center), axis=1))
            scores = dist - self.radius
            val = K.eval(scores)
            val = np.sort(val)
            R_new = val[-out_idx] + self.radius
            # R_new = np.percentile(val,Cfg.nu*100) # qth quantile of the radius.

        self.radius = R_new

        # print("[INFO:] \n scores...",scores.shape)
        # print("[INFO:] \n self.reps...", reps.shape)
        # print("[INFO:] \n center Value...", center.shape)
        # print("[INFO:] \n val Value...", type(val))
        print("[INFO:] \n Updated Radius Value...", R_new)

        return self.radius


class OneClass_SVDD:
    ## Initialise static variables
    INPUT_DIM = 0
    HIDDEN_SIZE = 0
    DATASET = "mnist"
    mean_square_error_dict = {}
    RESULT_PATH = ""

    def __init__(self, dataset, lossfuncType, inputdim, hiddenLayerSize, img_hgt, img_wdt, img_channel, modelSavePath,
                 reportSavePath,
                 preTrainedWtPath, intValue=0, stringParam="defaultValue",
                 otherParam=None):
        """
        Called when initializing the classifier
        """
        OneClass_SVDD.DATASET = dataset
        OneClass_SVDD.INPUT_DIM = inputdim
        OneClass_SVDD.HIDDEN_SIZE = hiddenLayerSize
        OneClass_SVDD.RESULT_PATH = reportSavePath

        # print("OneClass_SVDD.RESULT_PATH:",OneClass_SVDD.RESULT_PATH)
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
        self.h_size = OneClass_SVDD.HIDDEN_SIZE
        global model
        self.r = 1.0
        self.kvar = 0.0
        self.pretrain = True
        self.load_dcae_path = ""
        self.num_decoder_layers = 14
        # self.num_decoder_layers = 7
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
            self.nn_model = MNIST_DataLoader()
            self.load_dcae_path = self.prj_path+"/models/MNIST/RCAE/"
            self.lossfuncType = lossfuncType
            self.n_train = len(self.data._X_train)

    def load_data(self, data_loader=None, pretrain=False):
        self.data = data_loader()
        return

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

    def load_pretrained_model(self):

        base_model = self.build_svdd_network()

        saved_encoderPath = self.get_pretrainedEncoderWts()

        model_name = self.get_pretrainedModelName()

        # load weights into new model
        base_model.load_weights(saved_encoderPath + model_name + '.h5')
        # print("[INFO:] BASE MODEL SUMMARY ------>")
        # base_model.summary(line_length=100)

        encoder = base_model

        return encoder

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
            model.pop()
        encoder = model

        return encoder

    def build_OneClass_SVDD_network(self):

        # load the pretrained model and its weights
        print(
            "[INFO:]  Before removing : This might take a while Please wait .... Loading pre-trained autoencoder weights")
        model = self.load_pretrained_model_full()
        self.encoder = model
        # self.encoder.summary(line_length=100)

        ## Remove the decoder portion and obtain a Flattened latent representation
        self.encoder = self.remove_decoder_layers(model)

        print("[INFO:] After removal Loaded pretrained DCAE Encoder model weights from disk.....")
        self.encoder.summary(line_length=100)

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

    # use the pretrained model saved to create the network with SVDD loss
    def get_OneClass_SVDD_network_reps(self, inputs):

        # Build the SVDD one class model
        self.model_svdd = self.build_OneClass_SVDD_network()

        self.reps = self.encoder.predict(inputs[:len(inputs), :])

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

        X_train = self.data._X_train
        y_train = self.data._y_train

        trainXPos = X_train[np.where(y_train == 1)]
        trainYPos = np.ones(len(trainXPos))
        trainXNeg = X_train[np.where(y_train == -1)]
        trainYNeg = -1 * np.ones(len(trainXNeg))
        PosBoundary = len(trainXPos)
        NegBoundary = len(trainXNeg)

        # X_train = np.concatenate((trainXPos, trainXNeg))
        # y_train = np.concatenate((trainYPos, trainYNeg))

        # Assign only one class of data
        X_train = trainXPos
        y_train = trainYPos

        print("[INFO: ] Shape of One Class Input Data used in training", X_train.shape)

        return X_train

    def get_oneClass_testData(self):

        X_train = self.data._X_train
        y_train = self.data._y_train

        trainXPos = X_train[np.where(y_train == 1)]
        trainYPos = np.ones(len(trainXPos))
        trainXNeg = X_train[np.where(y_train == -1)]

        trainYNeg = -1 * np.ones(len(trainXNeg))

        PosBoundary = len(trainXPos)
        NegBoundary = len(trainXNeg)

        X_test = np.concatenate((trainXPos, trainXNeg))
        y_test = np.concatenate((trainYPos, trainYNeg))

        print("[INFO: ] Shape of One Class Input Data used in testing", X_test.shape)
        print("[INFO: ] Shape of (Positive) One Class Input Data used in testing", trainXPos.shape)
        print("[INFO: ] Shape of (Negative) One Class Input Data used in testing", trainXNeg.shape)

        return [X_test, y_test, PosBoundary, NegBoundary]

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
            # dist = K.sum((y_pred - center) ** 2)
            dist = (K.sum(K.square(y_pred - center), axis=1))
            avg_distance_to_c = K.mean(dist)

            return (avg_distance_to_c)

        return custom_obj_ball

    def custom_ocnn_hyperplane_loss(self, nu, w, V):

        r = self.Rvar

        def custom_hinge(y_true, y_pred):
            term1 = 0.5 * tf.reduce_sum(w[0] ** 2)
            term2 = 0.5 * tf.reduce_sum(V[0] ** 2)
            term3 = 1 / nu * K.mean(K.maximum(0.0, r - tf.reduce_max(y_pred, axis=1)), axis=-1)
            term4 = -1 * r
            return (term1 + term2 + term3 + term4)

        return custom_hinge

    def build_svdd_network(self):

        # initialize the model
        model = Sequential()
        inputShape = (28, 28, 1)
        chanDim = -1  # since depth is appearing the end
        # first set of CONV => RELU => POOL layers
        model.add(Conv2D(20, (5, 5), padding="same", kernel_regularizer=customL2_regularizer, input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # second set of CONV => RELU => POOL layers
        model.add(Conv2D(50, (5, 5), kernel_regularizer=customL2_regularizer, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # first (and only) set of FC => RELU layers
        model.add(Flatten())

        model.add(Dense(2450, kernel_regularizer=customL2_regularizer))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        model.add(Dense(32, kernel_regularizer=customL2_regularizer))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        return model

    def build_ocnn_network(self):

        return model

    # Call fit function for each of the batches with svdd_loss
    def fit(self):

        if (self.lossfuncType == "SVDD_HYPERSPHERE_SOFT_BOUNDARY" or self.lossfuncType == "SVDD_HYPERSPHERE_ONE_CLASS_SVDD"):
            # initialize_c_as_mean
            n_batches = -1  # -1 refers to considering all the data in one batch
            inputs = self.get_oneClass_trainData()
            trainX = inputs
            self.initialize_c_as_mean(inputs, n_batches)

        elif (self.lossfuncType == "SVDD_HYPEPLANE"):

            # Build OC-NN network
            print("Build OCNN network.......")
            self.build_ocnn_network()

        # decay and fine tune the learning rate
        # decay_lr_callback = LearningRateScheduler(self.decay_learning_rate)
        # fine_tune_lr_callback = LearningRateScheduler(self.adjust_learning_rate_finetune)
        # #
        #
        # ## Call back function after every batch to adjust the value of radius
        # adjust_svdd_Radius = LearningRateScheduler(self.adjust_svdd_radius)

        # tbCallBack = keras.callbacks.TensorBoard(log_dir='../graph', histogram_freq=0, write_graph=True,
        #                                          write_images=True)

        # Call back obtain the l2_penalty_wts at end of each epoch
        # l2_penalty_wts = LambdaCallback(on_epoch_end=self.get_l2_penalty_wts(self.model_svdd))

        out_batch = Adjust_svdd_Radius(self.model_svdd, self.cvar, self.Rvar, self.data._X_train)
        callbacks = [out_batch]

        # define SGD optimizer

        # opt = SGD(lr=0.01, decay=0.01 / 50, momentum=0.9, nesterov=True)
        # opt = Adam(lr=1e-8, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        opt = Adagrad(lr=1e-5, epsilon=1e-08, decay=0.0)
        # fit the svdd model with hypersphere loss
        self.encoder.compile(loss=self.custom_ocnn_hypershere_loss(),
                                optimizer=opt)
        print("[INFO:] Fitting the encoder model =================>")
        # fit the  model_svdd by defining a custom loss function
        H = self.encoder.fit(trainX, trainX, shuffle=True,
                                batch_size=32,
                                epochs=10,
                                validation_split=0.01,
                                verbose=1,
                                callbacks=callbacks
                                )

        self.Rvar = out_batch.radius
        print("[INFO:] \n Outside CB: Updated Radius Value...", self.Rvar)

        print("[INFO:] Model compiled and fit with custom ocnn_hypershere_loss")

        return

    def compute_predictedLabels_soft_margin(self, scores,poslabelBoundary, negBoundary):

        # Normal datapoints have positive scores, whereas inlier have negative scores.
        result = np.sign(scores)
        y_pred = -1 * result  # Anomalies have postive scores and normal points have negative scores

        pos_decisionScore = y_pred[0:poslabelBoundary]
        neg_decisionScore = y_pred[0:negBoundary]

        self.plot_anomaly_score_histogram(pos_decisionScore, neg_decisionScore)

        return y_pred

    def plot_anomaly_score_histogram(self,pos_decisionScore,neg_decisionScore):
        import matplotlib.pyplot as plt

        plt.hist(pos_decisionScore, label='Normal')
        plt.hist(neg_decisionScore, label='Anomaly')
        plt.legend(loc='upper left')
        plt.title(self.lossfuncType)
        # plt.show()
        plt.savefig(self.results + self.lossfuncType+".png")

        return


    def compute_predictedLabels_OneclassSVDD(self, scores, poslabelBoundary, negBoundary):

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

        return y_pred

    def predict(self):

        testX, testY, PosBoundary, NegBoundary = self.get_oneClass_testData()
        # center = np.ones(Cfg.mnist_rep_dim) * 0.5
        center = self.cvar
        # Compute the predicted reps
        predicted_reps = self.model_svdd.predict(testX[:len(testX), :])
        # compute the score
        dist = K.sum(((predicted_reps - center) ** 2), axis=1)
        if(self.lossfuncType=="SVDD_HYPERSPHERE_SOFT_BOUNDARY"):
            scores = dist - self.Rvar
            scores = K.eval(scores)
            np.savetxt(self.prj_path +"/scores_SB.csv", scores,
                       delimiter=",")
            pred_labels = self.compute_predictedLabels_soft_margin(scores,PosBoundary,NegBoundary)

        elif(self.lossfuncType=="SVDD_HYPERSPHERE_ONE_CLASS_SVDD"):
            scores = dist
            scores = K.eval(scores)
            from sklearn import preprocessing
            min_max_scaler = preprocessing.MinMaxScaler()
            scores = np.reshape(scores,(-1,1))
            scores = min_max_scaler.fit_transform(scores)
            scores = scores.flatten()

            np.savetxt(self.prj_path +"/scores_ONE_CLASS_SVDD.csv", scores,
                   delimiter=",")
            pred_labels = self.compute_predictedLabels_OneclassSVDD(scores,PosBoundary,NegBoundary)



        # Compute AUROC
        auc = roc_auc_score(testY, pred_labels)
        print("=" * 35)
        print("[INFO:]  Scores....PosBoundary", scores, PosBoundary)
        print("[INFO:]  AUROC Oneclass SVDD (Hypersphere)....", auc)
        print("=" * 35)

        return




