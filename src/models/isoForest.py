import os
import time
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

from src.data.main import load_dataset
from src.utils.log import AD_Log
from src.utils.pickle import dump_isoForest, load_isoForest


class IsoForest(object):
    
    DATASET = "mnist"
    def __init__(self, dataset, n_estimators=100, max_samples='auto', contamination=0.1, **kwargs):

        # load dataset
        load_dataset(self, dataset)
        IsoForest.DATASET = dataset
        # initialize
        self.isoForest = None
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.initialize_isoForest(seed=self.data.seed, **kwargs)

        # train and test time
        self.clock = 0
        self.clocked = 0
        self.train_time = 0
        self.test_time = 0

        # Scores and AUC
        self.diag = {}

        self.diag['train'] = {}
        self.diag['val'] = {}
        self.diag['test'] = {}

        self.diag['train']['scores'] = np.zeros((len(self.data._y_train), 1))
        self.diag['val']['scores'] = np.zeros((len(self.data._y_val), 1))
        self.diag['test']['scores'] = np.zeros((len(self.data._y_test), 1))

        self.diag['train']['auc'] = np.zeros(1)
        self.diag['val']['auc'] = np.zeros(1)
        self.diag['test']['auc'] = np.zeros(1)

        self.diag['train']['acc'] = np.zeros(1)
        self.diag['val']['acc'] = np.zeros(1)
        self.diag['test']['acc'] = np.zeros(1)

        # AD results log
        self.ad_log = AD_Log()

        # diagnostics
        self.best_weight_dict = None  # attribute to reuse nnet plot-functions

    def initialize_isoForest(self, seed=0, **kwargs):

        self.isoForest = IsolationForest(n_estimators=self.n_estimators, max_samples=self.max_samples,
                                         contamination=self.contamination, n_jobs=-1, random_state=seed, **kwargs)

    def load_data(self, data_loader=None, pretrain=False):

        self.data = data_loader()

    def start_clock(self):

        self.clock = time.time()

    def stop_clock(self):

        self.clocked = time.time() - self.clock
        print("Total elapsed time: %g" % self.clocked)

    def get_oneClass_mnist_train_test_Data(self):
        # X_train = np.concatenate((self.data._X_train, self.data._X_val))
        # y_train = np.concatenate((self.data._y_train, self.data._y_val))

        X_train = self.data._X_train
        y_train = self.data._y_train
        
        X_test = self.data._X_test
        y_test = self.data._y_test

        
        ## Combine the positive data
        trainXPos = X_train[np.where(y_train == 0)]
        # trainYPos = np.zeros(len(trainXPos))
        trainYPos = np.ones(len(trainXPos))
        
        testXPos = X_test[np.where(y_test == 0)]
        # testYPos = np.zeros(len(testXPos))
        testYPos = np.ones(len(testXPos))
        
        
        # Combine the negative data
        trainXNeg = X_train[np.where(y_train == 1)]
        trainYNeg = -1*np.ones(len(trainXNeg))
        
        testXNeg = X_test[np.where(y_test == 1)]
        testYNeg = -1*np.ones(len(testXNeg))

     
        X_trainPOS = np.concatenate((trainXPos, testXPos))
        y_trainPOS = np.concatenate((trainYPos, testYPos))
        
        X_trainNEG = np.concatenate((trainXNeg, testXNeg))
        y_trainNEG = np.concatenate((trainYNeg, testYNeg))
        
        # Just 0.01 points are the number of anomalies.
        if(IsoForest.DATASET == "mnist"):
            num_of_anomalies = int(0.01 * len(X_trainPOS))
        elif(IsoForest.DATASET == "cifar10"):
            num_of_anomalies = int(0.1 * len(X_trainPOS))
        elif(IsoForest.DATASET == "gtsrb"):
            num_of_anomalies = int(0.1 * len(X_trainPOS))
        
        X_trainNEG = X_trainNEG[0:num_of_anomalies]
        y_trainNEG = y_trainNEG[0:num_of_anomalies]
        
        
        X_train = np.concatenate((X_trainPOS, X_trainNEG))
        y_train = np.concatenate((y_trainPOS, y_trainNEG))
        
    
        # print("[INFO: ] Shape of One Class Input Data used in training", X_train.shape)
        # print("[INFO: ] Shape of (Positive) One Class Input Data used in training", X_trainPOS.shape)
        # print("[INFO: ] Shape of (Negative) One Class Input Data used in training", X_trainNEG.shape)
        
        ## Making sure the same train and test data is used for both training and testing 
        self.data._X_train =  X_train
        self.data._y_train = y_train
        
        self.data._X_test =  X_train
        self.data._y_test =  y_train
        
        return [X_train,y_train]
    
    def get_oneClass_cifar10_train_test_Data(self):
        # X_train = np.concatenate((self.data._X_train, self.data._X_val))
        # y_train = np.concatenate((self.data._y_train, self.data._y_val))
        
        trainXPos = self.data._X_train[np.where(self.data._y_train == 0)]
        trainYPos = np.ones(len(trainXPos))
        trainXNeg = self.data._X_train[np.where(self.data._y_train == 1)]
        trainYNeg = -1*np.ones(len(trainXNeg))
        X_train = np.concatenate((trainXPos,trainXNeg))
        y_train = np.concatenate((trainYPos, trainYNeg)) ## Switch labels normal = 1 anomalies = -1

        self.data._X_train = X_train
        self.data._y_train = y_train
        
        ## Making sure the same train and test data is used for both training and testing 
        self.data._X_test =  X_train
        self.data._y_test =  y_train
         
        return 
    
    def get_oneClass_gtsrb_train_test_Data(self):
        # X_train = np.concatenate((self.data._X_train, self.data._X_val))
        # y_train = np.concatenate((self.data._y_train, self.data._y_val))
        
        trainXPos = self.data._X_train[np.where(self.data._y_train == 0)]
        trainYPos = np.ones(len(trainXPos))
        trainXNeg = self.data._X_train[np.where(self.data._y_train == 1)]
        trainYNeg = -1*np.ones(len(trainXNeg))
        X_train = np.concatenate((trainXPos,trainXNeg))
        y_train = np.concatenate((trainYPos, trainYNeg)) ## Switch labels normal = 1 anomalies = -1

        self.data._X_train = X_train
        self.data._y_train = y_train
        
        ## Making sure the same train and test data is used for both training and testing 
        self.data._X_test =  X_train
        self.data._y_test =  y_train
        
        print("[INFO: ] Shape of One Class Input Data used in training", X_train.shape)
        print("[INFO: ] Shape of (Positive) One Class Input Data used in training", trainXPos.shape)
        print("[INFO: ] Shape of (Negative) One Class Input Data used in training", trainXNeg.shape)
        
         
        return 
    
    def fit(self):
        
        # Obtaining the training and test data 
        ## In this experiment setting the training set is same as test set
        ## MNIST experiment 
        if(IsoForest.DATASET == "mnist"):
            self.get_oneClass_mnist_train_test_Data()
        elif(IsoForest.DATASET == "cifar10"):
            self.get_oneClass_cifar10_train_test_Data()
        elif(IsoForest.DATASET == "gtsrb"):    
            self.get_oneClass_gtsrb_train_test_Data()
            
        
        self.diag = {}
        self.diag['train'] = {}
        self.diag['val'] = {}
        self.diag['test'] = {}
        self.diag['train']['scores'] = np.zeros((len(self.data._y_train), 1))
        self.diag['val']['scores'] = np.zeros((len(self.data._y_train), 1))
        self.diag['test']['scores'] = np.zeros((len(self.data._y_train), 1))
        self.diag['train']['auc'] = np.zeros(1)
        self.diag['val']['auc'] = np.zeros(1)
        self.diag['test']['auc'] = np.zeros(1)
        self.diag['train']['acc'] = np.zeros(1)
        self.diag['val']['acc'] = np.zeros(1)
        self.diag['test']['acc'] = np.zeros(1)
        
        trainXPos = self.data._X_train[np.where(self.data._y_train == 1)]
        trainXNeg = self.data._X_train[np.where(self.data._y_train == -1)]
        
        print("The shape of Training data : ",self.data._X_train.shape)
        print("The shape of POS data : ",trainXPos.shape)
        print("The shape of NEG data : ",trainXNeg.shape)
        
        if self.data._X_train.ndim > 2:
            X_train_shape = self.data._X_train.shape
            X_train = self.data._X_train.reshape(X_train_shape[0], -1)
        else:
            X_train = self.data._X_train

        print("Starting training...")
        self.start_clock()

        self.isoForest.fit(X_train.astype(np.float32))

        self.stop_clock()
        self.train_time = self.clocked

    def predict(self, which_set='train'):

        assert which_set in ('train', 'test')
        
        
       
        
        if which_set == 'train':
            X = self.data._X_train
            y = self.data._y_train
        if which_set == 'test':
            X = self.data._X_test
            y = self.data._y_test
            
            testXPos = self.data._X_test[np.where(self.data._y_test == 0)]
            testXNeg = self.data._X_test[np.where(self.data._y_test == 1)]
        
            # print("The shape of Testing data : ",self.data._X_test.shape,self.data._y_test.shape)
            # print("The shape of POS data : ",testXPos.shape)
            # print("The shape of NEG data : ",testXNeg.shape)

        # reshape to 2D if input is tensor
        if X.ndim > 2:
            X_shape = X.shape
            X = X.reshape(X_shape[0], -1)

        print("Starting prediction...")
        self.start_clock()

        scores = (1.0) * self.isoForest.decision_function(X.astype(np.float32))  # compute anomaly score
        y_pred = (self.isoForest.predict(X.astype(np.float32)) == 1) * 1  # get prediction

        self.diag[which_set]['scores'][:, 0] = scores.flatten()
        self.diag[which_set]['acc'][0] = 100.0 * sum(y == y_pred) / len(y)

        if sum(y) > 0:
            auc = roc_auc_score(y, scores.flatten())
            self.diag[which_set]['auc'][0] = auc

        self.stop_clock()
        if which_set == 'test':
            self.test_time = self.clocked
            
        return auc

    def dump_model(self, filename=None):

        dump_isoForest(self, filename)

    def load_model(self, filename=None):

        assert filename and os.path.exists(filename)

        load_isoForest(self, filename)

    def log_results(self, filename=None):
        """
        log the results relevant for anomaly detection
        """

        self.ad_log['train_auc'] = self.diag['train']['auc'][-1]
        self.ad_log['train_accuracy'] = self.diag['train']['acc'][-1]
        self.ad_log['train_time'] = self.train_time

        self.ad_log['test_auc'] = self.diag['test']['auc'][-1]
        self.ad_log['test_accuracy'] = self.diag['test']['acc'][-1]
        self.ad_log['test_time'] = self.test_time

        self.ad_log.save_to_file(filename=filename)
