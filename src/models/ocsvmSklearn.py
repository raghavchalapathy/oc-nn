# Import libraries for data wrangling, preprocessing and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import os
from keras import backend as K
from keras import callbacks
from keras import layers
from keras import models
from keras.wrappers.scikit_learn import KerasClassifier
import pandas as pd
import tensorflow as tf
from sklearn import metrics
from sklearn import pipeline
from sklearn import preprocessing
from sklearn.externals import joblib
# Importing libraries for building the neural network
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from sklearn.base import BaseEstimator, ClassifierMixin

class OCSVM(BaseEstimator, ClassifierMixin):




    """An example of classifier"""

    def __init__(self, img_hgt,img_wdt,intValue=0, stringParam="defaultValue", otherParam=None):
        """
        Called when initializing the classifier
        """
        self.intValue = intValue
        self.stringParam = stringParam

        # THIS IS WRONG! Parameters should have same name as attributes
        self.differentParam = otherParam 
               
        self.directory = "../models/supervisedBC/"
        self.IMG_HGT = img_hgt
        self.IMG_WDT=img_wdt

    @staticmethod
    def image_to_feature_vector(image, IMG_HGT, IMG_WDT):
        # resize the image to a fixed size, then flatten the image into
        # a list of raw pixel intensities
        return np.reshape(image, (len(image), IMG_HGT * IMG_WDT))

    def fit(self,X,nu,kernel):
  
        print("Training the OCSVM classifier.....")
        from sklearn import svm
        clf = svm.OneClassSVM(nu = nu, kernel = kernel)
        X = OCSVM.image_to_feature_vector(X, self.IMG_HGT, self.IMG_WDT)

        clf.fit(X) 

        return clf
        
    def compute_au_roc(self,y_true, df_score):
        y_scores_pos = df_score[0]
        y_scores_neg = df_score[1]
        y_score = np.concatenate((y_scores_pos, y_scores_neg))
        
        from sklearn.metrics import roc_auc_score
        roc_score = roc_auc_score(y_true, y_score)
 
        return roc_score
    
    def score(self,clf,Xtest_Pos,Xtest_Neg):

        Xtest_Pos= OCSVM.image_to_feature_vector(Xtest_Pos, self.IMG_HGT, self.IMG_WDT)
        Xtest_Neg = OCSVM.image_to_feature_vector(Xtest_Neg, self.IMG_HGT, self.IMG_WDT)

        decisionScore_POS = clf.decision_function(Xtest_Pos)
        decisionScore_Neg = clf.decision_function(Xtest_Neg)
        df_score = [ decisionScore_POS, decisionScore_Neg ]
        ## y_true
        y_true_pos = np.ones(Xtest_Pos.shape[0])
        y_true_neg = np.zeros(Xtest_Neg.shape[0])
        y_true = np.concatenate((y_true_pos, y_true_neg))
        
        plt.hist(decisionScore_POS, bins = 25, label = 'Normal');
        plt.hist(decisionScore_Neg, bins = 25, label = 'Anomaly');
        plt.legend(loc = 'upper right');
        plt.title('OC-SVM Normalised Decision Score');

        result = self.compute_au_roc(y_true,df_score)
        return result
        
        

        X_test = np.concatenate((X_testPos,X_testNeg),axis=0)
        X_testPosLabel = np.ones(len(X_testPos))
        X_testNegLabel = np.zeros(len(X_testNeg))
        y_test = np.concatenate((X_testPosLabel,X_testNegLabel),axis=0)
        pipe = joblib.load(os.path.join(self.directory, 'pipeline.pkl'))
        model = models.load_model(os.path.join(self.directory, 'model.h5'))
        pipe.steps.append(('nn', model))


        y_pred_keras = pipe.predict_proba(X_test)[:, 0]
        from sklearn.metrics import roc_curve
        fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)
        from sklearn.metrics import auc
        auc_keras = auc(fpr_keras, tpr_keras)
        print(auc_keras)
        return auc_keras
 
    def predict(self, X, y=None):
        # counts number of values bigger than mean
        print(" predict  function is not implemented for FakeNN")
        return
      
