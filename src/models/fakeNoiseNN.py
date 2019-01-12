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
from keras.layers import Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
import tensorflow as tf


class SupervisedFakeNoiseNN(BaseEstimator, ClassifierMixin):
    """An example of classifier"""

    def __init__(self, intValue=0, stringParam="defaultValue",
                 otherParam=None):
        """
        Called when initializing the classifier
        """
        self.intValue = intValue
        self.stringParam = stringParam

        # THIS IS WRONG! Parameters should have same name as attributes
        self.differentParam = otherParam

        self.directory = "../models/supervisedFakeNN/"
        self.h_size = 64

    def train_KerasBinaryClassifier(self, X_train, y_train,noOfepochs):

        # Use Tenserflow backend
        sess = tf.Session()
        K.set_session(sess)

        def custom_activation(x):
            return (1 / np.sqrt(self.h_size)) * tf.cos(x / 0.02)

        get_custom_objects().update({
            'custom_activation':
            Activation(custom_activation)
        })

        def model():
            # Load the 
            # print("Loading the Pretrained Supervised NN Model..... ")
            from keras.models import load_model
            from keras.models import model_from_json
            

            # # Model reconstruction from JSON file
            # with open('../models/supervisedBC/model_architecture.json', 'r') as f:
            #     best_model = model_from_json(f.read())

            # # Load weights into the new model
            # best_model.load_weights('../models/supervisedBC/model_weights.h5')
            # best_model.compile(
            #     optimizer='rmsprop',
            #     loss='binary_crossentropy',
            #     metrics=['accuracy'])
           
            
            model = Sequential()
            model.add(Dense(128, input_dim=X_train.shape[1]))
            model.add(Activation(custom_activation))
            model.add(Dense(64, activation='linear'))
            model.add(Dense(1))
            model.compile(
                optimizer='rmsprop',
                loss='binary_crossentropy',
                metrics=['accuracy'])

            # ## Copy the weights from one model to another model
            # model.set_weights(best_model.get_weights()) 
            

            return model

        # early_stopping = callbacks.EarlyStopping(
        #     monitor='val_loss', patience=1, verbose=0, mode='auto')
        # print("Removed Early stopping......")
        pipe = pipeline.Pipeline([('rescale', preprocessing.StandardScaler()),
                                  ('nn',
                                   KerasClassifier(
                                       build_fn=model,
                                       epochs=noOfepochs,
                                       batch_size=128,
                                       verbose=0,
                                       validation_split=0.2))])

                                    #    callbacks=[early_stopping]
        pipe.fit(X_train, y_train)

        model_step = pipe.steps.pop(-1)[1]
        joblib.dump(pipe, os.path.join(self.directory, 'pipeline.pkl'))
        # print("Trained Model is Saved at relative path inside PROJECT_DIR ",
            #   self.directory)
        models.save_model(model_step.model,
                          os.path.join(self.directory, 'model.h5'))
        return

    def fit(self, X_train,Y_train,epochs):

        # print("Training the Keras Binary classifier.....")
        self.train_KerasBinaryClassifier(X_train, Y_train,epochs)

    def predict(self, X_testPos, X_testNeg):

        X_test = np.concatenate((X_testPos, X_testNeg), axis=0)
        X_testPosLabel = np.ones(len(X_testPos))
        X_testNegLabel = np.zeros(len(X_testNeg))
        y_test = np.concatenate((X_testPosLabel, X_testNegLabel), axis=0)
        pipe = joblib.load(os.path.join(self.directory, 'pipeline.pkl'))
        model = models.load_model(os.path.join(self.directory, 'model.h5'))
        pipe.steps.append(('nn', model))

        y_pred_keras = pipe.predict_proba(X_test)[:, 0]
        from sklearn.metrics import roc_curve
        fpr_keras, tpr_keras, thresholds_keras = roc_curve(
            y_test, y_pred_keras)
        from sklearn.metrics import auc
        auc_keras = auc(fpr_keras, tpr_keras)
        print("AUC:",auc_keras)
        return auc_keras

    def score(self, X, y=None):
        # counts number of values bigger than mean
        print(" Score function is not implemented for FakeNN")
        return
