import numpy as np
import pandas as pd
from sklearn import utils
import matplotlib
from sklearn import svm

dataPath = './data/'
from tf_OneClass_CNN_model import func_get_ImageVectors

colNames = ["sklearn-OCSVM-Linear-Train","sklearn-OCSVM-RBF-Train","sklearn-OCSVM-Linear-Test","sklearn-OCSVM-RBF-Test","sklearn-explicit-Linear-Train","sklearn-explicit-Sigmoid-Train","sklearn-explicit-Linear-Test","sklearn-explicit-Sigmoid-Test","tf-Linear-Train","tf-Sigmoid-Train","tf-Linear-Test","tf-Sigmoid-Test","tfLearn-Linear-Train","tfLearn-Sigmoid-Train","tfLearn-Linear-Test","tfLearn-Sigmoid-Test"]

# Create empty dataframe with given column names.
df_usps_scores = {}
df_fake_news_scores = {}
df_spam_vs_ham_scores = {}
df_cifar_10_scores = {}
# dataPathTrain = '/Users/raghav/Documents/Uni/oc-nn/data/cifar-10_data/train/dogs/'
# dataPathTest = '/Users/raghav/Documents/Uni/oc-nn/data/cifar-10_data/test/'
from itertools import izip_longest
import csv
decision_scorePath = "/Users/raghav/Documents/Uni/oc-nn/Decision_Scores/synthetic/"
def write_decisionScores2Csv(path, filename, positiveScores, negativeScores):

    newfilePath = path+filename
    print "Writing file to ", path+filename
    poslist = positiveScores.tolist()
    neglist = negativeScores.tolist()

    # rows = zip(poslist, neglist)
    d = [poslist, neglist]
    export_data = izip_longest(*d, fillvalue='')
    with open(newfilePath, 'w') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(("Normal", "Anomaly"))
        wr.writerows(export_data)
    myfile.close()

    return

def func_getKerasModelfeatures():
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    import numpy as np
    import cPickle
    import h5py
    import os
    import json
    import seaborn as sns
    import matplotlib.pyplot as plt

    # load the user configs
    with open('/Users/raghav/Documents/Uni/oc-nn/flower-recognition/conf/conf.json') as f:
        config = json.load(f)

    # config variables
    test_size = config["test_size"]
    seed = config["seed"]
    features_path = config["features_path"]
    labels_path = config["labels_path"]
    results = config["results"]
    classifier_path = config["classifier_path"]
    train_path = config["train_path"]
    num_classes = config["num_classes"]

    # import features and labels
    h5f_data = h5py.File(features_path, 'r')
    h5f_label = h5py.File(labels_path, 'r')

    features_string = h5f_data['dataset_1']
    labels_string = h5f_label['dataset_1']

    features = np.array(features_string)
    labels = np.array(labels_string)

    print type(features)
    h5f_data.close()
    h5f_label.close()

    # verify the shape of features and labels
    print "[INFO] features shape: {}".format(features.shape)
    print "[INFO] labels shape: {}".format(labels.shape)
    print "[INFO] type shape: ", type(features)

    data_train = features[0:220]
    data_test = features[220:231]

    return [data_train,data_test]

import time
def sklearn_OCSVM_linear(data_train,data_test,nu):


    ocSVM = svm.OneClassSVM(nu = nu, kernel = 'linear')
    start_time = time.time()
    ocSVM.fit(data_train)
    trainTime = time.time() - start_time

    start_time = time.time()
    pos_decisionScore = ocSVM.decision_function(data_train)
    neg_decisionScore = ocSVM.decision_function(data_test)
    testTime = time.time() - start_time

    print pos_decisionScore
    print neg_decisionScore

    write_decisionScores2Csv(decision_scorePath,"ocsvm_linear.csv",pos_decisionScore,neg_decisionScore)
    return [pos_decisionScore, neg_decisionScore,trainTime,testTime]


def sklearn_OCSVM_rbf(data_train,data_test,nu):


    ocSVM = svm.OneClassSVM(nu = nu, kernel = 'rbf')
    start_time = time.time()
    ocSVM.fit(data_train)
    trainTime = time.time() - start_time
    start_time = time.time()
    pos_decisionScore = ocSVM.decision_function(data_train)
    neg_decisionScore = ocSVM.decision_function(data_test)
    testTime = time.time() - start_time

    print "pos_decisionScore",(pos_decisionScore)
    print "neg_decisionScore",(neg_decisionScore)
    write_decisionScores2Csv(decision_scorePath, "ocsvm_rbf.csv", pos_decisionScore, neg_decisionScore)
    return [pos_decisionScore, neg_decisionScore,trainTime,testTime]


def func_getDecision_Scores_sklearn_OCSVM(dataset,data_train,data_test):


    # print "Decision_Scores_sklearn_OCSVM Using Linear and RBF Kernels....."

    if(dataset=="USPS" ):
        
        result = sklearn_OCSVM_linear(data_train,data_test)
        df_usps_scores["sklearn-OCSVM-Linear-Train"] = result[0]
        df_usps_scores["sklearn-OCSVM-Linear-Test"] =  result[1]
 
        result = sklearn_OCSVM_rbf(data_train,data_test)
        df_usps_scores["sklearn-OCSVM-RBF-Train"] = result[0]
        df_usps_scores["sklearn-OCSVM-RBF-Test"] = result[1]


    if(dataset=="FAKE_NEWS" ):   
        result = sklearn_OCSVM_linear(data_train,data_test)
        df_fake_news_scores["sklearn-OCSVM-Linear-Train"] = result[0]
        df_fake_news_scores["sklearn-OCSVM-Linear-Test"] = result[1]
        
        result = sklearn_OCSVM_rbf(data_train,data_test)
        df_fake_news_scores["sklearn-OCSVM-RBF-Train"] = result[0]
        df_fake_news_scores["sklearn-OCSVM-RBF-Test"] = result[1]


    if(dataset=="SPAM_Vs_HAM" ):
        result = sklearn_OCSVM_linear(data_train,data_test)
        df_spam_vs_ham_scores["sklearn-OCSVM-Linear-Train"] = result[0] 
        df_spam_vs_ham_scores["sklearn-OCSVM-Linear-Test"] = result[1]
        
        result = sklearn_OCSVM_rbf(data_train,data_test)
        df_spam_vs_ham_scores["sklearn-OCSVM-RBF-Train"] = result[0]
        df_spam_vs_ham_scores["sklearn-OCSVM-RBF-Test"] = result[1]


    if(dataset=="CIFAR-10" ):
        result = sklearn_OCSVM_linear(data_train,data_test)
        df_cifar_10_scores["sklearn-OCSVM-Linear-Train"] = result[0]
        df_cifar_10_scores["sklearn-OCSVM-Linear-Test"] = result[1]
        
        result = sklearn_OCSVM_rbf(data_train,data_test)
        df_cifar_10_scores["sklearn-OCSVM-RBF-Train"] = result[0]
        df_cifar_10_scores["sklearn-OCSVM-RBF-Test"] = result[1]

    return [df_usps_scores,df_fake_news_scores,df_spam_vs_ham_scores,df_cifar_10_scores]


