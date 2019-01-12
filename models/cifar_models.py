import os
cwd = os.getcwd()
import sys
sys.path.append(cwd)
print cwd

# Import all the models functions
from sklearn_OCSVM_model import sklearn_OCSVM_linear, sklearn_OCSVM_rbf
from OneClass_NN_model import One_Class_NN_explicit_linear, One_Class_NN_explicit_sigmoid
from  sklearn_OCSVM_rpca import sklearn__RPCA_OCSVM
from sklearn_isolation_forest import sklearn_IsolationForest
from CAE_OCSVM_models import CAE_OCSVM_Linear,CAE_OCSVM_RBF
from AE_SVDD_models import AE2_SVDD_Linear, AE2_SVDD_RBF

from tf_Cifar_OC_NN_Models import tf_OneClass_NN_linear,tf_OneClass_NN_sigmoid,tf_OneClass_NN_relu

# Declare a dictionary to store the results
df_cifar_scores = {}
df_cifar_time = {}
nu = 0.1
import time


def func_getDecision_Scores_cifar(dataset,data_train,data_test,raw_train,raw_test,data_train_ae2,data_test_ae2,modelpath):

    print "Calling.....",func_getDecision_Scores_cifar
    result = CAE_OCSVM_Linear(data_train,data_test,nu)
    df_cifar_scores["cae_ocsvm-linear-Train"] = result[0]
    df_cifar_scores["cae_ocsvm-linear-Test"] =  result[1]

    df_cifar_time["cae_ocsvm-linear-Train"] = result[2]
    df_cifar_time["cae_ocsvm-linear-Test"] = result[3]


    print ("Finished cae_ocsvm-linear")


    result = CAE_OCSVM_RBF(data_train,data_test,nu)
    df_cifar_scores["cae_ocsvm-rbf-Train"] = result[0]
    df_cifar_scores["cae_ocsvm-rbf-Test"] = result[1]

    df_cifar_time["cae_ocsvm-rbf-Train"] = result[2]
    df_cifar_time["cae_ocsvm-rbf-Test"] = result[3]
    print ("Finished cae_ocsvm-rbf")


    result = tf_OneClass_NN_linear(data_train,data_test,nu)
    df_cifar_scores["tf_OneClass_NN-Linear-Train"] = result[0]
    df_cifar_scores["tf_OneClass_NN-Linear-Test"] =  result[1]

    df_cifar_time["tf_OneClass_NN-Linear-Train"] = result[2]
    df_cifar_time["tf_OneClass_NN-Linear-Test"] =  result[3]

    print ("Finished tf_OneClass_NN_linear")


    result = tf_OneClass_NN_sigmoid(data_train,data_test,nu)
    df_cifar_scores["tf_OneClass_NN-Sigmoid-Train"] = result[0]
    df_cifar_scores["tf_OneClass_NN-Sigmoid-Test"] = result[1]

    df_cifar_time["tf_OneClass_NN-Sigmoid-Train"] = result[2]
    df_cifar_time["tf_OneClass_NN-Sigmoid-Test"] = result[3]
    print ("Finished tf_OneClass_NN_sigmoid")

    result = tf_OneClass_NN_relu(data_train,data_test,nu)
    df_cifar_scores["tf_OneClass_NN-Relu-Train"] = result[0]
    df_cifar_scores["tf_OneClass_NN-Relu-Test"] = result[1]
    print ("Finished tf_OneClass_NN_relu")


    result = sklearn_OCSVM_linear(raw_train,raw_test,nu)

    df_cifar_scores["sklearn-OCSVM-Linear-Train"] = result[0]
    df_cifar_scores["sklearn-OCSVM-Linear-Test"] =  result[1]
    print result[2],result[3]
    df_cifar_time["sklearn-OCSVM-Linear-Train"] = result[2]
    df_cifar_time["sklearn-OCSVM-Linear-Test"] =  result[3]
    print ("Finished sklearn_OCSVM_linear")


    result = sklearn_OCSVM_rbf(raw_train,raw_test,nu)
    df_cifar_scores["sklearn-OCSVM-RBF-Train"] = result[0]
    df_cifar_scores["sklearn-OCSVM-RBF-Test"] = result[1]

    df_cifar_time["sklearn-OCSVM-RBF-Train"] = result[2]
    df_cifar_time["sklearn-OCSVM-RBF-Test"] = result[3]
    print ("Finished sklearn_OCSVM_RBF")



    result = sklearn__RPCA_OCSVM(raw_train,raw_test,nu)
    df_cifar_scores["rpca_ocsvm-Train"] = result[0]
    df_cifar_scores["rpca_ocsvm-Test"] =  result[1]
    ## Training and Testing Times recorded
    df_cifar_time["rpca_ocsvm-Train"] = result[2]
    df_cifar_time["rpca_ocsvm-Test"] =  result[3]

    print ("Finished rpca_ocsvm")

    result = sklearn_IsolationForest(raw_train,raw_test)
    df_cifar_scores["isolation-forest-Train"] = result[0]
    df_cifar_scores["isolation-forest-Test"] = result[1]
    ## Training and Testing Times recorded
    df_cifar_time["isolation-forest-Train"] = result[2]
    df_cifar_time["isolation-forest-Test"] = result[3]
    print ("Finished isolation-forest")


    result = AE2_SVDD_Linear(data_train_ae2,data_test_ae2,nu,modelpath)
    df_cifar_scores["ae_svdd-linear-Train"] = result[0]
    df_cifar_scores["ae_svdd-linear-Test"] =  result[1]
    ## Training and Testing Times recorded
    df_cifar_time["ae_svdd-linear-Train"] = result[2]
    df_cifar_time["ae_svdd-linear-Test"] =  result[3]
    print ("Finished ae_ocsvm-linear")

    result = AE2_SVDD_RBF(data_train_ae2,data_test_ae2,nu,modelpath)
    df_cifar_scores["ae_svdd-rbf-Train"] = result[0]
    df_cifar_scores["ae_svdd-rbf-Test"] = result[1]

    ## Training and Testing Times recorded
    df_cifar_time["ae_svdd-rbf-Train"] = result[2]
    df_cifar_time["ae_svdd-rbf-Test"] = result[3]
    print ("Finished ae_ocsvm-rbf")

    # Write a CSV file for Cifar-10 data consisting of Methods, Train and test time
    #Method, Train, Test
    methods = ['OC-NN-Linear', 'OC-NN-Sigmoid',  'CAE-OCSVM-Linear', 'CAE-OCSVM-RBF', 'AE2-SVDD-Linear',
               'AE2-SVDD-RBF', 'OCSVM-Linear', 'OCSVM-RBF', 'RPCA_OCSVM', 'Isolation_Forest']
    write_training_test_results(df_cifar_time,methods)

    return df_cifar_scores


import csv
def write_training_test_results(df_time,methods):

    download_dir = "/Users/raghav/Documents/Uni/oc-nn/trainTest_Time/cifar_trainTest.csv"  # where you want the file to be downloaded to
    print "Writing file to ", download_dir
    csv = open(download_dir, "a")
    for method in methods:
        if(method == "OC-NN-Linear"):
            row = method + "," + str(df_time["tf_OneClass_NN-Linear-Train"] ) + "," + str(df_time["tf_OneClass_NN-Linear-Test"]) + "\n"
            csv.write(row)
        if(method=="OC-NN-Sigmoid"):
            row = method + "," + str(df_time["tf_OneClass_NN-Sigmoid-Train"]) + "," + str(df_time["tf_OneClass_NN-Sigmoid-Test"]) + "\n"
            csv.write(row)

        if (method == "CAE-OCSVM-Linear"):
            row = method + "," + str(df_time["cae_ocsvm-linear-Train"]) + "," + str(df_time["cae_ocsvm-linear-Test"]) + "\n"
            csv.write(row)

        if (method == "CAE-OCSVM-RBF"):
            row = method + "," + str(df_time["cae_ocsvm-rbf-Train"]) + "," + str(df_time["cae_ocsvm-rbf-Test"]) + "\n"
            csv.write(row)

        if (method == "AE2-SVDD-Linear"):
            row = method + "," + str(df_time["ae_svdd-linear-Train"]) + "," + str(df_time["ae_svdd-linear-Test"]) + "\n"
            csv.write(row)

        if (method == "AE2-SVDD-RBF"):
            row = method + "," + str(df_time["ae_svdd-rbf-Train"]) + "," + str(df_time["ae_svdd-rbf-Test"]) + "\n"
            csv.write(row)

        if (method == "OCSVM-Linear"):
            row = method + "," + str(df_time["sklearn-OCSVM-Linear-Train"]) + "," + str(df_time["sklearn-OCSVM-Linear-Test"]) + "\n"
            csv.write(row)

        if (method == "OCSVM-RBF"):
            row = method + "," + str(df_time["sklearn-OCSVM-RBF-Train"]) + "," + str(df_time["sklearn-OCSVM-RBF-Test"]) + "\n"
            csv.write(row)

        if (method == "RPCA_OCSVM"):
            row = method + "," + str(df_time["rpca_ocsvm-Train"]) + "," + str(df_time["rpca_ocsvm-Test"]) + "\n"
            csv.write(row)

        if (method == "Isolation_Forest"):
            row = method + "," + str(df_time["isolation-forest-Train"]) + "," + str(df_time["isolation-forest-Test"])+ "\n"
            csv.write(row)


    return
#
# def func_getDecision_Scores_cifar_10(dataset, data_train, data_test,  data_train_ocsvm, data_test_ocsvm):
#
#     # data_train = func_getKerasModelfeatures(dataPathTrain)
#     # data_test = func_get_ImageVectors(dataPathTest)
#
#     [data_train,data_test ]= func_getKerasModelfeatures()
#
#
#     data_train_ocsvm = data_train
#     data_test_ocsvm = data_test
#
#     print "OCSVM-input-train",data_train_ocsvm.shape
#     print "OCSVM-input-test", data_test_ocsvm.shape
#
#     result = sklearn_OCSVM_linear(data_train_ocsvm, data_test_ocsvm)
#     df_cifar_scores["sklearn-OCSVM-Linear-Train"] = result[0]
#     df_cifar_scores["sklearn-OCSVM-Linear-Test"] = result[1]
#     print ("Finished sklearn_OCSVM_linear")
#     result = sklearn_OCSVM_rbf(data_train_ocsvm, data_test_ocsvm)
#     df_cifar_scores["sklearn-OCSVM-RBF-Train"] = result[0]
#     df_cifar_scores["sklearn-OCSVM-RBF-Test"] = result[1]
#     print ("Finished sklearn_OCSVM_RBF")
#     #
#     # # sklearn _OCSVM_explicit
#     # result = sklearn_OCSVM_explicit_linear(data_train_ocsvm, data_test_ocsvm)
#     # df_cifar_scores["sklearn-OCSVM-explicit-Linear-Train"] = result[0]
#     # df_cifar_scores["sklearn-OCSVM-explicit-Linear-Test"] = result[1]
#     # print ("Finished sklearn _OCSVM_explicit_Linear")
#     #
#     # result = sklearn_OCSVM_explicit_sigmoid(data_train_ocsvm, data_test_ocsvm)
#     # df_cifar_scores["sklearn-OCSVM-explicit-Sigmoid-Train"] = result[0]
#     # df_cifar_scores["sklearn-OCSVM-explicit-Sigmoid-Test"] = result[1]
#     # #
#     # print ("Finished sklearn _OCSVM_explicit_Sigmoid")
#
#     #One Class NN Explicit
#     result = One_Class_NN_explicit_linear(data_train_ocsvm,data_test_ocsvm)
#     df_cifar_scores["One_Class_NN_explicit-Linear-Train"] = result[0]
#     df_cifar_scores["One_Class_NN_explicit-Linear-Test"] =  result[1]
#
#     result = One_Class_NN_explicit_sigmoid(data_train_ocsvm,data_test_ocsvm)
#     df_cifar_scores["One_Class_NN_explicit-Sigmoid-Train"] = result[0]
#     df_cifar_scores["One_Class_NN_explicit-Sigmoid-Test"] = result[1]
#
#     print ("Finished One Class NN Explicit")
#
#     # result = tf_OneClass_NN_linear(data_train,data_test)
#     # df_cifar_scores["tf_OneClass_NN-Linear-Train"] = result[0]
#     # df_cifar_scores["tf_OneClass_NN-Linear-Test"] =  result[1]
#     #
#     # result = tf_OneClass_NN_sigmoid(data_train,data_test)
#     # df_cifar_scores["tf_OneClass_NN-Sigmoid-Train"] = result[0]
#     # df_cifar_scores["tf_OneClass_NN-Sigmoid-Test"] = result[1]
#
#     result = tf_OneClass_CNN_linear(data_train, data_test)
#     df_cifar_scores["tf_OneClass_CNN-Linear-Train"] = result[0]
#     df_cifar_scores["tf_OneClass_CNN-Linear-Test"] = result[1]
#     print ("Finished tf_OneClass_CNN_linear")
#
#     result = tf_OneClass_CNN_sigmoid(data_train, data_test)
#     df_cifar_scores["tf_OneClass_CNN-Sigmoid-Train"] = result[0]
#     df_cifar_scores["tf_OneClass_CNN-Sigmoid-Test"] = result[1]
#     print ("Finished tf_OneClass_CNN_sigmoid")
#
#
#
#     # Y = labels_train
#     # labels_train = [[i] for i in Y]
#     # result = tflearn_OneClass_NN_linear(data_train,data_test,labels_train)
#     # df_cifar_scores["tflearn_OneClass_NN-Linear-Train"] = result[0]
#     # df_cifar_scores["tflearn_OneClass_NN-Linear-Test"] =  result[1]
#
#     # result = tflearn_OneClass_NN_Sigmoid(data_train,data_test,labels_train)
#     # df_cifar_scores["tflearn_OneClass_NN-Sigmoid-Train"] = result[0]
#     # df_cifar_scores["tflearn_OneClass_NN-Sigmoid-Test"] = result[1]
#     # print ("Finished tflearn_OneClass")
#
#     # print (type(df_cifar_scores))
#     # print ( (df_cifar_scores.keys()))
#
#     return df_cifar_scores
