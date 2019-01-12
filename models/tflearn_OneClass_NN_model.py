import numpy as np
import pandas as pd
from sklearn import utils
import matplotlib
from scipy.optimize import minimize
from tflearn import DNN
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression, oneClassNN
import tensorflow as tf
import tflearn
import numpy as np
import tflearn.variables as va
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as srn


dataPath = './data/'

colNames = ["sklearn-OCSVM-Linear-Train","sklearn-OCSVM-RBF-Train","sklearn-OCSVM-Linear-Test","sklearn-OCSVM-RBF-Test","sklearn-explicit-Linear-Train","sklearn-explicit-Sigmoid-Train","sklearn-explicit-Linear-Test","sklearn-explicit-Sigmoid-Test","tf-Linear-Train","tf-Sigmoid-Train","tf-Linear-Test","tf-Sigmoid-Test","tfLearn-Linear-Train","tfLearn-Sigmoid-Train","tfLearn-Linear-Test","tfLearn-Sigmoid-Test"]

# Create empty dataframe with given column names.
df_usps_scores = {}
df_fake_news_scores = {}
df_spam_vs_ham_scores = {}
df_cifar_10_scores = {}
nu = 0.04
K  = 4
# Hyper parameters for the one class Neural Network
v = 0.04




def tflearn_OneClass_NN_linear(data_train,data_test,labels_train):

    X  = data_train
    Y = labels_train

    D  = X.shape[1]

    No_of_inputNodes = X.shape[1]

    # Clear all the graph variables created in previous run and start fresh
    tf.reset_default_graph()

    # Define the network
    input_layer = input_data(shape=[None, No_of_inputNodes])  # input layer of size

    np.random.seed(42)
    theta0 = np.random.normal(0, 1, K + K*D + 1) *0.0001
    #theta0 = np.random.normal(0, 1, K + K*D + 1) # For linear
    hidden_layer = fully_connected(input_layer, 4, bias=False, activation='linear', name="hiddenLayer_Weights",
                                   weights_init="normal")  # hidden layer of size 2


    output_layer = fully_connected(hidden_layer, 1, bias=False, activation='linear', name="outputLayer_Weights",
                                   weights_init="normal")  # output layer of size 1



    # Initialize rho
    value = 0.01
    init = tf.constant_initializer(value)
    rho = va.variable(name='rho', dtype=tf.float32, shape=[], initializer=init)

    rcomputed = []
    auc = []


    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    # print sess.run(tflearn.get_training_mode()) #False
    tflearn.is_training(True, session=sess)
    print sess.run(tflearn.get_training_mode())  #now True

    temp = theta0[-1]


    oneClassNN_Net = oneClassNN(output_layer, v, rho, hidden_layer, output_layer, optimizer='sgd',
                        loss='OneClassNN_Loss',
                            learning_rate=1)

    model = DNN(oneClassNN_Net, tensorboard_verbose=3)

    model.set_weights(output_layer.W, theta0[0:K][:,np.newaxis])
    model.set_weights(hidden_layer.W, np.reshape(theta0[K:K +K*D],(D,K)))


    iterStep = 0
    while (iterStep < 100):
        print "Running Iteration :", iterStep
        # Call the cost function
        y_pred = model.predict(data_train)  # Apply some ops
        tflearn.is_training(False, session=sess)
        y_pred_test = model.predict(data_test)  # Apply some ops
        tflearn.is_training(True, session=sess)
        value = np.percentile(y_pred, v * 100)
        tflearn.variables.set_value(rho, value,session=sess)
        rStar = rho
        model.fit(X, Y, n_epoch=2, show_metric=True, batch_size=100)
        iterStep = iterStep + 1
        rcomputed.append(rho)
        temp = tflearn.variables.get_value(rho, session=sess)

    # print "Rho",temp
    # print "y_pred",y_pred
    # print "y_predTest", y_pred_test

    # g = lambda x: x
    g   = lambda x : 1/(1 + tf.exp(-x))

    def nnScore(X, w, V, g):
        return tf.matmul(g((tf.matmul(X, w))), V)


    # Format the datatype to suite the computation of nnscore
    X = X.astype(np.float32)
    X_test = data_test
    X_test = X_test.astype(np.float32)
    # assign the learnt weights
    # wStar = hidden_layer.W
    # VStar = output_layer.W
    # Get weights values of fc2
    wStar = model.get_weights(hidden_layer.W)
    VStar = model.get_weights(output_layer.W)

    # print "Hideen",wStar
    # print VStar

    train = nnScore(X, wStar, VStar, g)
    test = nnScore(X_test, wStar, VStar, g)

    # Access the value inside the train and test for plotting
    # Create a new session and run the example
    # sess = tf.Session()
    # sess.run(tf.initialize_all_variables())
    arrayTrain = train.eval(session=sess)
    arrayTest = test.eval(session=sess)

    # print "Train Array:",arrayTrain
    # print "Test Array:",arrayTest

    # plt.hist(arrayTrain-temp,  bins = 25,label='Normal');
    # plt.hist(arrayTest-temp, bins = 25, label='Anomalies');
    # plt.legend(loc='upper right')
    # plt.title('r = %1.6f- Sigmoid Activation ' % temp)
    # plt.show()

    pos_decisionScore = arrayTrain-temp
    neg_decisionScore = arrayTest-temp

    return [pos_decisionScore,neg_decisionScore]




def tflearn_OneClass_NN_Sigmoid(data_train,data_test,labels_train):

    X  = data_train
    Y = labels_train

    D  = X.shape[1]

    No_of_inputNodes = X.shape[1]

    # Clear all the graph variables created in previous run and start fresh
    tf.reset_default_graph()

    # Define the network
    input_layer = input_data(shape=[None, No_of_inputNodes])  # input layer of size

    np.random.seed(42)
    theta0 = np.random.normal(0, 1, K + K*D + 1) *0.0001
    #theta0 = np.random.normal(0, 1, K + K*D + 1) # For linear
    hidden_layer = fully_connected(input_layer, 4, bias=False, activation='sigmoid', name="hiddenLayer_Weights",
                                   weights_init="normal")  # hidden layer of size 2


    output_layer = fully_connected(hidden_layer, 1, bias=False, activation='linear', name="outputLayer_Weights",
                                   weights_init="normal")  # output layer of size 1



    # Initialize rho
    value = 0.01
    init = tf.constant_initializer(value)
    rho = va.variable(name='rho', dtype=tf.float32, shape=[], initializer=init)

    rcomputed = []
    auc = []


    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    # print sess.run(tflearn.get_training_mode()) #False
    tflearn.is_training(True, session=sess)
    print sess.run(tflearn.get_training_mode())  #now True

    temp = theta0[-1]


    oneClassNN_net = oneClassNN(output_layer, v, rho, hidden_layer, output_layer, optimizer='sgd',
                        loss='OneClassNN_Loss',
                            learning_rate=1)

    model = DNN(oneClassNN_net, tensorboard_verbose=3)

    model.set_weights(output_layer.W, theta0[0:K][:,np.newaxis])
    model.set_weights(hidden_layer.W, np.reshape(theta0[K:K +K*D],(D,K)))


    iterStep = 0
    while (iterStep < 100):
        print "Running Iteration :", iterStep
        # Call the cost function
        y_pred = model.predict(data_train)  # Apply some ops
        tflearn.is_training(False, session=sess)
        y_pred_test = model.predict(data_test)  # Apply some ops
        tflearn.is_training(True, session=sess)
        value = np.percentile(y_pred, v * 100)
        tflearn.variables.set_value(rho, value,session=sess)
        rStar = rho
        model.fit(X, Y, n_epoch=2, show_metric=True, batch_size=100)
        iterStep = iterStep + 1
        rcomputed.append(rho)
        temp = tflearn.variables.get_value(rho, session=sess)

    # print "Rho",temp
    # print "y_pred",y_pred
    # print "y_predTest", y_pred_test

    # g = lambda x: x
    g   = lambda x : 1/(1 + tf.exp(-x))

    def nnScore(X, w, V, g):
        return tf.matmul(g((tf.matmul(X, w))), V)


    # Format the datatype to suite the computation of nnscore
    X = X.astype(np.float32)
    X_test = data_test
    X_test = X_test.astype(np.float32)
    # assign the learnt weights
    # wStar = hidden_layer.W
    # VStar = output_layer.W
    # Get weights values of fc2
    wStar = model.get_weights(hidden_layer.W)
    VStar = model.get_weights(output_layer.W)

    # print "Hideen",wStar
    # print VStar

    train = nnScore(X, wStar, VStar, g)
    test = nnScore(X_test, wStar, VStar, g)

    # Access the value inside the train and test for plotting
    # Create a new session and run the example
    # sess = tf.Session()
    # sess.run(tf.initialize_all_variables())
    arrayTrain = train.eval(session=sess)
    arrayTest = test.eval(session=sess)

    # print "Train Array:",arrayTrain
    # print "Test Array:",arrayTest

    # plt.hist(arrayTrain-temp,  bins = 25,label='Normal');
    # plt.hist(arrayTest-temp, bins = 25, label='Anomalies');
    # plt.legend(loc='upper right')
    # plt.title('r = %1.6f- Sigmoid Activation ' % temp)
    # plt.show()

    pos_decisionScore = arrayTrain-temp
    neg_decisionScore = arrayTest-temp

    return [pos_decisionScore,neg_decisionScore]


def func_getDecision_Scores_tflearn_OneClass_NN(dataset,data_train,data_test,labels_train):


    # print "Decision_Scores_sklearn_OCSVM Using Linear and RBF Kernels....."

    if(dataset=="USPS" ):

        Y = labels_train

        Y = Y.tolist()
        labels_train = [[i] for i in Y]
        result = tflearn_OneClass_NN_linear(data_train,data_test,labels_train)
        df_usps_scores["tflearn_OneClass_NN-Linear-Train"] = result[0]
        df_usps_scores["tflearn_OneClass_NN-Linear-Test"] =  result[1]
 
        result = tflearn_OneClass_NN_Sigmoid(data_train,data_test,labels_train)
        df_usps_scores["tflearn_OneClass_NN-Sigmoid-Train"] = result[0]
        df_usps_scores["tflearn_OneClass_NN-Sigmoid-Test"] = result[1]


    # if(dataset=="FAKE_NEWS" ): 
    #     Y = labels_train
    #     Y = Y.tolist()
    #     labels_train = [[i] for i in Y]  
        
    #     result = tflearn_OneClass_NN_linear(data_train,data_test,labels_train)
    #     df_fake_news_scores["tflearn_OneClass_NN-Linear-Train"] = result[0]
    #     df_fake_news_scores["tflearn_OneClass_NN-Linear-Test"] = result[1]
        
    #     result = tflearn_OneClass_NN_Sigmoid(data_train,data_test,labels_train)
    #     df_fake_news_scores["tflearn_OneClass_NN-Sigmoid-Train"] = result[0]
    #     df_fake_news_scores["tflearn_OneClass_NN-Sigmoid-Test"] = result[1]


    # if(dataset=="SPAM_Vs_HAM" ):
        # Y = labels_train

        # Y = Y.tolist()
        # labels_train = [[i] for i in Y]
    #     result = tflearn_OneClass_NN_linear(data_train,data_test)
    #     df_spam_vs_ham_scores["tflearn_OneClass_NN-Linear-Train"] = result[0] 
    #     df_spam_vs_ham_scores["tflearn_OneClass_NN-Linear-Test"] = result[1]
        
    #     result = tflearn_OneClass_NN_Sigmoid(data_train,data_test)
    #     df_spam_vs_ham_scores["tflearn_OneClass_NN-Sigmoid-Train"] = result[0]
    #     df_spam_vs_ham_scores["tflearn_OneClass_NN-Sigmoid-Test"] = result[1]


    # if(dataset=="CIFAR-10" ):
    #     Y = labels_train
    #     Y = Y.tolist()
    #     labels_train = [[i] for i in Y]
    #     result = tflearn_OneClass_NN_linear(data_train,data_test,labels_train)
    #     df_cifar_10_scores["tflearn_OneClass_NN-Linear-Train"] = result[0]
    #     df_cifar_10_scores["tflearn_OneClass_NN-Linear-Test"] = result[1]
        
    #     result = tflearn_OneClass_NN_Sigmoid(data_train,data_test,labels_train)
    #     df_cifar_10_scores["tflearn_OneClass_NN_Sigmoid-Train"] = result[0]
    #     df_cifar_10_scores["tflearn_OneClass_NN_Sigmoid-Test"] = result[1]

    return [df_usps_scores,df_fake_news_scores,df_spam_vs_ham_scores,df_cifar_10_scores]




