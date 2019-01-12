# USAGE
# python test_network.py --model dog_not_dog.model --image images/examples/dog_01.png

# import the necessary packages

from keras.preprocessing.image import img_to_array
import numpy as np
import cv2
from imutils import paths
import os
from keras.optimizers import SGD
import keras.backend as K
from keras.models import load_model
from keras.losses import customLoss
import tensorflow as tf
from keras.models import Model
import keras
from keras.layers import Dense
from keras.layers import Dense, GlobalAveragePooling2D, Activation
from keras.applications.vgg16 import preprocess_input
import time

activations = ["linear", "rbf"]
cifar_modelpath = "/Users/raghav/Documents/Uni/oc-nn/models/transfer_learning/cae_autoencoders/trained_models/cifar10_conv_3_id_32_e_1000_encoder.model"
usps_modelpath = "/Users/raghav/Documents/Uni/oc-nn/models/transfer_learning/cae_autoencoders/trained_models/cifar10_conv_3_id_32_e_1000_encoder.model"
MNIST_modelpath = "/Users/raghav/Documents/Uni/oc-nn/models/transfer_learning/cae_autoencoders/trained_models/cifar10_conv_3_id_32_e_1000_encoder.model"



def AE_OCSVM_RBF(data_train,data_test,nu):

    [X, X_test] = [data_train, data_test]
    print("[INFO] preparing the network to extract flattened inputs(X,X_test) for scoring..... done!!!")
    import matplotlib.pyplot as plt
    from sklearn import svm

    ocSVM = svm.OneClassSVM(nu=nu, kernel='rbf')
    start_time = time.time()
    ocSVM.fit(X)
    trainTime = time.time() - start_time
    activations = ["linear", "rbf"]
    start_time = time.time()
    pos_decisionScore = ocSVM.decision_function(X)
    neg_decisionScore = ocSVM.decision_function(X_test)
    testTime = time.time() - start_time

    print "AE_OCSVM_RBF+",pos_decisionScore
    print "AE_OCSVM_RBF-",neg_decisionScore

    return [pos_decisionScore, neg_decisionScore,trainTime,testTime]


def AE_OCSVM_Linear(data_train,data_test,nu):
    # train_path = "/Users/raghav/Documents/Uni/oc-nn/models/transfer_learning/cae_autoencoders/data/cifar-10/normal/"
    # test_path = "/Users/raghav/Documents/Uni/oc-nn/models/transfer_learning/cae_autoencoders/data/cifar-10/anoma/"

    # [X, X_test] = prepare_cifar_data_for_cae_ocsvm(train_path, test_path, cifar_modelpath)
    [X, X_test] = [data_train,data_test]
    print("[INFO] preparing the network to extract flattened inputs(X,X_test) for scoring..... done!!!")
    import matplotlib.pyplot as plt
    from sklearn import svm

    ocSVM = svm.OneClassSVM(nu=nu, kernel='linear')
    # ocSVM = svm.OneClassSVM(nu=nu, kernel='linear')
    start_time = time.time()
    ocSVM.fit(X)
    trainTime = time.time() - start_time
    activations = ["linear", "rbf"]
    start_time = time.time()
    pos_decisionScore = ocSVM.decision_function(X)
    neg_decisionScore = ocSVM.decision_function(X_test)
    testTime = time.time() - start_time

    print "AE_OCSVM_RBF+",pos_decisionScore
    print "AE_OCSVM_RBF-",neg_decisionScore

    return [pos_decisionScore, neg_decisionScore,trainTime,testTime]


def LSTMAE_OCSVM_RBF(encodedDataPath,nu):
    import numpy as np
    X = np.load(encodedDataPath)
    train_encoded = X[0:50]
    # test_encoded=X[50:60]
    a1 = X[50:51]
    a2 = X[51:52]
    a3 = X[52:53]
    a4 = X[53:54]
    a5 = X[54:55]
    a6 = X[55:56]
    a7 = X[56:57]
    a8 = X[57:58]
    a9 = X[58:59]
    a10 = X[59:60]
    # a1,a4,a9,a10,a7
    # a8,a6,a5,a3,a2
    anomalies = []
    anomalies.append(a1)
    anomalies.append(a4)
    anomalies.append(a9)
    anomalies.append(a10)
    anomalies.append(a7)

    test_encoded = np.asarray(anomalies)
    test_encoded = np.reshape(test_encoded, (5, 240))
    print "Encoded Training samples:", train_encoded.shape
    print "Encoded Testing samples:", test_encoded.shape

    # Preprocess the inputs
    X = train_encoded
    X_test = test_encoded

    print X.shape
    print X_test.shape

    print("[INFO] preparing the network to extract flattened inputs(X,X_test) for scoring..... done!!!")
    import matplotlib.pyplot as plt
    from sklearn import svm
    ocSVM = svm.OneClassSVM(nu=nu, kernel='rbf')
    # ocSVM = svm.OneClassSVM(nu=nu, kernel='linear')
    ocSVM.fit(X)
    activations = ["linear", "rbf"]
    pos_decisionScore = ocSVM.decision_function(X)
    neg_decisionScore = ocSVM.decision_function(X_test)
    print pos_decisionScore
    print neg_decisionScore

    return [pos_decisionScore, neg_decisionScore]


    return [pos_decisionScore, neg_decisionScore]

def LSTMAE_OCSVM_Linear(encodedDataPath,nu):
    import numpy as np
    X = np.load(encodedDataPath)
    train_encoded = X[0:50]
    # test_encoded=X[50:60]
    a1 = X[50:51]
    a2 = X[51:52]
    a3 = X[52:53]
    a4 = X[53:54]
    a5 = X[54:55]
    a6 = X[55:56]
    a7 = X[56:57]
    a8 = X[57:58]
    a9 = X[58:59]
    a10 = X[59:60]
    # a1,a4,a9,a10,a7
    # a8,a6,a5,a3,a2
    anomalies = []
    anomalies.append(a1)
    anomalies.append(a4)
    anomalies.append(a9)
    anomalies.append(a10)
    anomalies.append(a7)

    test_encoded = np.asarray(anomalies)
    test_encoded = np.reshape(test_encoded, (5, 240))
    print "Encoded Training samples:", train_encoded.shape
    print "Encoded Testing samples:", test_encoded.shape

    # Preprocess the inputs
    X = train_encoded
    X_test = test_encoded

    print X.shape
    print X_test.shape

    print("[INFO] preparing the network to extract flattened inputs(X,X_test) for scoring..... done!!!")
    import matplotlib.pyplot as plt
    from sklearn import svm
    ocSVM = svm.OneClassSVM(nu=nu, kernel='linear')
    # ocSVM = svm.OneClassSVM(nu=nu, kernel='linear')
    ocSVM.fit(X)
    activations = ["linear", "rbf"]
    pos_decisionScore = ocSVM.decision_function(X)
    neg_decisionScore = ocSVM.decision_function(X_test)
    print pos_decisionScore
    print neg_decisionScore

    return [pos_decisionScore, neg_decisionScore]

def add_new_last_layer(base_model_output,base_model_input):
  """Add last layer to the convnet
  Args:
    base_model: keras model excluding top
    nb_classes: # of classes
  Returns:
    new keras model with last layer
  """
  x = base_model_output
  print "base_model.output",x.shape
  inp = base_model_input
  print "base_model.input",inp.shape
  dense1 = Dense(512, name="dense_output1")(x)  # new sigmoid layer
  dense1out = Activation("relu", name="output_activation1")(dense1)
  dense2 = Dense(1, name="dense_output2")(dense1out) #new sigmoid layer
  dense2out = Activation("relu",name="output_activation2")(dense2)  # new sigmoid layer
  model = Model(inputs=inp, outputs=dense2out)
  return model


def prepare_data_LSTM_AE_OCSVM(encodedDataPath):
    # Basic libraries
    import numpy as np
    X = np.load(encodedDataPath)
    train_encoded = X[0:50]
    #test_encoded=  X[50:60]
    a1 = X[50:51]
    a2 = X[51:52]
    a3 = X[52:53]
    a4 = X[53:54]
    a5 = X[54:55]
    a6 = X[55:56]
    a7 = X[56:57]
    a8 = X[57:58]
    a9 = X[58:59]
    a10 = X[59:60]
    # a1,a4,a9,a10,a7
    # a8,a6,a5,a3,a2
    anomalies = []
    anomalies.append(a1)
    anomalies.append(a4)
    anomalies.append(a9)
    anomalies.append(a10)
    anomalies.append(a7)

    test_encoded = np.asarray(anomalies)
    test_encoded = np.reshape(test_encoded, (5, 240))
    print "Encoded Training samples:", train_encoded.shape
    print "Encoded Testing samples:", test_encoded.shape

    import tensorflow as tf

    # Preprocess the inputs
    X = train_encoded
    X_test = test_encoded

    print X.shape
    print X_test.shape

    print X.shape
    print X_test.shape

    return [X, X_test]



def tf_OneClass_LSTM_AE_NN_linear(data_train, data_test, nu):
    tf.reset_default_graph()

    RANDOM_SEED = 42
    tf.set_random_seed(RANDOM_SEED)

    train_X = data_train

    # Layer's sizes
    x_size = train_X.shape[1]  # Number of input nodes: 4 features and 1 bias
    print  "Input Shape:", x_size
    h_size = 240  # Number of hidden nodes
    y_size = 1  # Number of outcomes (3 iris flowers)
    D = x_size
    K = h_size

    theta = np.random.normal(0, 1, K + K * D + 1)
    rvalue = np.random.normal(0, 1, (len(train_X), y_size))

    def init_weights(shape):
        """ Weight initialization """
        weights = tf.random_normal(shape, mean=0, stddev=0.5)
        return tf.Variable(weights)

    def relu(x):
        y = x

        y = tf.nn.relu(x)

        return y

    def forwardprop(X, w_1, w_2):
        """
        Forward-propagation.
        IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
        """
        X = tf.cast(X, tf.float32)
        w_1 = tf.cast(w_1, tf.float32)
        w_2 = tf.cast(w_2, tf.float32)
        h = (tf.matmul(X, w_1))  #
        yhat = tf.matmul(h, w_2)  # The \varphi function
        return yhat

    g = lambda x: x

    def nnScore(X, w, V, g):
        X = tf.cast(X, tf.float32)
        w = tf.cast(w, tf.float32)
        V = tf.cast(V, tf.float32)
        return tf.matmul(g((tf.matmul(X, w))), V)

    def relu1(x):
        y = x
        y = tf.nn.relu(x)
        return y

    def ocnn_obj(theta, X, nu, w1, w2, g, r):
        w = w1
        V = w2

        X = tf.cast(X, tf.float32)
        w = tf.cast(w1, tf.float32)
        V = tf.cast(w2, tf.float32)

        term1 = 0.5 * tf.reduce_sum(w ** 2)
        term2 = 0.5 * tf.reduce_sum(V ** 2)

        term3 = 1 / nu * tf.reduce_mean(tf.nn.relu(r - nnScore(X, w, V, g)))

        term4 = -r

        return term1 + term2 + term3 + term4

    # For testing the algorithm
    test_X = data_test

    # Symbols
    X = tf.placeholder("float32", shape=[None, x_size])

    r = tf.get_variable("r", dtype=tf.float32, shape=(), trainable=False)

    # Weight initializations
    w_1 = init_weights((x_size, h_size))
    w_2 = init_weights((h_size, y_size))

    # Forward propagation
    # yhat    = forwardprop(X, w_1, w_2)
    # predict = tf.argmax(yhat, axis=1)


    # Backward propagation
    # cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    cost = ocnn_obj(theta, X, nu, w_1, w_2, g, r)

    updates = tf.train.AdamOptimizer(1e-2).minimize(cost)

    # Run optimization routine after initialization
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    rvalue = 0.1
    for epoch in range(100):
        # Train with each example
        sess.run(updates, feed_dict={X: train_X, r: rvalue})
        rvalue = nnScore(train_X, w_1, w_2, g)
        with sess.as_default():
            rvalue = rvalue.eval()
            rvalue = np.percentile(rvalue, q=100 * nu)
        print("Epoch = %d, r = %f"
              % (epoch + 1, rvalue))

    ### Get the optimized weights here


    train = nnScore(train_X, w_1, w_2, g)
    test = nnScore(test_X, w_1, w_2, g)
    with sess.as_default():
        arrayTrain = train.eval()
        arrayTest = test.eval()
    # rstar = r.eval()

    rstar = rvalue
    sess.close()
    print "Session Closed!!!"

    pos_decisionScore = arrayTrain - rstar
    pos_decisionScore[pos_decisionScore < 0] = 0
    neg_decisionScore = arrayTest - rstar

    return [pos_decisionScore, neg_decisionScore]


def tf_OneClass_LSTM_AE_NN_sigmoid(data_train, data_test, nu):
    tf.reset_default_graph()
    sess = tf.Session()

    train_X = data_train

    RANDOM_SEED = 42
    tf.set_random_seed(RANDOM_SEED)

    # Layer's sizes
    x_size = train_X.shape[1]  # Number of input nodes: 4 features and 1 bias
    print  "Input Shape:", x_size
    h_size = 240  # Number of hidden nodes
    y_size = 1  # Number of outcomes (3 iris flowers)
    D = x_size
    K = h_size

    theta = np.random.normal(0, 1, K + K * D + 1)
    rvalue = np.random.normal(0, 1, (len(train_X), y_size))

    def init_weights(shape):
        """ Weight initialization """
        weights = tf.random_normal(shape, mean=0, stddev=0.1)
        return tf.Variable(weights)

    def forwardprop(X, w_1, w_2):
        """
        Forward-propagation.
        IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
        """
        X = tf.cast(X, tf.float32)
        w_1 = tf.cast(w_1, tf.float32)
        w_2 = tf.cast(w_2, tf.float32)
        h = tf.nn.sigmoid(tf.matmul(X, w_1))  # The \sigma function
        yhat = tf.matmul(h, w_2)  # The \varphi function
        return yhat

    g = lambda x: 1 / (1 + tf.exp(-x))

    def nnScore(X, w, V, g):
        X = tf.cast(X, tf.float32)
        w = tf.cast(w, tf.float32)
        V = tf.cast(V, tf.float32)
        return tf.matmul(g((tf.matmul(X, w))), V)

    def relu(x):
        y = tf.nn.relu(x)

        return y

    def ocnn_obj(theta, X, nu, w1, w2, g, r):
        w = w1
        V = w2

        X = tf.cast(X, tf.float32)
        w = tf.cast(w1, tf.float32)
        V = tf.cast(w2, tf.float32)

        term1 = 0.5 * tf.reduce_sum(w ** 2)
        term2 = 0.5 * tf.reduce_sum(V ** 2)
        term3 = 1 / nu * tf.reduce_mean(relu(r - nnScore(X, w, V, g)))

        term4 = -r

        return term1 + term2 + term3 + term4

    # For testing the algorithm
    test_X = data_test

    # Symbols
    X = tf.placeholder("float32", shape=[None, x_size])

    r = tf.get_variable("r", dtype=tf.float32, shape=(), trainable=False)

    # Weight initializations
    w_1 = init_weights((x_size, h_size))
    w_2 = init_weights((h_size, y_size))

    # Forward propagation
    yhat = forwardprop(X, w_1, w_2)
    predict = tf.argmax(yhat, axis=1)

    # Backward propagation
    # cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    cost = ocnn_obj(theta, X, nu, w_1, w_2, g, r)
    # updates = tf.train.GradientDescentOptimizer(0.006).minimize(cost)
    updates = tf.train.AdamOptimizer(1e-2).minimize(cost)

    # Run SGD

    init = tf.global_variables_initializer()
    sess.run(init)
    rvalue = 0.1
    for epoch in range(100):
        # Train with each example
        sess.run(updates, feed_dict={X: train_X, r: rvalue})
        with sess.as_default():
            w1 = w_1.eval()
            w2 = w_2.eval()
        rvalue = nnScore(train_X, w1, w2, g)

        with sess.as_default():
            rvalue = rvalue.eval()
            rvalue = np.percentile(rvalue, q=100 * nu)
        print("Epoch = %d, r = %f"
              % (epoch + 1, rvalue))

    with sess.as_default():
        w1 = w_1.eval()
        w2 = w_2.eval()

    train = nnScore(train_X, w1, w2, g)
    test = nnScore(test_X, w1, w2, g)
    with sess.as_default():
        arrayTrain = train.eval()
        arrayTest = test.eval()
    # rstar = r.eval()

    rstar = rvalue
    sess.close()
    print "Session Closed!!!"

    pos_decisionScore = arrayTrain - rstar
    pos_decisionScore[pos_decisionScore < 0] = 0  ## Clip all the negative values to zero
    neg_decisionScore = arrayTest - rstar

    return [pos_decisionScore, neg_decisionScore]



