# USAGE
# python test_network.py --model dog_not_dog.model --image images/examples/dog_01.png

# import the necessary packages

import numpy as np
import tensorflow as tf
from keras import backend as K

import time

## Declare the scoring functions
g   = lambda x : 1/(1 + tf.exp(-x))
#g  = lambda x : x # Linear
def nnScore(X, w, V, g):

    # print "X",X.shape
    # print "w",w[0].shape
    # print "v",V[0].shape
    return tf.matmul(g((tf.matmul(X, w))), V)
def relu(x):
    y = x
    y[y < 0] = 0
    return y
import csv
from itertools import izip_longest
import matplotlib as plt
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
decision_scorePath = "/Users/raghav/Documents/Uni/oc-nn/Decision_Scores/cifar/"

def tf_OneClass_NN_linear(data_train,data_test,nu):



    tf.reset_default_graph()

    RANDOM_SEED = 42
    tf.set_random_seed(RANDOM_SEED)

    train_X = data_train

     # Layer's sizes
    x_size = train_X.shape[1]   # Number of input nodes: 4 features and 1 bias
    print  "Input Shape:",x_size
    h_size = 16                # Number of hidden nodes
    y_size = 1   # Number of outcomes (3 iris flowers)
    D = x_size
    K = h_size

    theta = np.random.normal(0, 1, K + K*D + 1)
    rvalue = np.random.normal(0,1,(len(train_X),y_size))
    # nu = 0.1



    def init_weights(shape):
        """ Weight initialization """
        weights = tf.random_normal(shape,mean=0, stddev=1)
        return tf.Variable(weights)

    def forwardprop(X, w_1, w_2):
        """
        Forward-propagation.
        IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
        """
        X = tf.cast(X, tf.float32)
        w_1 = tf.cast(w_1, tf.float32)
        w_2 = tf.cast(w_2, tf.float32)
        h    = (tf.matmul(X, w_1))  #
        yhat = tf.matmul(h, w_2)  # The \varphi function
        return yhat

    g   = lambda x : x

    def nnScore(X, w, V, g):
        X = tf.cast(X, tf.float32)
        w = tf.cast(w, tf.float32)
        V = tf.cast(V, tf.float32)
        return tf.matmul(g((tf.matmul(X, w))), V)

    def relu1(x):
        y = x
        y = tf.nn.relu(x)
        return y

    def relu(x):

        with sess.as_default():
            x = x.eval()
        y = x

        y[y< 0] = 0
        # y = tf.nn.relu(x)

        return y

    def ocnn_obj(theta, X, nu, w1, w2, g,r):

        w = w1
        V = w2


        X = tf.cast(X, tf.float32)
        w = tf.cast(w1, tf.float32)
        V = tf.cast(w2, tf.float32)


        term1 = 0.5  * tf.reduce_sum(w**2)
        term2 = 0.5  * tf.reduce_sum(V**2)

        term3 = 1/nu * tf.reduce_mean(tf.nn.relu(r - nnScore(X, w, V, g)))

        term4 = -r

        return term1 + term2 + term3 + term4




    # For testing the algorithm
    test_X = data_test


    # Symbols
    X = tf.placeholder("float32", shape=[None, x_size])

    r = tf.get_variable("r", dtype=tf.float32,shape=(),trainable=False)

    # Weight initializations
    w_1 = init_weights((x_size, h_size))
    w_2 = init_weights((h_size, y_size))

    # Forward propagation
    # yhat    = forwardprop(X, w_1, w_2)
    # predict = tf.argmax(yhat, axis=1)


    # Backward propagation
    # cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    cost    = ocnn_obj(theta, X, nu, w_1, w_2, g,r)


    updates = tf.train.AdamOptimizer(0.05).minimize(cost)

    # Run optimization routine after initialization
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    rvalue = 0.1
    start_time = time.time()
    for epoch in range(100):
            # Train with each example
                sess.run(updates, feed_dict={X: train_X,r:rvalue})
                rvalue = nnScore(train_X, w_1, w_2, g)
                with sess.as_default():
                    rvalue = rvalue.eval()
                    rvalue = np.percentile(rvalue,q=100*nu)
                print("Epoch = %d, r = %f"
                  % (epoch + 1,rvalue))

    trainTime = time.time() - start_time
    ### Get the optimized weights here

    start_time = time.time()
    train = nnScore(train_X, w_1, w_2, g)
    test = nnScore(test_X, w_1, w_2, g)
    testTime = time.time() - start_time
    with sess.as_default():
        arrayTrain = train.eval()
        arrayTest = test.eval()
    #     rstar = r.eval()

    rstar =rvalue
    sess.close()
    print "Session Closed!!!"


    pos_decisionScore = arrayTrain-rstar
    pos_decisionScore[pos_decisionScore < 0] = 0
    neg_decisionScore = arrayTest-rstar
    print "&&&&&&&&&&&&"
    print pos_decisionScore
    print neg_decisionScore

    # write_decisionScores2Csv(decision_scorePath, "OneClass_NN_linear.csv", pos_decisionScore, neg_decisionScore)

    return [pos_decisionScore, neg_decisionScore,trainTime,testTime]

def tf_OneClass_NN_sigmoid(data_train,data_test,nu):



    tf.reset_default_graph()
    sess = tf.Session()

    train_X = data_train

    RANDOM_SEED = 42
    tf.set_random_seed(RANDOM_SEED)

     # Layer's sizes
    x_size = train_X.shape[1]   # Number of input nodes: 4 features and 1 bias
    print  "Input Shape:", x_size
    h_size = 16                # Number of hidden nodes
    y_size = 1   # Number of outcomes (3 iris flowers)
    D = x_size
    K = h_size

    theta = np.random.normal(0, 1, K + K*D + 1)
    rvalue = np.random.normal(0,1,(len(train_X),y_size))
    # nu = 0.1
    import math
    def plotNNFilter(units):
        filters = 3
        fig = plt.figure(1, figsize=(20, 20))
        n_columns = 6
        n_rows = math.ceil(filters / n_columns) + 1
        for i in range(filters):
            plt.subplot(n_rows, n_columns, i + 1)
            plt.title('Filter ' + str(i))
            plt.imshow(units[0, :, :, i], interpolation="nearest", cmap="gray")
        plt.savefig('/Users/raghav/Documents/Uni/oc-nn/models/representation_sigmoid_dog.png')

        # def getActivations(layer, stimuli):
    #     units = sess.run(layer, feed_dict={x: np.reshape(stimuli, [1, 784], order='F'), keep_prob: 1.0})
    #     plotNNFilter(units)

    def init_weights(shape):
        """ Weight initialization """
        weights = tf.random_normal(shape,mean=0, stddev=0.00001)
        return tf.Variable(weights)

    def forwardprop(X, w_1, w_2):
        """
        Forward-propagation.
        IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
        """
        X = tf.cast(X, tf.float32)
        w_1 = tf.cast(w_1, tf.float32)
        w_2 = tf.cast(w_2, tf.float32)
        h    = tf.nn.sigmoid(tf.matmul(X, w_1))  # The \sigma function
        yhat = tf.matmul(h, w_2)  # The \varphi function
        return yhat

    g   = lambda x : 1/(1 + tf.exp(-x))

    def nnScore(X, w, V, g):
        X = tf.cast(X, tf.float32)
        w = tf.cast(w, tf.float32)
        V = tf.cast(V, tf.float32)
        return tf.matmul(g((tf.matmul(X, w))), V)

    def data_rep(X, w, V, g):
        X = tf.cast(X, tf.float32)
        w = tf.cast(w, tf.float32)
        return g((tf.matmul(X, w)))

    def relu(x):


        y = tf.nn.relu(x)

        return y

    def ocnn_obj(theta, X, nu, w1, w2, g,r):

        w = w1
        V = w2


        X = tf.cast(X, tf.float32)
        w = tf.cast(w1, tf.float32)
        V = tf.cast(w2, tf.float32)


        term1 = 0.5  * tf.reduce_sum(w**2)
        term2 = 0.5  * tf.reduce_sum(V**2)
        term3 = 1/nu * tf.reduce_mean(relu(r - nnScore(X, w, V, g)))

        term4 = -r

        return term1 + term2 + term3 + term4




    # For testing the algorithm
    test_X = data_test


    # Symbols
    X = tf.placeholder("float32", shape=[None, x_size])

    r = tf.get_variable("r", dtype=tf.float32,shape=(),trainable=False)

    # Weight initializations
    w_1 = init_weights((x_size, h_size))
    w_2 = init_weights((h_size, y_size))

    # Forward propagation
    yhat    = forwardprop(X, w_1, w_2)
    predict = tf.argmax(yhat, axis=1)


    # Backward propagation
    # cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    cost    = ocnn_obj(theta, X, nu, w_1, w_2, g,r)
    updates = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)

    # Run SGD

    init = tf.global_variables_initializer()
    sess.run(init)
    rvalue = 0.1
    start_time = time.time()
    for epoch in range(100):
            # Train with each example
                units = sess.run(updates, feed_dict={X: train_X,r:rvalue})
                # plotNNFilter(units)
                with sess.as_default():
                    w1 = w_1.eval()
                    w2 = w_2.eval()
                rvalue = nnScore(train_X, w1, w2, g)

                with sess.as_default():
                    rvalue = rvalue.eval()
                    rvalue = np.percentile(rvalue,q=100*nu)
                print("Epoch = %d, r = %f"
                  % (epoch + 1,rvalue))
    trainTime = time.time() - start_time
    with sess.as_default():
        w1 = w_1.eval()
        w2 = w_2.eval()



    start_time = time.time()
    train = nnScore(train_X, w1, w2, g)
    test = nnScore(test_X, w1, w2, g)
    train_rep = data_rep(train_X, w1, w2, g)
    test_rep = data_rep(test_X, w1, w2, g)

    testTime = time.time() - start_time
    with sess.as_default():
        arrayTrain = train.eval()
        arrayTest = test.eval()
        arraytrain_rep =train_rep.eval()
        arraytest_rep= test_rep.eval()
    #     rstar = r.eval()

    rstar =rvalue
    sess.close()
    print "Session Closed!!!"

    print "Saving Hidden layer weights w1 for cifar.. data"
    import scipy.io as sio
    sio.savemat('/Users/raghav/Documents/Uni/oc-nn/models/w1.mat', {'data': arraytrain_rep})
    sio.savemat('/Users/raghav/Documents/Uni/oc-nn/models/w2.mat', {'data': arraytest_rep})


    pos_decisionScore = arrayTrain-rstar
    pos_decisionScore[pos_decisionScore< 0] = 0 ## Clip all the negative values to zero
    neg_decisionScore = arrayTest-rstar

    # write_decisionScores2Csv(decision_scorePath, "OneClass_NN_sigmoid.csv", pos_decisionScore, neg_decisionScore)

    return [pos_decisionScore, neg_decisionScore,trainTime,testTime]

def tf_OneClass_NN_relu(data_train,data_test,nu):



    tf.reset_default_graph()
    sess = tf.Session()

    train_X = data_train

    RANDOM_SEED = 42
    tf.set_random_seed(RANDOM_SEED)

     # Layer's sizes
    x_size = train_X.shape[1]   # Number of input nodes: 4 features and 1 bias
    print  "Input Shape:", x_size
    h_size = 16                # Number of hidden nodes
    y_size = 1   # Number of outcomes (3 iris flowers)
    D = x_size
    K = h_size

    theta = np.random.normal(0, 1, K + K*D + 1)
    rvalue = np.random.normal(0,1,(len(train_X),y_size))
    # nu = 0.1



    def init_weights(shape):
        """ Weight initialization """
        weights = tf.random_normal(shape,mean=0, stddev=0.00001)
        return tf.Variable(weights)

    def forwardprop(X, w_1, w_2):
        """
        Forward-propagation.
        IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
        """
        X = tf.cast(X, tf.float32)
        w_1 = tf.cast(w_1, tf.float32)
        w_2 = tf.cast(w_2, tf.float32)
        h    = tf.nn.sigmoid(tf.matmul(X, w_1))  # The \sigma function
        yhat = tf.matmul(h, w_2)  # The \varphi function
        return yhat

    g   = lambda x : relu(x)

    def nnScore(X, w, V, g):
        X = tf.cast(X, tf.float32)
        w = tf.cast(w, tf.float32)
        V = tf.cast(V, tf.float32)
        return tf.matmul(g((tf.matmul(X, w))), V)

    def relu(x):


        y = tf.nn.relu(x)

        return y

    def ocnn_obj(theta, X, nu, w1, w2, g,r):

        w = w1
        V = w2


        X = tf.cast(X, tf.float32)
        w = tf.cast(w1, tf.float32)
        V = tf.cast(w2, tf.float32)


        term1 = 0.5  * tf.reduce_sum(w**2)
        term2 = 0.5  * tf.reduce_sum(V**2)
        term3 = 1/nu * tf.reduce_mean(relu(r - nnScore(X, w, V, g)))

        term4 = -r

        return term1 + term2 + term3 + term4




    # For testing the algorithm
    test_X = data_test


    # Symbols
    X = tf.placeholder("float32", shape=[None, x_size])

    r = tf.get_variable("r", dtype=tf.float32,shape=(),trainable=False)

    # Weight initializations
    w_1 = init_weights((x_size, h_size))
    w_2 = init_weights((h_size, y_size))

    # Forward propagation
    yhat    = forwardprop(X, w_1, w_2)
    predict = tf.argmax(yhat, axis=1)


    # Backward propagation
    # cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    cost    = ocnn_obj(theta, X, nu, w_1, w_2, g,r)
    updates = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)

    # Run SGD

    init = tf.global_variables_initializer()
    sess.run(init)
    rvalue = 0.1
    for epoch in range(100):
            # Train with each example
                sess.run(updates, feed_dict={X: train_X,r:rvalue})
                with sess.as_default():
                    w1 = w_1.eval()
                    w2 = w_2.eval()
                rvalue = nnScore(train_X, w1, w2, g)

                with sess.as_default():
                    rvalue = rvalue.eval()
                    rvalue = np.percentile(rvalue,q=100*nu)
                print("Epoch = %d, r = %f"
                  % (epoch + 1,rvalue))

    with sess.as_default():
        w1 = w_1.eval()
        w2 = w_2.eval()

    train = nnScore(train_X, w1, w2, g)
    test = nnScore(test_X, w1, w2, g)
    with sess.as_default():
        arrayTrain = train.eval()
        arrayTest = test.eval()
    #     rstar = r.eval()

    rstar =rvalue
    sess.close()
    print "Session Closed!!!"


    pos_decisionScore = arrayTrain-rstar
    pos_decisionScore[pos_decisionScore< 0] = 0 ## Clip all the negative values to zero
    neg_decisionScore = arrayTest-rstar


    return [pos_decisionScore,neg_decisionScore]

