import numpy as np
import pandas as pd
from sklearn import utils
import matplotlib
from scipy.optimize import minimize
import tensorflow as tf
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from img_to_vec import Img2Vec
from PIL import Image
import glob


import prettytensor as pt

dataPathTrain = '/Users/raghav/Documents/Uni/oc-nn/data/cifar-10_data/train/dogs/'
dataPathTest = '/Users/raghav/Documents/Uni/oc-nn/data/cifar-10_data/test/'

colNames = ["sklearn-OCSVM-Linear-Train", "sklearn-OCSVM-RBF-Train", "sklearn-OCSVM-Linear-Test",
            "sklearn-OCSVM-RBF-Test", "sklearn-explicit-Linear-Train", "sklearn-explicit-Sigmoid-Train",
            "sklearn-explicit-Linear-Test", "sklearn-explicit-Sigmoid-Test", "tf-Linear-Train", "tf-Sigmoid-Train",
            "tf-Linear-Test", "tf-Sigmoid-Test", "tfLearn-Linear-Train", "tfLearn-Sigmoid-Train", "tfLearn-Linear-Test",
            "tfLearn-Sigmoid-Test"]

# Create empty dataframe with given column names.
df_usps_scores = {}
df_fake_news_scores = {}
df_spam_vs_ham_scores = {}
df_cifar_10_scores = {}
nu = 0.04
K = 4
# Layer's sizes
x_size = 2048  # Number of input nodes: 4 features and 1 bias
h_size = 256  # Number of hidden nodes
y_size = 1  # Number of outcomes (3 iris flowers)
D = x_size
K = h_size
img_size = 128
num_channels = 3
# Symbols
Imgs = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='Imgs')
X = tf.placeholder(name="X", dtype=tf.float32, shape=[None, x_size]) # for Convolution features
r = tf.get_variable(name="r", dtype=tf.float32, shape=(), trainable=False)
theta = np.random.normal(0, 1, K + K * D + 1)
nu = 0.04

### Learning Rate
lr_linear = 0.00001  #{0.001,0.00001
lr_sigmoid = 0.0001 #{0.001,0.09,0.0001
def func_get_ImageVectors(path):
    veclist = []
    # Initialize Img2Vec with GPU
    img2vec = Img2Vec(cuda=False)
    print path
    # Read in an image
    for filename in glob.iglob(path+'/*.jpg'):
      # print filename
      img = Image.open(filename)
      # Get a vector from img2vec
      vec = img2vec.get_vec(img)
      veclist.append(vec)
    #   print vec.shape
    #   print len(veclist)
    #
    # print "Above are vec details===>"
    # convert the veclist to numpy array
    data = np.asarray(veclist)
    # print "data----shape"
    # print data.shape

    return data


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

def g(x): return 1 / (1 + tf.exp(-x))


def getConv_features(images, training):

    # Wrap the input images as a Pretty Tensor object.
    x_pretty = pt.wrap(images)

    # Pretty Tensor uses special numbers to distinguish between
    # the training and testing phases.
    if training:
        phase = pt.Phase.train
    else:
        phase = pt.Phase.infer

    # Create the convolutional neural network using Pretty Tensor.
    # It is very similar to the previous tutorials, except
    # the use of so-called batch-normalization in the first layer.
    with pt.defaults_scope(activation_fn=tf.nn.relu, phase=phase):
        p_input_flatten = x_pretty. \
            conv2d(kernel=5, depth=64, name='layer_conv1_feature_extraction', batch_normalize=True). \
            max_pool(kernel=2, stride=2). \
            conv2d(kernel=5, depth=64, name='layer_conv2_feature_extraction'). \
            max_pool(kernel=2, stride=2). \
            flatten(). \
            fully_connected(size=256, name='layer_fc1_feature_extraction')

    return p_input_flatten


def create_ConvFeatureInputs(training):
    # Wrap the neural network in the scope named 'network'.
    # Create new variables during training, and re-use during testing.

    with tf.variable_scope('network', reuse=not training):
        # Just rename the input placeholder variable for convenience.
        images = Imgs

        input_flattened = getConv_features(images=images, training=training)

    return input_flattened
feature_inp = create_ConvFeatureInputs(training=True)

def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, mean=0, stddev=0.0001)
    return tf.Variable(weights)

def init_weights_1(shape):
    """ Weight initialization """
    init_w_1 = np.load("/Users/raghav/Documents/Uni/oc-nn/models/weights/inp_hidden.npy")
    return tf.Variable(init_w_1)

def init_weights_2(shape):
    """ Weight initialization """
    init_w_2 = np.load("/Users/raghav/Documents/Uni/oc-nn/models/weights/hid_out.npy")
    return tf.Variable(init_w_2)


def forwardprop(X, w_1, w_2):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    w_1 = tf.cast(w_1, tf.float32)
    w_2 = tf.cast(w_2, tf.float32)

    print "+++++++++++++++++++++"
    print w_1.get_shape()
    print w_2.get_shape()
    print X.get_shape()

    print "+++++++++++++++++++++"
    h = (tf.matmul(X, w_1))  #

    yhat = tf.matmul(h, w_2)  # The \varphi function
    return yhat

def nnScore(X, w, V, g):
    w = tf.cast(w, tf.float32)
    V = tf.cast(V, tf.float32)
    return tf.matmul(g((tf.matmul(X, w))), V)

def relu(x):
    y = x
    #     y[y < 0] = 0
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


def tf_OneClass_CNN_linear(data_train, data_test):


    RANDOM_SEED = 42
    tf.set_random_seed(RANDOM_SEED)

    # Weight initializations
    # w_1 = init_weights((x_size, h_size))
    # w_2 = init_weights((h_size, y_size))

    w_1 = init_weights_1((x_size, h_size))
    w_2 = init_weights_2((h_size, y_size))


    # Forward propagation
    yhat = forwardprop(X, w_1, w_2)
    predict = tf.argmax(yhat, axis=1)

    # Backward propagation
    # cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    def g(x): return x
    cost = ocnn_obj(theta, X, nu, w_1, w_2, g, r)
    updates = tf.train.GradientDescentOptimizer(lr_linear).minimize(cost)


    # Run SGD
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    # train_X = sess.run(feature_inp, feed_dict={Imgs: data_train})
    # test_X = sess.run(feature_inp, feed_dict={Imgs: data_test})

    # train_X = func_get_ImageVectors(dataPathTrain)
    # test_X = func_get_ImageVectors(dataPathTest)

    [train_X,test_X] = func_getKerasModelfeatures()



    print train_X.shape
    print test_X.shape

    # rvalue = np.random.normal(0, 1, (len(train_X), y_size))
    rvalue = 0.1
    for epoch in range(100):
        # Train with each example
        _,loss = sess.run([updates, cost], feed_dict={X: train_X, r: rvalue})
        rvalue = nnScore(train_X, w_1, w_2, g)
        with sess.as_default():
            # cost = cost.eval()
            rvalue = rvalue.eval()
            rvalue = np.percentile(rvalue, q=100 * 0.04)
        print("Epoch = %d, r = %f, loss = %s"
              % (epoch + 1, rvalue,loss))


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
    neg_decisionScore = arrayTest - rstar

    return [pos_decisionScore, neg_decisionScore]



def tf_OneClass_CNN_sigmoid(data_train, data_test):


    RANDOM_SEED = 42
    tf.set_random_seed(RANDOM_SEED)

    # Weight initializations
    # w_1 = init_weights((x_size, h_size))
    # w_2 = init_weights((h_size, y_size))

    # w_1 = np.load("/Users/raghav/Documents/Uni/oc-nn/models/weights/inp_hidden.npy")
    # w_2 = np.load("/Users/raghav/Documents/Uni/oc-nn/models/weights/hid_out.npy")

    w_1 = init_weights_1((x_size, h_size))
    w_2 = init_weights_2((h_size, y_size))

    # Forward propagation
    yhat = forwardprop(X, w_1, w_2)
    predict = tf.argmax(yhat, axis=1)

    # Backward propagation
    # cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    def g(x): return 1 / (1 + tf.exp(-x))
    cost = ocnn_obj(theta, X, nu, w_1, w_2, g, r)
    updates = tf.train.GradientDescentOptimizer(lr_sigmoid).minimize(cost)


    # Run SGD
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    # train_X = sess.run(feature_inp, feed_dict={Imgs: data_train})
    # test_X = sess.run(feature_inp, feed_dict={Imgs: data_test})

    # train_X = func_get_ImageVectors(dataPathTrain)
    # test_X = func_get_ImageVectors(dataPathTest)

    [train_X, test_X] = func_getKerasModelfeatures()

    # rvalue = np.random.normal(0, 1, (len(train_X), y_size))
    rvalue = 0.1
    for epoch in range(100):
        # Train with each example
        _,loss=sess.run([updates,cost], feed_dict={X: train_X, r: rvalue})
        rvalue = nnScore(train_X, w_1, w_2, g)
        with sess.as_default():
            # cost = cost.eval()
            rvalue = rvalue.eval()
            rvalue = np.percentile(rvalue, q=100 * 0.04)
        print("Epoch = %d, r = %f,loss = %s"
              % (epoch + 1, rvalue,loss))

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
    neg_decisionScore = arrayTest - rstar

    return [pos_decisionScore, neg_decisionScore]
