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
from keras.layers import Dense, GlobalAveragePooling2D,Activation
from keras.applications.vgg16 import preprocess_input


### Declare the training and test paths
modelpath = "/Users/raghav/Documents/Uni/oc-nn/models/transfer_learning/autoencoder1.model"
train_path = "/Users/raghav/Documents/Uni/oc-nn/data/test_data/"
test_path = "/Users/raghav/Documents/Uni/oc-nn/data/anomalies/"
NB_IV3_LAYERS_TO_FREEZE=23
side = 224
side = 224
channel = 3



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

# load the trained convolutional neural network freeze all the weights except for last four layers
print("[INFO] loading network...")
model = load_model(modelpath)
model = add_new_last_layer(model.output,model.input)
print(model.summary())
print "Length of model layers...", len(model.layers)

# Freeze the weights of untill the last four layers
for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
    layer.trainable = False
for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
    layer.trainable = True



print("[INFO] loading images for training...")
data = []
data_test = []
labels = []
labels_test = []
# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(train_path)))
# loop over the input training images
image_dict = {}
test_image_dict ={}
i = 0
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (224, 224))
    image = img_to_array(image)
    data.append(image)
    image_dict.update({i:imagePath})
    i = i + 1

    # extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2]
    label = 1 if label == "dogs" else 0
    labels.append(label)
# scale the raw pixel intensities to the range [0, 1]
data = np.array(data)
y = keras.utils.to_categorical(labels, num_classes=2)
# print image_dict

print("[INFO] preparing test data (anomalous )...")
testimagePaths = sorted(list(paths.list_images(test_path)))
# loop over the test images
j = 0
for imagePath in testimagePaths:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (224, 224))
    image = img_to_array(image)
    data_test.append(image)
    test_image_dict.update({j: imagePath})
    j = j + 1
    # extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2]
    label = 0 if label == "cats" else 1
    labels_test.append(label)
# scale the raw pixel intensities to the range [0, 1]
data_test = np.array(data_test)
# print test_image_dict


# Preprocess the inputs
x_train = data
x_test = data_test



## Obtain the intermediate output
layer_name = 'flatten_1'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)

layer1 = model.get_layer("dense_output1")
layer2 = model.get_layer("dense_output2")
intermediate_output = intermediate_layer_model.predict(x_train)
X = intermediate_output
intermediate_output = intermediate_layer_model.predict(x_test)
X_test = intermediate_output

print X.shape
print X_test.shape


# img_dimension = 1568
# X = data
# X_test = data_test

# X = np.reshape(X,(len(X),img_dimension))
# X_test= np.reshape(X_test,(len(X_test),img_dimension))
#
# print X.shape
# print X_test.shape
#

print("[INFO] preparing the network to extract flattened inputs(X,X_test) for scoring..... done!!!")
import matplotlib.pyplot as plt
from sklearn import svm
nu = 0.04
ocSVM = svm.OneClassSVM(nu=nu, kernel='rbf')
# ocSVM = svm.OneClassSVM(nu=nu, kernel='linear')
ocSVM.fit(X)
activations = ["linear","rbf"]
pos_decisionScore = ocSVM.decision_function(X)
neg_decisionScore = ocSVM.decision_function(X_test)
plt.hist(pos_decisionScore, bins = 25, label = 'Normal')
plt.hist(neg_decisionScore, bins = 25, label = 'Anomaly')
plt.title("ocsvm:  " + activations[1]);
plt.legend(loc='upper right')
plt.show()
