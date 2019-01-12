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
from keras.layers import Dense, GlobalAveragePooling2D

### Declare the training and test paths
modelpath = "/Users/raghav/Documents/Uni/oc-nn/models/transfer_learning/dogs_not_dogs.model"
train_path = "/Users/raghav/Documents/Uni/oc-nn/data/test_data/"
test_path = "/Users/raghav/Documents/Uni/oc-nn/data/anomalies/"
NB_IV3_LAYERS_TO_FREEZE=7




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

def add_new_last_layer(base_model):
  """Add last layer to the convnet
  Args:
    base_model: keras model excluding top
    nb_classes: # of classes
  Returns:
    new keras model with last layer
  """
  x = base_model.output
  print "base_model.output",x.shape
  inp = base_model.input
  print "base_model.input",inp.shape
  # predictions = Dense(1, activation='linear',name="dense_2")(x) #new sigmoid layer
  predictions = Dense(1, activation='sigmoid', name="dense_2")(x)  # new sigmoid layer
  model = Model(inputs=base_model.input, outputs=predictions)
  return model

# load the trained convolutional neural network freeze all the weights except for last four layers
print("[INFO] loading network...")
model = load_model(modelpath)
model.pop() ## remove the outer layer
model.pop() ## remove the outer activation layer
model = add_new_last_layer(model)
print model.summary()
# print len(model.layers)
print model.layers[10]


# Freeze the weights of untill the last four layers
for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
    layer.trainable = False
for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
    layer.trainable = True

# declare the global variable



print("[INFO] loading images for training...")
data = []
data_test = []
labels = []
labels_test = []
# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(train_path)))
import random
random.seed(42)
random.shuffle(imagePaths)
# loop over the input training images
image_dict = {}
i = 0
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (28, 28))
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
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
y = keras.utils.to_categorical(labels, num_classes=2)

print("[INFO] preparing test data (anomalous )...")
testimagePaths = sorted(list(paths.list_images(test_path)))
# loop over the test images
for imagePath in testimagePaths:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (28, 28))
    image = img_to_array(image)
    data_test.append(image)
    # extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2]
    label = 0 if label == "cats" else 1
    labels_test.append(label)
# scale the raw pixel intensities to the range [0, 1]
data_test = np.array(data_test, dtype="float") / 255.0
labels_test = np.array(labels)



layer_name = 'flatten_1'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
layer1 = model.get_layer("dense_1")

layer2 = model.get_layer("dense_2")
intermediate_output = intermediate_layer_model.predict(data)
X = intermediate_output
intermediate_output = intermediate_layer_model.predict(data_test)
X_test = intermediate_output
print("[INFO] preparing the network to extract flattened inputs(X,X_test) for scoring..... done!!!")


w_1 = layer1.get_weights()
w_2 =layer2.get_weights()


print("[INFO] Initialize the weights ..... done!!!")

opt = SGD(lr=0.01, momentum=0.1)
# recompile the model
# model.compile(loss= customLoss(X,w_1[0],w_2[0], r),optimizer=opt)

# print model.summary()
# Train the model with given epochs
print("[INFO] Recompile the model with customLoss ..... done!!!")
rvalue = 0.1
# (y,output,X,w_1[0],w_2[0],rvalue)
for epoch in range(10):
    # Train with each example
    output = model.get_layer('dense_2').output
    model.compile(loss= customLoss(output,X,w_1[0],w_2[0], rvalue), optimizer=opt)

    model.fit(data,y, epochs=1, batch_size=50)
    w_1 = layer1.get_weights()
    w_2 = layer2.get_weights()
    rvalue = nnScore(X, w_1[0], w_2[0], g)
    rvalue = K.eval(rvalue)
    rvalue = np.percentile(rvalue,q=100*0.04)

    print "[ Iteration:=============] ",epoch+1
    print "[R value computed =======]",rvalue




# get the updated weights after compute the score
w_1 = layer1.get_weights()
w_2 =layer2.get_weights()
rstar = rvalue
print("[INFO] Optimized value of r computed is  ..... done!!!",rstar)
train = nnScore(X, w_1[0], w_2[0], g)
test = nnScore(X_test, w_1[0], w_2[0], g)
pos_decisionScore = train - rstar
neg_decisionScore = test - rstar
pos_decisionScore = K.eval(pos_decisionScore)
neg_decisionScore = K.eval(neg_decisionScore)
# pos_decisionScore = np.amax(pos_decisionScore,axis=1)
# neg_decisionScore = np.amax(neg_decisionScore,axis=1)
print "Pos_decisonScore",pos_decisionScore
print "Neg_decisonScore",neg_decisionScore

# Get the index of the images whose scores are negative and show the images
train_explewithnegativescore = np.where(pos_decisionScore < 0)
print "Training examples with negative scores:============="
print train_explewithnegativescore[0],len(train_explewithnegativescore[0]),len(pos_decisionScore)
print "Training examples with negative scores:============="
# Plot the scores
indexes = train_explewithnegativescore[0].tolist()
for key,value in image_dict.iteritems():
    if key in indexes:
        print value
print "Model Summary",model.summary()
import matplotlib.pyplot as plt
f, axarr = plt.subplots(1, 1, figsize=(5, 5))
st = f.suptitle("One Class NN:  " + "CIFAR-10", fontsize="x-large", fontweight='bold');
plt.hist(pos_decisionScore, bins=25, label='Normal')
plt.hist(neg_decisionScore, bins=25, label='Anomaly')
plt.title("OC-NN :  " + "Sigmoid")
plt.legend()
plt.show()
