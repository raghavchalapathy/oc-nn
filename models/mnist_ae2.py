import numpy as np
import keras, keras.layers as L
from keras.datasets import mnist
import cv2
import os
from visualize_reconstruction import visualize
import glob

orig_dimension = 784
code_size =32
img_rows, img_cols, img_chns = 32, 32, 3
intermediate_dim = code_size
filters = 32
num_conv = 3
num_layer = 2
epochs = 1000

dataPath = "/Users/raghav/Documents/Uni/oc-nn/data/"
savedModel = "/Users/raghav/Documents/Uni/oc-nn/models/transfer_learning/cae_autoencoders/trained_models/"

def prepare_mnist_mlfetch():
    (x_train, x_trainLabels), (x_test, x_testLabels) = mnist.load_data()
    labels = x_trainLabels
    data = x_train
    # print "******",labels

    ## 4859 digit- 4
    k_four = np.where(labels == 4)
    label_four  = labels[k_four]
    data_four = data[k_four]

    k_zeros = np.where(labels == 0)
    k_sevens = np.where(labels == 7)
    k_nine = np.where(labels == 9)

    ## 265 (0,7,9)

    label_zeros = labels[k_zeros]
    data_zeros = data[k_zeros]

    label_sevens = labels[k_sevens]
    data_sevens = data[k_sevens]

    label_nine = labels[k_nine]
    data_nines = data[k_nine]


    #
    # print "data_sevens:",data_sevens.shape
    # print "label_sevens:",label_sevens.shape
    # print "data_ones:",data_ones.shape
    # print "label_ones:",label_ones.shape
    #
    data_four = data_four[:220]

    data_zeros = data_zeros[:5]
    data_sevens = data_sevens[:3]
    data_nines = data_nines[:3]

    data_sevens = data_sevens[:11]

    normal = data_four
    anomalies = np.concatenate((data_zeros, data_sevens,data_nines), axis=0)




    return [normal,anomalies]


## Import all the datasets and prepare the Training and test set for respective datasets
[X_train,X_test] = prepare_mnist_mlfetch()

### Prepare training and test set for respective datasets
# USPS Train and Test Data


def build_deep_autoencoder(img_shape=(28, 28, 1)):


    # encoder
    encoder = keras.models.Sequential()
    encoder.add(L.InputLayer(img_shape))
    encoder.add(L.Flatten())
    encoder.add(L.Dense(128))
    encoder.add(L.Dense(code_size))

    # encoder.add(L.Dense(code_size, kernel_regularizer=keras.regularizer.l2(0.01))


    # decoder
    decoder = keras.models.Sequential()
    decoder.add(L.InputLayer((code_size,)))
    decoder.add(L.Dense(128))
    decoder.add(L.Dense(orig_dimension))
    decoder.add(L.Reshape((28, 28, 1)))

    return encoder, decoder

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

print "X_train.shape", X_train.shape
print "X_test.shape",X_test.shape
X_train = np.reshape(X_train,(len(X_train),28,28,1))
X_test = np.reshape(X_test,(len(X_test),28,28,1))

img_shape = X_train.shape[1:]

print "Image Shape",img_shape

encoder, decoder = build_deep_autoencoder((28, 28, 1))
print "Encoder Summary: ====",encoder.summary()

inp = L.Input(img_shape)
code = encoder(inp)
reconstruction = decoder(code)


autoencoder = keras.models.Model(inp, reconstruction)
print autoencoder.summary()


autoencoder.compile('adamax', 'mse')


print("[INFO:]Training: Autoencoder")

autoencoder.fit(x=X_train, y=X_train, epochs=epochs)


# Evaluation using  input
mse = autoencoder.evaluate((X_train),X_train,verbose=0)
print("Final MSE on Training:", mse)
mse = autoencoder.evaluate((X_test),X_test,verbose=0)
print("Final MSE on Testing:", mse)

# save all 3 models for future use - especially generator
autoencoder.save(savedModel+'/mnist_AE2_%d_id_%d_e_%d_ae.model' % ( num_layer, intermediate_dim, epochs))
encoder.save(savedModel+'/mnist_AE2_%d_id_%d_e_%d_encoder.model' % (num_layer, intermediate_dim, epochs))


# for i in range(1):
#     img = X_train[i]
#     visualize(img,encoder,decoder)

