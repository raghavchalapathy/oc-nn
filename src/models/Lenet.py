# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
 
# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import matplotlib.pyplot as plt

from keras.models import load_model
import numpy as np


class LeNet:



    def __init__(self, intValue=0, stringParam="defaultValue",
                 otherParam=None):
        """
        Called when initializing the classifier
        """
        self.intValue = intValue
        self.stringParam = stringParam

        # THIS IS WRONG! Parameters should have same name as attributes
        self.differentParam = otherParam

        self.directory = "../models/LeNet/"
        self.results = "../reports/figures/LeNet/"
        self.model = ""

    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            # first set of CONV => RELU => POOL layers
        model.add(Conv2D(20, (5, 5), padding="same",
                         input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # second set of CONV => RELU => POOL layers
        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model
    
    def fit(self,trainX,trainY,testX,testY,nEpochs,IMG_HGT,IMG_WDT,IMG_DEPTH,nClass):
        # initialize the model
        EPOCHS = nEpochs
        INIT_LR = 1e-3
        BS = 32
        print("[INFO] compiling model...")
        model = LeNet.build(width=IMG_HGT, height=IMG_WDT, depth=IMG_DEPTH, classes=nClass)
        opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
        model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])
        # train the network
        print("[INFO] training network...")
        aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                                 height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                                 horizontal_flip=True, fill_mode="nearest")

        H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
            validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
            epochs=EPOCHS, verbose=1)
        # save the model to disk
        print("[INFO] serializing network...")
        model.save(self.directory+"LeNet.h5")
        # plot the training loss and accuracy
        plt.style.use("ggplot")
        plt.figure()
        N = EPOCHS
        plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
        plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
        plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
        plt.title("LeNet Training Loss and Accuracy on 1's / 7's")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="upper right")
        plt.savefig(self.results+"trainValLoss.png")

    def score(self,testPosX,testPosY ,testNegX,testNegY):
        # load the trained convolutional neural network
        print("[INFO] loading network...")

        model = load_model(self.directory+"LeNet.h5")

        x_test =  np.concatenate((testPosX, testNegX), axis=0)
        y_test = np.concatenate((testPosY, testNegY), axis=0)

        print(y_test.shape[0], 'Actual test samples')



        from sklearn.metrics import roc_curve,accuracy_score,roc_auc_score
        y_pred_keras = model.predict_proba(x_test)


        y_pred = np.argmax(y_pred_keras, axis=1)
        y_true = np.argmax(y_test, axis=1)

        # print "y_pred.shape",y_pred.shape
        accuracy = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        ####
        print ("=" * 35)
        print ("auccary_score:", accuracy)
        print ("roc_auc_score:", auc)
        print("y_true",y_true[4950:5050])
        print("y_pred", y_pred[4950:5050])
        print ("=" * 35)

        return auc



