# import the necessary packages
import numpy as np
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
# set the matplotlib backend so figures can be saved in the background

 
# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam,Adagrad
import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np
import cv2 
import tensorflow as tf
from keras.utils.generic_utils import get_custom_objects

class Fake_Noise_FF_NN:

    ## Initialise static variables
    INPUT_DIM = 0
    HIDDEN_SIZE = 0

    def __init__(self, inputdim,hiddenLayerSize,img_hgt,img_wdt,modelSavePath,reportSavePath, intValue=0, stringParam="defaultValue",
                 otherParam=None):
        """
        Called when initializing the classifier
        """
        Fake_Noise_FF_NN.INPUT_DIM = inputdim
        self.intValue = intValue
        self.stringParam = stringParam

        # THIS IS WRONG! Parameters should have same name as attributes
        self.differentParam = otherParam

        self.directory = modelSavePath
        self.results = reportSavePath
        self.model = ""
        self.IMG_HGT = img_hgt
        self.IMG_WDT = img_wdt
        self.h_size= 196
        Fake_Noise_FF_NN.HIDDEN_SIZE = hiddenLayerSize

    @staticmethod
    def image_to_feature_vector(image, IMG_HGT,IMG_WDT):
        # resize the image to a fixed size, then flatten the image into
        # a list of raw pixel intensities
        return np.reshape(image,(len(image),IMG_HGT*IMG_WDT))



    @staticmethod
    def build(width, height, depth, classes):

        h_size = Fake_Noise_FF_NN.HIDDEN_SIZE

        def custom_activation(x):
            return (1 / np.sqrt(h_size)) * tf.cos(x / 0.02)

        get_custom_objects().update({
            'custom_activation':
                Activation(custom_activation)
        })

        model = Sequential()
        model.add(Dense(h_size, input_dim=Fake_Noise_FF_NN.INPUT_DIM, kernel_initializer="glorot_normal"))
        model.add(Activation(custom_activation))
        model.add(Dense(classes))
        model.add(Activation("linear"))

        # return the constructed network architecture
        return model
    
    def fit(self,trainX,trainY,nEpochs,IMG_HGT,IMG_WDT,IMG_DEPTH,nClass):
        # initialize the model
        EPOCHS = nEpochs
        INIT_LR = 1e-1
        BS = 100
        print("[INFO] compiling model...")
        trainX= Fake_Noise_FF_NN.image_to_feature_vector(trainX, IMG_HGT, IMG_WDT)

        model = Fake_Noise_FF_NN.build(width=IMG_HGT, height=IMG_WDT, depth=IMG_DEPTH, classes=nClass)
        opt = Adagrad(lr=INIT_LR, decay=INIT_LR / EPOCHS)
        model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])
        # train the network
        print("[INFO] training network...")


        H = model.fit(trainX, trainY, batch_size=BS,epochs=EPOCHS,validation_split=0.1, verbose=0)
        # save the model to disk
        print("[INFO] serializing network...")
        model.save(self.directory+"FakeNoise_FF_NN.h5")
        # plot the training loss and accuracy
        plt.style.use("ggplot")
        plt.figure()
        N = EPOCHS
        plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
        plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
        plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
        plt.title("Fake_Noise_FF_NN Training Loss and Accuracy on 1's / 7's")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="upper right")
        plt.savefig(self.results+"trainValLoss.png")

    def score(self,testPosX,testPosY ,testNegX,testNegY):
        # load the trained convolutional neural network
        print("[INFO] loading network...")

        model = load_model(self.directory+"FakeNoise_FF_NN.h5")

        x_test =  np.concatenate((testPosX, testNegX), axis=0)
        y_test = np.concatenate((testPosY, testNegY), axis=0)
        IMG_HGT = self.IMG_HGT
        IMG_WDT=self.IMG_WDT
        x_test = Fake_Noise_FF_NN.image_to_feature_vector(x_test, IMG_HGT, IMG_WDT)


        print(y_test.shape[0], 'Actual test samples')



        from sklearn.metrics import roc_curve,accuracy_score,roc_auc_score
        y_pred_keras = model.predict_proba(x_test)
        print(y_pred_keras.shape[0], 'Predicted test samples')

        y_pred = np.argmax(y_pred_keras, axis=1)
        y_true = np.argmax(y_test, axis=1)

        # print "y_pred.shape",y_pred.shape
        accuracy = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        ####
        print ("=" * 35)
        print ("auccary_score:", accuracy)
        print ("roc_auc_score:", auc)
        start = len(x_test) - 100  # Print the last 100 labels among which last 50 are known anomalies
        end = len(x_test)
        print("y_true", y_true[start:end])
        print("y_pred", y_pred[start:end])
        print ("=" * 35)

        return auc


