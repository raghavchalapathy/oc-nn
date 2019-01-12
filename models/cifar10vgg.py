
from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
import numpy as np
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers
import os
cwd = os.getcwd()
import sys
sys.path.append(cwd+"../data_load/")
sys.path.append(cwd)
sys.path.append(cwd+"../data/")
print(cwd)
def prepare_cifar_data_with_anamolies(original,original_labels,image_and_anamolies):

    imagelabel = image_and_anamolies['image']
    imagecnt = image_and_anamolies['imagecount']

    idx = np.where(original_labels ==imagelabel)

    idx = idx[0][:imagecnt]


    images = original[idx]

    images_labels = original_labels[idx]

    anamoliescnt = image_and_anamolies['anomaliesCount']
    anamolieslabel1 = image_and_anamolies['anomalies1']

    anmolies_idx1 = np.where(original_labels ==anamolieslabel1)
    anmolies_idx1 = anmolies_idx1[0][:(anamoliescnt/2)]
    ana_images1 = original[anmolies_idx1]
    ana_images1_labels = original_labels[anmolies_idx1]

    anamolieslabel2 = image_and_anamolies['anomalies2']

    anmolies_idx2 = np.where(original_labels ==anamolieslabel2)
    anmolies_idx2 = anmolies_idx2[0][:(anamoliescnt/2)]
    ana_images2 = original[anmolies_idx2]
    ana_images2_labels = original_labels[anmolies_idx2]

    temp = np.concatenate((images, ana_images1), axis=0)
    data = np.concatenate((temp, ana_images2), axis=0)

    #labels for these images
    templabel = np.concatenate((images_labels, ana_images1_labels), axis=0)
    datalabels = np.concatenate((templabel, ana_images2_labels), axis=0)


    return [data,datalabels]
class cifar10vgg:
    def __init__(self,train=False):
        self.num_classes = 10
        self.weight_decay = 0.0005
        self.x_shape = [32,32,3]

        self.model = self.build_model()
        if train:
            self.model = self.train(self.model)
        else:
            self.model.load_weights('cifar10vgg.h5')


    def build_model(self):
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

        model = Sequential()
        weight_decay = self.weight_decay

        model.add(Conv2D(64, (3, 3), padding='same',
                         input_shape=self.x_shape,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))
        return model


    def normalize(self,X_train,X_test):
        #this function normalize inputs for zero mean and unit variance
        # it is used when training a model.
        # Input: training set and test set
        # Output: normalized training set and test set according to the trianing set statistics.
        mean = np.mean(X_train,axis=(0,1,2,3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        X_train = (X_train-mean)/(std+1e-7)
        X_test = (X_test-mean)/(std+1e-7)
        return X_train, X_test

    def normalize_production(self,x):
        #this function is used to normalize instances in production according to saved training set statistics
        # Input: X - a training set
        # Output X - a normalized training set according to normalization constants.

        #these values produced during first training and are general for the standard cifar10 training set normalization
        mean = 120.707
        std = 64.15
        return (x-mean)/(std+1e-7)

    def predict(self,x,normalize=True,batch_size=50):
        if normalize:
            x = self.normalize_production(x)
        return self.model.predict(x,batch_size)

    def train(self,model):

        #training parameters
        batch_size = 128
        maxepoches = 250
        learning_rate = 0.1
        lr_decay = 1e-6

        # The data, shuffled and split between train and test sets:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train, x_test = self.normalize(x_train, x_test)

        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        lrf = learning_rate


        #data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)



        #optimization details
        sgd = optimizers.SGD(lr=lrf, decay=lr_decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])


        # training process in a for loop with learning rate drop every 25 epoches.

        for epoch in range(1,maxepoches):

            if epoch%25==0 and epoch>0:
                lrf/=2
                sgd = optimizers.SGD(lr=lrf, decay=lr_decay, momentum=0.9, nesterov=True)
                model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

            historytemp = model.fit_generator(datagen.flow(x_train, y_train,
                                             batch_size=batch_size),
                                steps_per_epoch=x_train.shape[0] // batch_size,
                                epochs=epoch,
                                validation_data=(x_test, y_test),initial_epoch=epoch-1)
        model.save_weights('cifar10vgg.h5')
        return model
image_and_anamolies = {'image': 5,'anomalies1':3,'anomalies2':3,'imagecount': 220,'anomaliesCount':11}
if __name__ == '__main__':


    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    [x_train, y_train] = prepare_cifar_data_with_anamolies(x_train, y_train, image_and_anamolies)
    [x_test, y_test] = prepare_cifar_data_with_anamolies(x_test, y_test, image_and_anamolies)


    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    model = cifar10vgg()
    print(x_test.shape)
    print(type(model))

    print(model.model.summary())
    print(model.model.layers[56].output.get_shape())
    print(model.model.layers[56].output)
    predicted_x = model.predict(x_test[219:231])
    print (np.argmax(predicted_x, 1))
    residuals = np.argmax(predicted_x,1)!=np.argmax(y_test,1)

    # loss = sum(residuals)/len(residuals)
    # print("the validation 0/1 loss is: ",loss)
    # print(len(residuals)+1)



