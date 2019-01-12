from sklearn.preprocessing import StandardScaler
import numpy as np
import json
from keras.datasets import mnist
from keras.utils import to_categorical
##
TRAIN_RANDOM_SEED = 42
TEST_RANDOM_SEED = 41
class CreateDataSet:

    dataPath = '../data/raw/'
    scaler = StandardScaler()

    def get_MNIST_TrainingData(self, numNormal, numAnomalies):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        pos_filter = np.where((y_train == 1))  # One are considered positive
        neg_filter = np.where((y_test == 7))  ## 7 are considered negative
        x_train, y_train = x_train[pos_filter], y_train[pos_filter]
        x_test, y_test = x_test[neg_filter], y_test[neg_filter]

        np.random.seed(TRAIN_RANDOM_SEED)
        np.random.shuffle(x_train)  # shuffle the numpy array
        np.random.shuffle(x_test)
        # Get the normal points
        x_pos = x_train[0:numNormal]
        y_pos = y_train[0:numNormal]
        ## Get the anomalies
        x_neg = x_test[0:numAnomalies]
        y_neg = y_test[0:numAnomalies]
        y_neg = np.zeros(len(y_neg))  ## Convert the label value from 7 to 0 for anomalies
        print(x_pos.shape[0], 'Positive train with shape',x_pos.shape)
        print(x_neg.shape[0], 'Negative train samples')

        # partition the data into training and testing splits using 75% of
        # the data for training and the remaining 25% for testing


        # convert the labels from integers to vectors
        y_pos = to_categorical(y_pos, num_classes=2)
        y_neg = to_categorical(y_neg, num_classes=2)
        return [x_pos, y_pos, x_neg, y_neg]
 

    def get_MNIST_TestingData(self,numNormal,numAnomalies):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
  
        pos_filter = np.where((y_train == 1 )) # One are considered positive
        neg_filter = np.where((y_test == 7)) ## 7 are considered negative
        x_train, y_train = x_train[pos_filter], y_train[pos_filter]
        x_test, y_test = x_test[neg_filter], y_test[neg_filter]

        np.random.seed(TRAIN_RANDOM_SEED)
        np.random.shuffle(x_train) # shuffle the numpy array
        np.random.shuffle(x_test)
        # Get the normal points
        x_pos = x_train[0:numNormal]
        y_pos= y_train[0:numNormal]
        ## Get the anomalies
        x_neg = x_test[0:numAnomalies]
        y_neg= y_test[0:numAnomalies]
        y_neg = np.zeros(len(y_neg)) ## Convert the label value from 7 to 0 for anomalies
        print(x_pos.shape[0], 'Positive test samples with shape',x_pos.shape)
        print(x_neg.shape[0], 'Negative test samples')

        # partition the data into training and testing splits using 75% of
        # the data for training and the remaining 25% for testing


        # convert the labels from integers to vectors
        y_pos = to_categorical(y_pos, num_classes=2)
        y_neg = to_categorical(y_neg, num_classes=2)
        return [x_pos,y_pos,x_neg,y_neg]

    def load_LHCDataset(self,data_dir_path):

        background = np.array([
            json.loads(s)
            for s in open(data_dir_path + '/bgimages_njge3_100k.dat')
        ])
        signal = np.array([
            json.loads(s)
            for s in open(data_dir_path + '/sigimages_njge3_100k.dat')
        ])

        return [background, signal]

    def get_LHC_TrainingData(self, normal,anomalies, NUM_OF_NORMAL,NUM_OF_ANOMALIES):


        normal = normal.astype('float32')
        anomalies = anomalies.astype('float32')
        # normal /= 255
        # anomalies /= 255

        np.random.seed(TRAIN_RANDOM_SEED)
        np.random.shuffle(normal)  # shuffle the numpy array
        np.random.shuffle(anomalies)
        # Get the normal points
        x_pos = normal[0:NUM_OF_NORMAL]
        y_pos = np.ones(len(x_pos))
        ## Get the anomalies
        x_neg = anomalies[0:NUM_OF_ANOMALIES]
        y_neg = np.zeros(len(x_neg))

        print(x_pos.shape[0], 'Positive train samples with shape', x_pos.shape)
        print(x_neg.shape[0], 'Negative train samples')

        # partition the data into training and testing splits using 75% of
        # the data for training and the remaining 25% for testing
        # convert the labels from integers to vectors
        y_pos = to_categorical(y_pos, num_classes=2)
        y_neg = to_categorical(y_neg, num_classes=2)
        return [x_pos, y_pos, x_neg, y_neg]


    def get_LHC_TestingData(self, normal, anomalies, NUM_OF_NORMAL,NUM_OF_ANOMALIES):
        normal = normal.astype('float32')
        anomalies = anomalies.astype('float32')
        # normal /= 255
        # anomalies /= 255

        np.random.seed(TEST_RANDOM_SEED)
        np.random.shuffle(normal)  # shuffle the numpy array
        np.random.shuffle(anomalies)
        # Get the normal points
        x_pos = normal[0:NUM_OF_NORMAL]
        y_pos = np.ones(len(x_pos))
        ## Get the anomalies
        x_neg = anomalies[0:NUM_OF_ANOMALIES]
        y_neg = np.zeros(len(x_neg))

        print(x_pos.shape[0], 'Positive test samples with shape', x_pos.shape)
        print(x_neg.shape[0], 'Negative test samples')

        # partition the data into training and testing splits using 75% of
        # the data for training and the remaining 25% for testing


        # convert the labels from integers to vectors
        y_pos = to_categorical(y_pos, num_classes=2)
        y_neg = to_categorical(y_neg, num_classes=2)
        return [x_pos, y_pos, x_neg, y_neg]

    def get_USPS_TestingData(self,NUM_OF_NORMAL,NUM_OF_ANOMALIES):


        import pickle

        with open(self.dataPath + 'usps_data.pkl', 'rb') as fp:
            loaded_data1 = pickle.load(fp, encoding='latin1')

        labels = loaded_data1['target']
        data = loaded_data1['data']



        ## Select Ones and sevens
        k_ones = np.where(labels == 2)
        data_ones = data[k_ones]
        k_sevens = np.where(labels == 8)
        data_sevens = data[k_sevens]

        data_ones = data_ones[0:NUM_OF_NORMAL]
        data_sevens = data_sevens[0:NUM_OF_ANOMALIES]



        data_ones = data_ones.astype('float32')
        data_sevens = data_sevens.astype('float32')
        data_ones /= 255
        data_sevens /= 255

        ## Reshape the USPS data to a 2D array func before fit flattens it anyways for sake of code resuse
        # do the reshape here
        data_ones = np.reshape(data_ones, (len(data_ones), 16, 16))
        data_sevens = np.reshape(data_sevens, (len(data_sevens), 16, 16))

        np.random.seed(TEST_RANDOM_SEED)
        np.random.shuffle(data_ones)  # shuffle the numpy array
        np.random.shuffle(data_sevens)

        print(data_ones.shape[0], 'Positive test samples with shape', data_ones.shape)
        print(data_sevens.shape[0], 'Negative test samples')

        label_ones = 1 * np.ones(len(data_ones))
        label_sevens = np.zeros(len(data_sevens))
        # convert the labels from integers to vectors
        label_ones = to_categorical(label_ones, num_classes=2)
        label_sevens = to_categorical(label_sevens, num_classes=2)

        return [data_ones, label_ones, data_sevens, label_sevens]


    def get_USPS_TrainingData(self,NUM_OF_NORMAL,NUM_OF_ANOMALIES):
        import pickle

        with open(self.dataPath + 'usps_data.pkl', 'rb') as fp:
            loaded_data1 = pickle.load(fp, encoding='latin1')

        labels = loaded_data1['target']
        data = loaded_data1['data']

        ## Select Ones and sevens
        k_ones = np.where(labels == 2)
        data_ones = data[k_ones]
        k_sevens = np.where(labels == 8)
        data_sevens = data[k_sevens]

        data_ones = data_ones[0:NUM_OF_NORMAL]
        data_sevens = data_sevens[0:NUM_OF_ANOMALIES]

        data_ones = data_ones.astype('float32')
        data_sevens = data_sevens.astype('float32')
        data_ones /= 255
        data_sevens /= 255

        ## Reshape the USPS data to a 2D array func before fit flattens it anyways for sake of code resuse
        # do the reshape here
        data_ones = np.reshape(data_ones, (len(data_ones), 16, 16))
        data_sevens = np.reshape(data_sevens, (len(data_sevens), 16, 16))

        np.random.seed(TRAIN_RANDOM_SEED)
        np.random.shuffle(data_ones)  # shuffle the numpy array
        np.random.shuffle(data_sevens)

        print(data_ones.shape[0], 'Positive train samples with shape', data_ones.shape)
        print(data_sevens.shape[0], 'Negative train samples')

        label_ones = 1 * np.ones(len(data_ones))
        label_sevens = np.zeros(len(data_sevens))
        # convert the labels from integers to vectors
        label_ones = to_categorical(label_ones, num_classes=2)
        label_sevens = to_categorical(label_sevens, num_classes=2)

        return [data_ones, label_ones, data_sevens, label_sevens]



    def get_FAKE_Noise_USPS_TrainingData(self, X):

        data_noise = np.random.uniform(0, 1, (len(X), 256))
        data_noise = np.reshape(data_noise, (len(data_noise), 16, 16))
        label_noise = np.zeros(len(data_noise))
        label_noise = to_categorical(label_noise, num_classes=2)

        return [data_noise, label_noise]
    
    def get_FAKE_Noise_LHC_TrainingData(self, X):

        data_noise = np.random.uniform(0, 1, (len(X), 1369))
        data_noise = np.reshape(data_noise, (len(data_noise), 37, 37))
        label_noise = np.zeros(len(data_noise))
        label_noise = to_categorical(label_noise, num_classes=2)

        return [data_noise, label_noise]

    def get_FAKE_Noise_MNIST_TrainingData(self, X):

        data_noise = np.random.uniform(0, 1, (len(X), 784))
        data_noise = np.reshape(data_noise,(len(data_noise),28,28))
        label_noise = np.zeros(len(data_noise))
        label_noise = to_categorical(label_noise, num_classes=2)
        return [data_noise, label_noise]
