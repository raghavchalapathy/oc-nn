import keras.backend as K
from tensorflow.python import debug as tf_debug
import tensorflow as tf
import keras

PROJECT_DIR = "/Users/raghav/envPython3/experiments/one_class_neural_networks/"
import sys,os
import numpy as np
sys.path.append(PROJECT_DIR)
from src.data.make_dataset import CreateDataSet
from src.models.FF_NN import FF_NN
## Create data for training and testing
createData = CreateDataSet()
NUM_NORMAL = 5000
TRAIN_NUM_ANOMALIES = 1000
TEST_NUM_ANOMALIES = 50
IMG_HGT =28
IMG_WDT=28
IMG_DEPTH=1
nClass=2
# trainX,trainY = createData.get_MNIST_TrainingData(NUM_NORMAL)
trainX,trainY,train_Anomaly_X,train_Anomaly_Y = createData.get_MNIST_TrainingData(NUM_NORMAL,TRAIN_NUM_ANOMALIES)
[test_ones,label_ones,test_sevens,label_sevens]= createData.get_MNIST_TestingData(NUM_NORMAL,TEST_NUM_ANOMALIES)

from src.models.OC_NN import OC_NN
ocnn = OC_NN()

nu= 0.01
NUM_EPOCHS = 100
# keras.backend.set_session(
#     tf_debug.TensorBoardDebugWrapperSession(tf.Session(), "vlan-2663-10-17-5-224.staff.wireless.sydney.edu.au:7000"))

keras.backend.set_session(
    tf_debug.TensorBoardDebugWrapperSession(tf.Session(), "localhost:7000"))

ocnn.fit(trainX,nu,NUM_EPOCHS,IMG_HGT,IMG_WDT,IMG_DEPTH,nClass)
res = ocnn.score(test_ones,test_sevens)
auc_OCNN = res
print("="*35)
print("AUC:",res)
print("="*35)
