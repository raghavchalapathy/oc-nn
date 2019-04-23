# Keras-Tensorflow Implementation of One Class Neural Networks.


This repository provides a Keras-Tensorflow implementation of the One Class Neural Network method presented in our paper ”Anomaly Detection using One Class Neural Networks”.

# Citations and Contact.

You find a PDF of the **One Class Neural Network paper** at:
https://arxiv.org/pdf/1802.06360.pdf
If you use our work, please also cite the paper:

```
@article{chalapathy2018anomaly,
  title={Anomaly Detection using One-Class Neural Networks},
  author={Chalapathy, Raghavendra and Menon, Aditya Krishna and Chawla, Sanjay},
  journal={arXiv preprint arXiv:1802.06360 },
  year={2018}
}
```


You find a PDF of the **Robust, Deep and Inductive Anomaly Detection** paper at:
https://arxiv.org/pdf/1704.06743.pdf

If you use our work, please also cite the paper:

```
 @inproceedings{chalapathy2017robust,
 title={Robust, deep and inductive anomaly detection},
 author={Chalapathy, Raghavendra and Menon, Aditya Krishna and Chawla, Sanjay},
 booktitle={Joint European Conference on Machine Learning and Knowledge Discovery in Databases},
 pages={36--51},
 year={2017},
 organization={Springer}
}
```

If you would like to get in touch, please contact .
raghav.chalapathy@gmail.com



# Abstract

>We propose a one-class neural network (OC-NN) model to detect anomalies in complex data sets. OC-NN combines the ability of deep networks to extract a progressively rich representation of data with the one-class objective of creating a tight envelope around normal data. The OC-NN approach breaks new ground for the following crucial reason: data representation in the hidden layer is driven by the OC-NN objective and is thus customized for anomaly detection. This is a departure from other approaches which use a hybrid approach of learning deep features using an autoencoder and then feeding the features into a separate anomaly detection method like one-class SVM (OC-SVM). The hybrid OC-SVM approach is sub-optimal because it is unable to influence representational learning in the hidden layers. A comprehensive set of experiments demonstrate that on complex data sets (like CIFAR and GTSRB), OC-NN performs on par with state-of-the-art methods and outperformed conventional shallow methods in some scenarios.



# Installation

This code is written in Python 3.6.7 and runs on Google Colab environment

Clone the repository to your local machine and directory of choice and upload to your Google drive account and open the notebooks present in directory "notebooks" in order to execute reproduce the results.



PS: 

Kindly make sure the PROJECT_DIR = "/content/drive/My Drive/2019/testing/oc-nn/" is set appropriately inside the notebook and in files src/models/RCAE.py, src/data/cifar10.py , src/data/mnist.py

Kindly make sure you configure the normal and anomalous class inside the src/config.py present within the Configuration class object  for all the data sets and for any new data sets which would be added in future.

The data related to GTSRB experiments can be downloaded [here](https://drive.google.com/open?id=1GFyzwQZk6ie4aPfpVzlRi4VICyFNGs7Y). Please note that this data was shared to me by [Lukas Ruff](https://github.com/lukasruff) kindly request to acknowledge by citing his work as well.


# Working with your own Data

Kindly customize the load_data function inside the CIFAR_10_DataLoader or MNIST_DataLoader class apppropriately to suite load experiment data. Also kindly RCAE.py and OneClass_SVDD.py to suite your training and test data needs.



# Running experiments

We currently have implemented the MNIST (http://yann.lecun.com/exdb/mnist/) and CIFAR-10 (https://www.cs.toronto.edu/~kriz/cifar.html) datasets and simple LeNet-type networks.



## MNIST Example
### RCAE

```python
%reload_ext autoreload
%autoreload 2
from src.models.RCAE import RCAE_AD
import numpy as np 
from src.config import Configuration as Cfg

DATASET = "mnist"
IMG_DIM= 784
IMG_HGT =28
IMG_WDT=28
IMG_CHANNEL=1
HIDDEN_LAYER_SIZE= 32
MODEL_SAVE_PATH = PROJECT_DIR + "/models/MNIST/RCAE/"
REPORT_SAVE_PATH = PROJECT_DIR + "/reports/figures/MNIST/RCAE/"
PRETRAINED_WT_PATH = ""
# RANDOM_SEED = [42,56,81,67,33,25,90,77,15,11]
RANDOM_SEED = [42]
AUC = []

for seed in RANDOM_SEED:  
  Cfg.seed = seed
  rcae = RCAE_AD(DATASET,IMG_DIM, HIDDEN_LAYER_SIZE, IMG_HGT, IMG_WDT,IMG_CHANNEL, MODEL_SAVE_PATH, REPORT_SAVE_PATH,PRETRAINED_WT_PATH,seed)
  print("Train Data Shape: ",rcae.data._X_train.shape)
  print("Train Label Shape: ",rcae.data._y_train.shape)
  print("Validation Data Shape: ",rcae.data._X_val.shape)
  print("Validation Label Shape: ",rcae.data._y_val.shape)
  print("Test Data Shape: ",rcae.data._X_test.shape)
  print("Test Label Shape: ",rcae.data._y_test.shape)
  print("===========TRAINING AND PREDICTING WITH DCAE============================")
  auc_roc = rcae.fit_and_predict()
  print("========================================================================")
  AUC.append(auc_roc)
  
print("===========TRAINING AND PREDICTING WITH DCAE============================")
print("AUROC computed ", AUC)
auc_roc_mean = np.mean(np.asarray(AUC))
auc_roc_std = np.std(np.asarray(AUC))
print ("AUROC =====", auc_roc_mean ,"+/-",auc_roc_std)
print("========================================================================")
```

### OC_NN or Deep SVDD approach

```python
%reload_ext autoreload
%autoreload 2

from src.models.OneClass_SVDD import OneClass_SVDD
from src.models.config import Configuration as Cfg

DATASET = "mnist"
IMG_DIM= 784
IMG_HGT =28
IMG_WDT=28
IMG_CHANNEL=1
HIDDEN_LAYER_SIZE= 32
MODEL_SAVE_PATH = PROJECT_DIR + "/models/MNIST/OC_NN/"
REPORT_SAVE_PATH = PROJECT_DIR + "/reports/figures/MNIST/OC_NN/"
PRETRAINED_WT_PATH = ""
#LOSS_FUNCTION = "SOFT_BOUND_DEEP_SVDD"
# LOSS_FUNCTION = "ONE_CLASS_DEEP_SVDD"
LOSS_FUNCTION = "ONE_CLASS_NEURAL_NETWORK"

import os
os.chdir(PROJECT_DIR)
# RANDOM_SEED = [42,56,81,67,33,25,90,77,15,11]
RANDOM_SEED = [42]
AUC = []

for seed in RANDOM_SEED:  
  Cfg.seed = seed
  ocnn = OneClass_SVDD(DATASET,LOSS_FUNCTION,IMG_DIM, HIDDEN_LAYER_SIZE, IMG_HGT, IMG_WDT,IMG_CHANNEL, MODEL_SAVE_PATH, REPORT_SAVE_PATH,PRETRAINED_WT_PATH,seed)
 
  print("[INFO:] Testing with ALL other  DIGITs  as anomalies")
  ocnn.fit()
  print("==============PREDICTING THE LABELS ==============================")
  auc_score = ocnn.predict()
  AUC.append(auc_score)

print("===========TRAINING AND PREDICTING WITH OCSVDD============================")
print("AUROC computed ", AUC)
auc_roc_mean = np.mean(np.asarray(AUC))
auc_roc_std = np.std(np.asarray(AUC))
print ("AUROC =====", auc_roc_mean ,"+/-",auc_roc_std)
print("========================================================================")
```




## CIFAR-10 Example
### RCAE

```python
%reload_ext autoreload
%autoreload 2
from src.models.RCAE import RCAE_AD
import numpy as np 
from src.config import Configuration as Cfg

DATASET = "cifar10"
IMG_DIM= 3072
IMG_HGT =32
IMG_WDT=32
IMG_CHANNEL=3
HIDDEN_LAYER_SIZE= 128
MODEL_SAVE_PATH = PROJECT_DIR + "/models/cifar10/RCAE/"
REPORT_SAVE_PATH = PROJECT_DIR + "/reports/figures/cifar10/RCAE/"

PRETRAINED_WT_PATH = ""
# RANDOM_SEED = [42,56,81,67,33,25,90,77,15,11]
RANDOM_SEED = [42]
AUC = []

for seed in RANDOM_SEED:  
  Cfg.seed = seed
  rcae = RCAE_AD(DATASET,IMG_DIM, HIDDEN_LAYER_SIZE, IMG_HGT, IMG_WDT,IMG_CHANNEL, MODEL_SAVE_PATH, REPORT_SAVE_PATH,PRETRAINED_WT_PATH,seed)
  print("Train Data Shape: ",rcae.data._X_train.shape)
  print("Train Label Shape: ",rcae.data._y_train.shape)
  print("Validation Data Shape: ",rcae.data._X_val.shape)
  print("Validation Label Shape: ",rcae.data._y_val.shape)
  print("Test Data Shape: ",rcae.data._X_test.shape)
  print("Test Label Shape: ",rcae.data._y_test.shape)
  print("===========TRAINING AND PREDICTING WITH DCAE============================")
  auc_roc = rcae.fit_and_predict()
  print("========================================================================")
  AUC.append(auc_roc)
  
print("===========TRAINING AND PREDICTING WITH DCAE============================")
print("AUROC computed ", AUC)
auc_roc_mean = np.mean(np.asarray(AUC))
auc_roc_std = np.std(np.asarray(AUC))
print ("AUROC =====", auc_roc_mean ,"+/-",auc_roc_std)
print("========================================================================")

```



### OC_NN or Deep SVDD approach


```python

%reload_ext autoreload
%autoreload 2

from src.models.OneClass_SVDD import OneClass_SVDD
from src.models.config import Configuration as Cfg

DATASET = "cifar10"
IMG_DIM= 3072
IMG_HGT =32
IMG_WDT=32
IMG_CHANNEL=3
HIDDEN_LAYER_SIZE= 128
MODEL_SAVE_PATH = PROJECT_DIR + "/models/cifar10/OC_NN/"
REPORT_SAVE_PATH = PROJECT_DIR + "/reports/figures/cifar10/OC_NN/DeepSVDD/"
PRETRAINED_WT_PATH = ""
#LOSS_FUNCTION = "SOFT_BOUND_DEEP_SVDD"
# LOSS_FUNCTION = "ONE_CLASS_DEEP_SVDD"
LOSS_FUNCTION = "ONE_CLASS_NEURAL_NETWORK"

import os
os.chdir(PROJECT_DIR)
# RANDOM_SEED = [42,56,81,67,33,25,90,77,15,11]
RANDOM_SEED = [42]
AUC = []

for seed in RANDOM_SEED:  
  Cfg.seed = seed
  ocnn = OneClass_SVDD(DATASET,LOSS_FUNCTION,IMG_DIM, HIDDEN_LAYER_SIZE, IMG_HGT, IMG_WDT,IMG_CHANNEL, MODEL_SAVE_PATH, REPORT_SAVE_PATH,PRETRAINED_WT_PATH,seed)
 
  print("[INFO:] Testing with ALL other  Image Classes  as anomalies")
  ocnn.fit()
  print("==============PREDICTING THE LABELS ==============================")
  auc_score = ocnn.predict()
  AUC.append(auc_score)

print("===========TRAINING AND PREDICTING WITH OCSVDD============================")
print("AUROC computed ", AUC)
auc_roc_mean = np.mean(np.asarray(AUC))
auc_roc_std = np.std(np.asarray(AUC))
print ("AUROC =====", auc_roc_mean ,"+/-",auc_roc_std)
print("========================================================================")

```



## Sample Results 


### MNIST
Example of the  most normal (left) and  most anomalous (right) test set examples per class on MNIST according to One Class Neural Networks and Robust Convolution Autoencoder (RCAE) anomaly scores.



![img](/results/mnist_rcae.png)

![img](/results/mnist_ocnn.png)


### CIFAR-10
Example of the  most normal (left) and  most anomalous (right) test set examples per class on CIFAR-10 according to One Class Neural Networks and Robust Convolution Autoencoder (RCAE) anomaly scores.


![img](/results/cifar10.png)



# License
MIT


# Disclosure
This implementation is based on the repository https://github.com/lukasruff/Deep-SVDD, which is licensed under the MIT license. The Deep SVDD repository is an implementation of the paper "Deep One-Class Classification by Ruff, Lukas and Vandermeulen, Robert A. and G{\"o}rnitz, Nico and Deecke, Lucas and Siddiqui, Shoaib A. and Binder, Alexander and M{\"u}ller, Emmanuel and Kloft, Marius


