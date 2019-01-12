import numpy as np
import pandas as pd
from sklearn import utils
# import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

dataPath = './data/'
activations = ["Linear","Sigmoid"]
methods = ["Linear","RBF"]





def plot_decision_scores_CIFAR_10(dataset,df_cifar_scores):
    # Four axes, returned as a 2-d array
    #
    # import matplotlib as mpl
    # label_size = 40
    # titleLabel = 22
    #
    # mpl.rcParams['xtick.labelsize'] = label_size
    # mpl.rcParams['ytick.labelsize'] = label_size
    # mpl.rcParams['font.weight'] ='bold'
    # mpl.rcParams['legend.fontsize']= 30
    # mpl.rcParams['axes.titlesize'] = 40
    # mpl.rcParams['figure.titlesize'] = 40
    # mpl.rcParams['figure.titleweight'] = 'bold'
    # mpl.rcParams['axes.titleweight'] = 'bold'


    f, axarr = plt.subplots(2, 2,figsize=(15,15))
    st = f.suptitle("One Class NN:  "+dataset, fontsize="x-large",fontweight='bold');

    # print "cae_ocsvm-linear-Train-----", df_cifar_scores["cae_ocsvm-linear-Train"]
    # print "cae_ocsvm-linear-Test-----", df_cifar_scores["cae_ocsvm-linear-Test"]

    _=axarr[0, 0].hist(df_cifar_scores["cae_ocsvm-linear-Train"], bins = 25, label = 'Normal')
    _=axarr[0, 0].hist(df_cifar_scores["cae_ocsvm-linear-Test"], bins = 25, label = 'Anomaly')
    _=axarr[0, 0].set_title("cae-ocsvm :  " + methods[0],fontsize='x-large')



    _=axarr[0, 1].hist(df_cifar_scores["cae_ocsvm-rbf-Train"], bins = 25, label = 'Normal')
    _=axarr[0, 1].hist(df_cifar_scores["cae_ocsvm-rbf-Test"], bins = 25, label = 'Anomaly')
    _=axarr[0, 1].set_title("cae-ocsvm :  " + methods[1],fontsize='x-large')
    _=axarr[0, 1].legend(loc="upper right")


    _=axarr[1, 0].hist(df_cifar_scores["tf_OneClass_NN-Linear-Train"], bins = 25, label = 'Normal')
    _=axarr[1, 0].hist(df_cifar_scores["tf_OneClass_NN-Linear-Test"], bins = 25, label = 'Anomaly')
    _=axarr[1, 0].set_title("oc-nn :  " + activations[0],fontsize='x-large');


    _=axarr[1, 1].hist(df_cifar_scores["tf_OneClass_NN-Sigmoid-Train"], bins = 25, label = 'Normal')
    _=axarr[1, 1].hist(df_cifar_scores["tf_OneClass_NN-Sigmoid-Test"], bins = 25, label = 'Anomaly')
    _=axarr[1, 1].set_title("oc-nn :  " + activations[1],fontsize='x-large');
    _=axarr[1, 1].legend(loc="upper right")

    # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
    # _=plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    _=plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
    # _=plt.title("One Class NN:  cifar ");
    _=plt.legend(loc = 'upper right');

    return






def plot_decision_scores_CIFAR(dataset,df_cifar_scores):
    # Four axes, returned as a 2-d array
    # 
    f, axarr = plt.subplots(3, 2,figsize=(20,20))
    st = f.suptitle("One Class NN:  "+dataset, fontsize="x-large",fontweight='bold');

    _=axarr[0, 0].hist(df_cifar_scores["sklearn-OCSVM-Linear-Train"], bins = 25, label = 'Normal')
    _=axarr[0, 0].hist(df_cifar_scores["sklearn-OCSVM-Linear-Test"], bins = 25, label = 'Anomaly')
    _=axarr[0, 0].set_title("sklearn-OCSVM :  " + methods[0])


    _=axarr[0, 1].hist(df_cifar_scores["sklearn-OCSVM-RBF-Train"], bins = 25, label = 'Normal')
    _=axarr[0, 1].hist(df_cifar_scores["sklearn-OCSVM-RBF-Test"], bins = 25, label = 'Anomaly')
    _=axarr[0, 1].set_title("sklearn-OCSVM :  " + methods[1])
    _=axarr[0, 1].legend(loc="upper right")

    #
    # _=axarr[1, 0].hist(df_cifar_scores["sklearn-OCSVM-explicit-Linear-Train"], bins = 25, label = 'Normal')
    # _=axarr[1, 0].hist(df_cifar_scores["sklearn-OCSVM-explicit-Linear-Test"], bins = 25, label = 'Anomaly')
    # _=axarr[1, 0].set_title("sklearn-OCSVM-explicit :  " + activations[0]);
    #
    # _=axarr[1, 1].hist(df_cifar_scores["sklearn-OCSVM-explicit-Sigmoid-Train"], bins = 25, label = 'Normal')
    # _=axarr[1, 1].hist(df_cifar_scores["sklearn-OCSVM-explicit-Sigmoid-Test"], bins = 25, label = 'Anomaly')
    # _=axarr[1, 1].set_title("sklearn-OCSVM-explicit :  " + activations[1]);
    # _=axarr[1, 1].legend(loc="upper right")

    #
    _=axarr[1, 0].hist(df_cifar_scores["One_Class_NN_explicit-Linear-Train"], bins = 25, label = 'Normal')
    _=axarr[1, 0].hist(df_cifar_scores["One_Class_NN_explicit-Linear-Test"], bins = 25, label = 'Anomaly')
    _=axarr[1, 0].set_title("One_Class_NN_explicit:  " + activations[0]);

    _=axarr[1, 1].hist(df_cifar_scores["One_Class_NN_explicit-Sigmoid-Train"], bins = 25, label = 'Normal')
    _=axarr[1, 1].hist(df_cifar_scores["One_Class_NN_explicit-Sigmoid-Test"], bins = 25, label = 'Anomaly')
    _=axarr[1, 1].set_title("One_Class_NN_explicit:  " + activations[1]);
    _=axarr[1, 1].legend(loc="upper right")


    # _=axarr[3, 0].hist(df_cifar_scores["tf_OneClass_NN-Linear-Train"], bins = 25, label = 'Normal')
    # _=axarr[3, 0].hist(df_cifar_scores["tf_OneClass_NN-Linear-Test"], bins = 25, label = 'Anomaly')
    # _=axarr[3, 0].set_title("tf_OneClass_NN:  " + activations[0]);
    #
    # _=axarr[3, 1].hist(df_cifar_scores["tf_OneClass_NN-Sigmoid-Train"], bins = 25, label = 'Normal')
    # _=axarr[3, 1].hist(df_cifar_scores["tf_OneClass_NN-Sigmoid-Test"], bins = 25, label = 'Anomaly')
    # _=axarr[3, 1].set_title("tf_OneClass_NN:  " + activations[1]);
    # _=axarr[3, 1].legend(loc="upper right")


    _=axarr[2, 0].hist(df_cifar_scores["tf_OneClass_CNN-Linear-Train"], bins = 25, label = 'Normal')
    _=axarr[2, 0].hist(df_cifar_scores["tf_OneClass_CNN-Linear-Test"], bins = 25, label = 'Anomaly')
    _=axarr[2, 0].set_title("tf_OneClass_NN:  " + activations[0]);

    _=axarr[2, 1].hist(df_cifar_scores["tf_OneClass_CNN-Sigmoid-Train"], bins = 25, label = 'Normal')
    _=axarr[2, 1].hist(df_cifar_scores["tf_OneClass_CNN-Sigmoid-Test"], bins = 25, label = 'Anomaly')
    _=axarr[2, 1].set_title("tf_OneClass_NN:  " + activations[1]);
    _=axarr[2, 1].legend(loc="upper right")


    # _=axarr[4, 0].hist(df_cifar_scores["tflearn_OneClass_NN-Linear-Train"], bins = 25, label = 'Normal')
    # _=axarr[4, 0].hist(df_cifar_scores["tflearn_OneClass_NN-Linear-Test"], bins = 25, label = 'Anomaly')
    # _=axarr[4, 0].set_title("tflearn_OneClass_NN:  " + activations[0]);

    # _=axarr[4, 1].hist(df_cifar_scores["tflearn_OneClass_NN-Sigmoid-Train"], bins = 25, label = 'Normal')
    # _=axarr[4, 1].hist(df_cifar_scores["tflearn_OneClass_NN-Sigmoid-Test"], bins = 25, label = 'Anomaly')
    # _=axarr[4, 1].set_title("tflearn_OneClass_NN:  " + activations[1]);
    # _=axarr[4, 1].legend(loc="upper right")

    # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
    # _=plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    _=plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
    # _=plt.title("One Class NN:  cifar ");
    _=plt.legend(loc = 'upper right');

    return 


