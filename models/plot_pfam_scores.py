import numpy as np
import pandas as pd
from sklearn import utils
# import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

dataPath = './data/'
activations = ["Linear","Sigmoid"]
methods = ["Linear","RBF"]




def plot_decision_scores_pfam(dataset,df_pfam_scores):
    # Four axes, returned as a 2-d array
    #
    # import matplotlib as mpl
    # label_size = 60
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

    # print "cae_ocsvm-linear-Train-----", df_pfam_scores["cae_ocsvm-linear-Train"]
    # print "cae_ocsvm-linear-Test-----", df_pfam_scores["cae_ocsvm-linear-Test"]
    _=axarr[0, 0].hist(df_pfam_scores["lstm_ocsvm-linear-Train"], bins = 25, label = 'Normal')
    _=axarr[0, 0].hist(df_pfam_scores["lstm_ocsvm-linear-Test"], bins = 25, label = 'Anomaly')
    _=axarr[0, 0].set_title("lstm-ae-ocsvm :  " + methods[0])

    # print "cae_ocsvm-rbf-Train-----", df_pfam_scores["cae_ocsvm-rbf-Train"]
    # print "cae_ocsvm-rbf-Test-----", df_pfam_scores["cae_ocsvm-rbf-Test"]
    _=axarr[0, 1].hist(df_pfam_scores["lstm_ocsvm-rbf-Train"], bins = 25, label = 'Normal')
    _=axarr[0, 1].hist(df_pfam_scores["lstm_ocsvm-rbf-Test"], bins = 25, label = 'Anomaly')
    _=axarr[0, 1].set_title("lstm-ae-ocsvm :  " + methods[1])
    _=axarr[0, 1].legend(loc="upper right")


    _=axarr[1, 0].hist(df_pfam_scores["tf_OneClass_NN-Linear-Train"], bins = 25, label = 'Normal')
    _=axarr[1, 0].hist(df_pfam_scores["tf_OneClass_NN-Linear-Test"], bins = 25, label = 'Anomaly')
    _=axarr[1, 0].set_title("oc-nn :  " + activations[0]);

    _=axarr[1, 1].hist(df_pfam_scores["tf_OneClass_NN-Sigmoid-Train"], bins = 25, label = 'Normal')
    _=axarr[1, 1].hist(df_pfam_scores["tf_OneClass_NN-Sigmoid-Test"], bins = 25, label = 'Anomaly')
    _=axarr[1, 1].set_title("oc-nn :  " + activations[1]);
    _=axarr[1, 1].legend(loc="upper right")

    # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
    # _=plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    _=plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
    # _=plt.title("One Class NN:  cifar ");
    _=plt.legend(loc = 'upper right');

    return


