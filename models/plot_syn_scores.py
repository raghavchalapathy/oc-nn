import numpy as np
import pandas as pd
from sklearn import utils
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
# SMALL_SIZE = 10
# MEDIUM_SIZE = 12
# BIGGER_SIZE = 20
#
# plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

dataPath = './data/'
activations = ["Linear","Sigmoid"]
methods = ["Linear","RBF"]

# plt.rcParams.update({'axes.titlesize': 'x-large'})


def plot_decision_scores_Synthetic(dataset,df_usps_scores):
    # Four axes, returned as a 2-d array

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
    #
    f, axarr = plt.subplots(2, 2,figsize=(15,15))
    # st = f.suptitle("One Class NN:  "+dataset, fontsize="x-large",fontweight='bold');
    plt.xticks(np.arange(-4, 4, 0.5))
    _=axarr[0, 0].hist(df_usps_scores["sklearn-OCSVM-Linear-Train"], bins = 25, label = 'Normal')
    _=axarr[0, 0].hist(df_usps_scores["sklearn-OCSVM-Linear-Test"], bins = 25, label = 'Anomaly')
    _=axarr[0, 0].set_title("oc-svm :  " + methods[0])


    # _ = axarr[0, 1].set_xlim(-3,3)
    # plt.xticks(np.arange(-4, 4, 0.5))
    _=axarr[0, 1].hist(df_usps_scores["sklearn-OCSVM-RBF-Train"], bins = 25, label = 'Normal')
    _=axarr[0, 1].hist(df_usps_scores["sklearn-OCSVM-RBF-Test"], bins = 25, label = 'Anomaly')
    _=axarr[0, 1].set_title("ocsvm :  " + methods[1])
    _=axarr[0, 1].legend(loc="upper right")




    # _=axarr[1, 0].hist(df_usps_scores["sklearn-OCSVM-explicit-Linear-Train"], bins = 25, label = 'Normal')
    # _=axarr[1, 0].hist(df_usps_scores["sklearn-OCSVM-explicit-Linear-Test"], bins = 25, label = 'Anomaly')
    # _=axarr[1, 0].set_title("sklearn-OCSVM-explicit :  " + activations[0]);
    #
    # _=axarr[1, 1].hist(df_usps_scores["sklearn-OCSVM-explicit-Sigmoid-Train"], bins = 25, label = 'Normal')
    # _=axarr[1, 1].hist(df_usps_scores["sklearn-OCSVM-explicit-Sigmoid-Test"], bins = 25, label = 'Anomaly')
    # _=axarr[1, 1].set_title("sklearn-OCSVM-explicit :  " + activations[1]);
    # _=axarr[1, 1].legend(loc="upper right")



    # _=axarr[2, 0].hist(df_usps_scores["One_Class_NN_explicit-Linear-Train"], bins = 25, label = 'Normal')
    # _=axarr[2, 0].hist(df_usps_scores["One_Class_NN_explicit-Linear-Test"], bins = 25, label = 'Anomaly')
    # _=axarr[2, 0].set_title("One_Class_NN_explicit:  " + activations[0]);
    #
    # _=axarr[2, 1].hist(df_usps_scores["One_Class_NN_explicit-Sigmoid-Train"], bins = 25, label = 'Normal')
    # _=axarr[2, 1].hist(df_usps_scores["One_Class_NN_explicit-Sigmoid-Test"], bins = 25, label = 'Anomaly')
    # _=axarr[2, 1].set_title("One_Class_NN_explicit:  " + activations[1]);
    # _=axarr[2, 1].legend(loc="upper right")


    _=axarr[1, 0].hist(df_usps_scores["tf_OneClass_NN-Linear-Train"], bins = 25, label = 'Normal')
    _=axarr[1, 0].hist(df_usps_scores["tf_OneClass_NN-Linear-Test"], bins = 25, label = 'Anomaly')
    _=axarr[1, 0].set_title("oc-nn:  " + activations[0]);

    _=axarr[1, 1].hist(df_usps_scores["tf_OneClass_NN-Sigmoid-Train"], bins = 25, label = 'Normal')
    _=axarr[1, 1].hist(df_usps_scores["tf_OneClass_NN-Sigmoid-Test"], bins = 25, label = 'Anomaly')
    _=axarr[1, 1].set_title("oc-nn :  " + activations[1]);
    _=axarr[1, 1].legend(loc="upper right")


    _=plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)


    return df_usps_scores


def plot_decision_scores_SYN_new(dataset,df_usps_scores):
    # Four axes, returned as a 2-d array
    # 
    fig, axarr1 = plt.subplots(1, 1, figsize=(20, 30))
    pos_decisionScore = df_usps_scores["sklearn-OCSVM-Linear-Train"]
    neg_decisionScore = df_usps_scores["sklearn-OCSVM-Linear-Test"]
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.hist(pos_decisionScore, bins=25, label='Normal')
    plt.hist(neg_decisionScore, bins=25, label='Anomaly')
    # plt.title("One Class NN:  " + "Synthetic Data", fontsize="x-large", fontweight='bold')
    plt.legend(loc="upper right", fontsize=40)
    plt.title("oc-svm: linear", fontsize=40)

    plt.savefig("/Users/raghav/Documents/Uni/oc-nn/results/synthetic/"+"oc-svm-linear.png")


    pos_decisionScore = df_usps_scores["sklearn-OCSVM-RBF-Train"]
    neg_decisionScore = df_usps_scores["sklearn-OCSVM-RBF-Test"]
    fig, axarr1 = plt.subplots(1, 1, figsize=(20, 30))
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.xticks(range(-4, 4, 1))
    plt.hist(pos_decisionScore, bins=25, label='Normal')
    plt.hist(neg_decisionScore, bins=25, label='Anomaly')
    plt.legend(loc="upper right", fontsize=40)
    plt.title("oc-svm: rbf", fontsize=40)


    plt.savefig("/Users/raghav/Documents/Uni/oc-nn/results/synthetic/"+"oc-svm-rbf.png")


    pos_decisionScore = df_usps_scores["tf_OneClass_NN-Linear-Train"]
    neg_decisionScore = df_usps_scores["tf_OneClass_NN-Linear-Test"]
    fig, axarr1 = plt.subplots(1, 1, figsize=(20, 30))
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.hist(pos_decisionScore, bins=25, label='Normal')
    plt.hist(neg_decisionScore, bins=25, label='Anomaly')
    plt.legend(loc="upper right",fontsize = 20)

    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.title('oc-nn : linear', fontsize=40)

    plt.savefig("/Users/raghav/Documents/Uni/oc-nn/results/synthetic/"+"oc-nn-linear.png")


    pos_decisionScore = df_usps_scores["tf_OneClass_NN-Sigmoid-Train"]
    neg_decisionScore = df_usps_scores["tf_OneClass_NN-Sigmoid-Test"]
    fig, axarr1 = plt.subplots(1, 1, figsize=(20, 30))
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.hist(pos_decisionScore, bins=25, label='Normal')
    plt.hist(neg_decisionScore, bins=25, label='Anomaly')
    plt.legend(loc="upper right", fontsize=40)
    plt.title("oc-nn : sigmoid",fontsize = 40)

    plt.savefig("/Users/raghav/Documents/Uni/oc-nn/results/synthetic/"+"oc-nn-sigmoid.png")


    return 



def plot_decision_scores(model,dataset,df_usps_scores,df_fake_news_scores,df_spam_vs_ham_scores,df_cifar_10_scores):

    df_usps = pd.DataFrame(df_usps_scores.items(), columns=df_usps_scores.keys())
    df_cifar = pd.DataFrame(df_cifar_10_scores.items(), columns=df_cifar_10_scores.keys())

    ## PLot for USPS 
    if(dataset=="USPS" ):
        plt.hist(df_usps["sklearn-OCSVM-Linear-Train"], bins = 25, label = 'Normal');
        plt.hist(df_usps["sklearn-OCSVM-Linear-Test"], bins = 25, label = 'Anomaly');
        plt.legend(loc = 'upper right');
        plt.title('OC-SVM Normalised Decision Score for : '+dataset+ ": "+ model);

        plt.hist(df_usps["sklearn-OCSVM-RBF-Train"], bins = 25, label = 'Normal');
        plt.hist(df_usps["sklearn-OCSVM-RBF-Test"], bins = 25, label = 'Anomaly');
        plt.legend(loc = 'upper right');
        plt.title('OC-SVM Normalised Decision Score for : '+dataset+ ": "+ model);


    ## PLot for CIFAR-10
    if(dataset=="CIFAR-10" ):
        plt.hist(df_cifar["sklearn-OCSVM-Linear-Train"], bins = 25, label = 'Normal');
        plt.hist(df_cifar["sklearn-OCSVM-Linear-Test"], bins = 25, label = 'Anomaly');
        plt.legend(loc = 'upper right');
        plt.title('OC-SVM Normalised Decision Score for : '+dataset+ ": "+ model);

        plt.hist(df_cifar["sklearn-OCSVM-RBF-Train"], bins = 25, label = 'Normal');
        plt.hist(df_cifar["sklearn-OCSVM-RBF-Test"], bins = 25, label = 'Anomaly');
        plt.legend(loc = 'upper right');
        plt.title('OC-SVM Normalised Decision Score for : '+dataset+ ": "+ model);

   
    return 

