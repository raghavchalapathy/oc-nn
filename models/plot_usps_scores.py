import numpy as np
import pandas as pd
from sklearn import utils
import matplotlib.pyplot as plt

dataPath = './data/'
activations = ["Linear","Sigmoid"]
methods = ["Linear","RBF"]



def plot_decision_scores_USPS(dataset,df_usps_scores):
    # Four axes, returned as a 2-d array

    # Four axes, returned as a 2-d array

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
    f, axarr = plt.subplots(2, 2, figsize=(20, 20))
    st = f.suptitle("One Class NN:  " + dataset);

    # print "cae_ocsvm-linear-Train-----", df_cifar_scores["cae_ocsvm-linear-Train"]
    # print "cae_ocsvm-linear-Test-----", df_cifar_scores["cae_ocsvm-linear-Test"]
    _ = axarr[0, 0].hist(df_usps_scores["cae_ocsvm-linear-Train"], bins=25, label='Normal')
    _ = axarr[0, 0].hist(df_usps_scores["cae_ocsvm-linear-Test"], bins=25, label='Anomaly')
    _ = axarr[0, 0].set_title("cae-ocsvm :  " + methods[0])

    # print "cae_ocsvm-rbf-Train-----", df_cifar_scores["cae_ocsvm-rbf-Train"]
    # print "cae_ocsvm-rbf-Test-----", df_cifar_scores["cae_ocsvm-rbf-Test"]
    _ = axarr[0, 1].hist(df_usps_scores["cae_ocsvm-rbf-Train"], bins=25, label='Normal')
    _ = axarr[0, 1].hist(df_usps_scores["cae_ocsvm-rbf-Test"], bins=25, label='Anomaly')
    _ = axarr[0, 1].set_title("cae-ocsvm :  " + methods[1])
    _ = axarr[0, 1].legend(loc="upper right")

    _ = axarr[1, 0].hist(df_usps_scores["tf_OneClass_NN-Linear-Train"], bins=25, label='Normal')
    _ = axarr[1, 0].hist(df_usps_scores["tf_OneClass_NN-Linear-Test"], bins=25, label='Anomaly')
    _ = axarr[1, 0].set_title("oc-nn :  " + activations[0]);

    _ = axarr[1, 1].hist(df_usps_scores["tf_OneClass_NN-Sigmoid-Train"], bins=25, label='Normal')
    _ = axarr[1, 1].hist(df_usps_scores["tf_OneClass_NN-Sigmoid-Test"], bins=25, label='Anomaly')
    _ = axarr[1, 1].set_title("oc-nn :  " + activations[1]);
    _ = axarr[1, 1].legend(loc="upper right")

    # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
    # _=plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    _ = plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
    # _=plt.title("One Class NN:  cifar ");
    _ = plt.legend(loc='upper right');

    return


def plot_decision_scores_USPS_new(dataset,df_usps_scores):
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

    plt.savefig("/Users/raghav/Documents/Uni/oc-nn/results/usps/"+"oc-svm-linear.png")


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


    plt.savefig("/Users/raghav/Documents/Uni/oc-nn/results/usps/"+"oc-svm-rbf.png")


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

    plt.savefig("/Users/raghav/Documents/Uni/oc-nn/results/usps/"+"oc-nn-linear.png")


    pos_decisionScore = df_usps_scores["tf_OneClass_NN-Sigmoid-Train"]
    neg_decisionScore = df_usps_scores["tf_OneClass_NN-Sigmoid-Test"]
    fig, axarr1 = plt.subplots(1, 1, figsize=(20, 30))
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.hist(pos_decisionScore, bins=25, label='Normal')
    plt.hist(neg_decisionScore, bins=25, label='Anomaly')
    plt.legend(loc="upper right", fontsize=40)
    plt.title("oc-nn : sigmoid",fontsize = 40)

    plt.savefig("/Users/raghav/Documents/Uni/oc-nn/results/usps/"+"oc-nn-sigmoid.png")


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

