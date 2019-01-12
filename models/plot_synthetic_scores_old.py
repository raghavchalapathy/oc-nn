import numpy as np
import pandas as pd
from sklearn import utils
import matplotlib.pyplot as plt

dataPath = './data/'
activations = ["Linear","Sigmoid"]
methods = ["Linear","RBF"]



def plot_decision_scores_USPS(dataset,df_usps_scores):
    # Four axes, returned as a 2-d array
    # 
    f, axarr = plt.subplots(4, 2,figsize=(20,20))
    st = f.suptitle("One Class NN:  "+dataset, fontsize="x-large",fontweight='bold');

    _=axarr[0, 0].hist(df_usps_scores["sklearn-OCSVM-Linear-Train"], bins = 25, label = 'Normal')
    _=axarr[0, 0].hist(df_usps_scores["sklearn-OCSVM-Linear-Test"], bins = 25, label = 'Anomaly')
    _=axarr[0, 0].set_title("sklearn-OCSVM :  " + methods[0])


    _=axarr[0, 1].hist(df_usps_scores["sklearn-OCSVM-RBF-Train"], bins = 25, label = 'Normal')
    _=axarr[0, 1].hist(df_usps_scores["sklearn-OCSVM-RBF-Test"], bins = 25, label = 'Anomaly')
    _=axarr[0, 1].set_title("sklearn-OCSVM :  " + methods[1])
    _=axarr[0, 1].legend(loc="upper right")



    _=axarr[1, 0].hist(df_usps_scores["sklearn-OCSVM-explicit-Linear-Train"], bins = 25, label = 'Normal')
    _=axarr[1, 0].hist(df_usps_scores["sklearn-OCSVM-explicit-Linear-Test"], bins = 25, label = 'Anomaly')
    _=axarr[1, 0].set_title("sklearn-OCSVM-explicit :  " + activations[0]);

    _=axarr[1, 1].hist(df_usps_scores["sklearn-OCSVM-explicit-Sigmoid-Train"], bins = 25, label = 'Normal')
    _=axarr[1, 1].hist(df_usps_scores["sklearn-OCSVM-explicit-Sigmoid-Test"], bins = 25, label = 'Anomaly')
    _=axarr[1, 1].set_title("sklearn-OCSVM-explicit :  " + activations[1]);
    _=axarr[1, 1].legend(loc="upper right")



    _=axarr[2, 0].hist(df_usps_scores["One_Class_NN_explicit-Linear-Train"], bins = 25, label = 'Normal')
    _=axarr[2, 0].hist(df_usps_scores["One_Class_NN_explicit-Linear-Test"], bins = 25, label = 'Anomaly')
    _=axarr[2, 0].set_title("One_Class_NN_explicit:  " + activations[0]);

    _=axarr[2, 1].hist(df_usps_scores["One_Class_NN_explicit-Sigmoid-Train"], bins = 25, label = 'Normal')
    _=axarr[2, 1].hist(df_usps_scores["One_Class_NN_explicit-Sigmoid-Test"], bins = 25, label = 'Anomaly')
    _=axarr[2, 1].set_title("One_Class_NN_explicit:  " + activations[1]);
    _=axarr[2, 1].legend(loc="upper right")


    _=axarr[3, 0].hist(df_usps_scores["tf_OneClass_NN-Linear-Train"], bins = 25, label = 'Normal')
    _=axarr[3, 0].hist(df_usps_scores["tf_OneClass_NN-Linear-Test"], bins = 25, label = 'Anomaly')
    _=axarr[3, 0].set_title("tf_OneClass_NN:  " + activations[0]);

    _=axarr[3, 1].hist(df_usps_scores["tf_OneClass_NN-Sigmoid-Train"], bins = 25, label = 'Normal')
    _=axarr[3, 1].hist(df_usps_scores["tf_OneClass_NN-Sigmoid-Test"], bins = 25, label = 'Anomaly')
    _=axarr[3, 1].set_title("tf_OneClass_NN:  " + activations[1]);
    _=axarr[3, 1].legend(loc="upper right")


    # _=axarr[4, 0].hist(df_usps_scores["tflearn_OneClass_NN-Linear-Train"], bins = 25, label = 'Normal')
    # _=axarr[4, 0].hist(df_usps_scores["tflearn_OneClass_NN-Linear-Test"], bins = 25, label = 'Anomaly')
    # _=axarr[4, 0].set_title("tflearn_OneClass_NN:  " + activations[0]);
    #
    # _=axarr[4, 1].hist(df_usps_scores["tflearn_OneClass_NN-Sigmoid-Train"], bins = 25, label = 'Normal')
    # _=axarr[4, 1].hist(df_usps_scores["tflearn_OneClass_NN-Sigmoid-Test"], bins = 25, label = 'Anomaly')
    # _=axarr[4, 1].set_title("tflearn_OneClass_NN:  " + activations[1]);
    # _=axarr[4, 1].legend(loc="upper right")

    # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
    # _=plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    _=plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)


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

