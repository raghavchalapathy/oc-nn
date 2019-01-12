import numpy as np
import pandas as pd
from sklearn import utils
import matplotlib.pyplot as plt

dataPath = './data/'

methods = ["Linear","Sigmoid"]

import matplotlib.pyplot as plt
from matplotlib import style



def plot_decision_scores_One_Class_NN_explicit(model,dataset,df_usps,df_fake_news,df_spam_vs_ham,df_cifar):

    # set up figure & axes
    fig = plt.figure();

    ## PLot for USPS 
    
    if(dataset=="USPS"):

            ax = fig.add_subplot(2, 2, 1);
            _=ax.hist(df_usps["One_Class_NN_explicit-Linear-Train"], bins = 25, label = 'Normal');
            _=ax.hist(df_usps["One_Class_NN_explicit-Linear-Test"], bins = 25, label = 'Anomaly');
            _=plt.title("One_Class_NN_explicit :  " + methods[0]+" : "+dataset);
            _=plt.tight_layout();
            
            ax=fig.add_subplot(222);
            _=ax.hist(df_usps["One_Class_NN_explicit-Sigmoid-Train"], bins = 25, label = 'Normal');
            _=ax.hist(df_usps["One_Class_NN_explicit-Sigmoid-Test"], bins = 25, label = 'Anomaly');
            _=plt.title("One_Class_NN_explicit:  " + methods[1]+" : "+dataset);
            _=plt.legend(loc = 'upper right');
            _=plt.tight_layout();


        # ## PLot for CIFAR-10
    if(dataset=="CIFAR-10"):
            ax = fig.add_subplot(2, 2, 3);
            _=ax.hist(df_cifar["One_Class_NN_explicit-Linear-Train"], bins = 25, label = 'Normal');
            _=ax.hist(df_cifar["One_Class_NN_explicit-Linear-Test"], bins = 25, label = 'Anomaly');
            _=plt.title("One_Class_NN_explicit :  " + methods[0]+" : "+dataset);
            _=plt.tight_layout();
            
            ax=fig.add_subplot(224);
            _=ax.hist(df_cifar["One_Class_NN_explicit_Sigmoid-Train"], bins = 25, label = 'Normal');
            _=ax.hist(df_cifar["One_Class_NN_explicit_Sigmoid-Test"], bins = 25, label = 'Anomaly');
            _=plt.title("One_Class_NN_explicit:  " + methods[1]+" : "+dataset);
            _=plt.tight_layout();


        # # ## PLot for FAKE_NEWS
    if(dataset=="FAKE_NEWS" ):  
            ax = fig.add_subplot(2, 2, 1);
            _=ax.hist(df_fake_news["One_Class_NN_explicit-Linear-Train"], bins = 25, label = 'Normal');
            _=ax.hist(df_fake_news["One_Class_NN_explicit-Linear-Test"], bins = 25, label = 'Anomaly');
            _=plt.title("One_Class_NN_explicit :  " + methods[0]+" : "+dataset);
            _=plt.tight_layout();
            
            ax=fig.add_subplot(222);
            _=ax.hist(df_fake_news["One_Class_NN_explicit-Sigmoid-Train"], bins = 25, label = 'Normal');
            _=ax.hist(df_fake_news["One_Class_NN_explicit-Sigmoid-Test"], bins = 25, label = 'Anomaly');

            _=plt.title("One_Class_NN_explicit :  " + methods[1]+" : "+dataset);
            _=plt.tight_layout();
        
         


    #      # ## PLot for SPAM Vs HAM
    # if(dataset=="SPAM_Vs_HAM"):
    #         ax = fig.add_subplot(2, 2, 3);
    #         _=ax.hist(df_spam_vs_ham["One_Class_NN_explicit-Linear-Train"], bins = 25, label = 'Normal');
    #         _=ax.hist(df_spam_vs_ham["One_Class_NN_explicit-Linear-Test"], bins = 25, label = 'Anomaly');
    #         _=plt.title("One_Class_NN_explicit :  " + methods[0]+" : "+dataset);
    #         _=plt.tight_layout();
            
    #         ax=fig.add_subplot(224);
    #         _=ax.hist(df_spam_vs_ham["One_Class_NN_explicit-Sigmoid-Train"], bins = 25, label = 'Normal');
    #         _=ax.hist(df_spam_vs_ham["One_Class_NN_explicit-Sigmoid-Test"], bins = 25, label = 'Anomaly');
    #         _=plt.title("One_Class_NN_explicit :  " + methods[1]+" : "+dataset);
    #         _=plt.tight_layout();



    return 

