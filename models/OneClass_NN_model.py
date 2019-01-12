import numpy as np
import pandas as pd
from sklearn import utils
import matplotlib
from scipy.optimize import minimize



dataPath = './data/'

colNames = ["sklearn-OCSVM-Linear-Train","sklearn-OCSVM-RBF-Train","sklearn-OCSVM-Linear-Test","sklearn-OCSVM-RBF-Test","sklearn-explicit-Linear-Train","sklearn-explicit-Sigmoid-Train","sklearn-explicit-Linear-Test","sklearn-explicit-Sigmoid-Test","tf-Linear-Train","tf-Sigmoid-Train","tf-Linear-Test","tf-Sigmoid-Test","tfLearn-Linear-Train","tfLearn-Sigmoid-Train","tfLearn-Linear-Test","tfLearn-Sigmoid-Test"]
from tf_OneClass_CNN_model import func_get_ImageVectors
dataPathTrain = '/Users/raghav/Documents/Uni/oc-nn/data/cifar-10_data/train/dogs/'
dataPathTest = '/Users/raghav/Documents/Uni/oc-nn/data/cifar-10_data/test/'

# Create empty dataframe with given column names.
df_usps_scores = {}
df_fake_news_scores = {}
df_spam_vs_ham_scores = {}
df_cifar_10_scores = {}
nu = 0.002
# nu = 0.04 # for usps
K  = 2


def relu(x):
    y = x
    y[y < 0] = 0
    return y

def dRelu(x):
    y = x
    y[x <= 0] = 0
    y[x > 0]  = np.ones((len(x[x > 0]),))
    return y

def nnScore(X, w, V, g):
    return g(X.dot(V)).dot(w)

def ocnn_obj(theta, X, nu, D, K, g, dG):
    
    w = theta[:K]
    V = theta[K:K+K*D].reshape((D, K))
    r = theta[K+K*D:]
    
    term1 = 0.5  * np.sum(w**2)
    term2 = 0.5  * np.sum(V**2)
    term3 = 1/nu * np.mean(relu(r - nnScore(X, w, V, g)))
    term4 = -r
    
    return term1 + term2 + term3 + term4

def ocnn_grad(theta, X, nu, D, K, g, dG):
    
    N = X.shape[0]
    w = theta[:K]
    V = theta[K:K+K*D].reshape((D, K))
    r = theta[K+K*D:]
    
    deriv = dRelu(r - nnScore(X, w, V, g))    

    term1 = np.concatenate(( w,
                             np.zeros((V.size,)),
                             np.zeros((1,)) ))

    term2 = np.concatenate(( np.zeros((w.size,)),
                             V.flatten(),
                             np.zeros((1,)) ))

    term3 = np.concatenate(( 1/nu * np.mean(deriv[:,np.newaxis] * (-g(X.dot(V))), axis = 0),
                             1/nu * np.mean((deriv[:,np.newaxis] * (dG(X.dot(V)) * -w)).reshape((N, 1, K)) * X.reshape((N, D, 1)), axis = 0).flatten(),
                             1/nu * np.array([ np.mean(deriv) ]) ))
    
    term4 = np.concatenate(( np.zeros((w.size,)),
                             np.zeros((V.size,)),
                             -1 * np.ones((1,)) ))
    
    return term1 + term2 + term3 + term4



def One_Class_NN_explicit_linear(data_train,data_test):


    X  = data_train
    D  = X.shape[1]

    g  = lambda x : x
    dG = lambda x : np.ones(x.shape)

    np.random.seed(42)
    theta0 = np.random.normal(0, 1, K + K*D + 1)

    from scipy.optimize import check_grad
    print('Gradient error: %s' % check_grad(ocnn_obj, ocnn_grad, theta0, X, nu, D, K, g, dG))

    res = minimize(ocnn_obj, theta0, method = 'L-BFGS-B', jac = ocnn_grad, args = (X, nu, D, K, g, dG),
                   options = {'gtol': 1e-8, 'disp': True, 'maxiter' : 50000, 'maxfun' : 10000})

    thetaStar = res.x

    wStar = thetaStar[:K]
    VStar = thetaStar[K:K+K*D].reshape((D, K))
    rStar = thetaStar[K+K*D:]

    pos_decisionScore = nnScore(data_train, wStar, VStar, g) - rStar
    neg_decisionScore = nnScore(data_test, wStar, VStar, g) - rStar

    print "pos_decisionScore", np.sort(pos_decisionScore)
    print "neg_decisionScore", np.sort(neg_decisionScore)


    return [pos_decisionScore,neg_decisionScore]


def One_Class_NN_explicit_sigmoid(data_train,data_test):

    X  = data_train
    D  = X.shape[1]


    g   = lambda x : 1/(1 + np.exp(-x))
    dG  = lambda x : 1/(1 + np.exp(-x)) * 1/(1 + np.exp(+x))

    np.random.seed(42)
    theta0 = np.random.normal(0, 1, K + K*D + 1)

    from scipy.optimize import check_grad
    print('Gradient error: %s' % check_grad(ocnn_obj, ocnn_grad, theta0, X, nu, D, K, g, dG))

    res = minimize(ocnn_obj, theta0, method = 'L-BFGS-B', jac = ocnn_grad, args = (X, nu, D, K, g, dG),
                   options = {'gtol': 1e-8, 'disp': True, 'maxiter' : 50000, 'maxfun' : 10000})

    thetaStar = res.x

    wStar = thetaStar[:K]
    VStar = thetaStar[K:K+K*D].reshape((D, K))
    rStar = thetaStar[K+K*D:]

    pos_decisionScore = nnScore(data_train, wStar, VStar, g) - rStar
    neg_decisionScore = nnScore(data_test, wStar, VStar, g) - rStar
    print "pos_decisionScore", np.sort(pos_decisionScore)
    print "neg_decisionScore", np.sort(neg_decisionScore)

    return [pos_decisionScore,neg_decisionScore]



def func_getDecision_Scores_One_Class_NN_explicit(dataset,data_train,data_test):


    # print "Decision_Scores_sklearn_OCSVM Using Linear and RBF Kernels....."

    if(dataset=="USPS" ):
        
        result = One_Class_NN_explicit_linear(data_train,data_test)
        df_usps_scores["One_Class_NN_explicit-Linear-Train"] = result[0]
        df_usps_scores["One_Class_NN_explicit-Linear-Test"] =  result[1]
 
        result = One_Class_NN_explicit_sigmoid(data_train,data_test)
        df_usps_scores["One_Class_NN_explicit-Sigmoid-Train"] = result[0]
        df_usps_scores["One_Class_NN_explicit-Sigmoid-Test"] = result[1]


    if(dataset=="FAKE_NEWS" ):   
        result = One_Class_NN_explicit_linear(data_train,data_test)
        df_fake_news_scores["One_Class_NN_explicit-Linear-Train"] = result[0]
        df_fake_news_scores["One_Class_NN_explicit-Linear-Test"] = result[1]
        
        result = One_Class_NN_explicit_sigmoid(data_train,data_test)
        df_fake_news_scores["One_Class_NN_explicit-Sigmoid-Train"] = result[0]
        df_fake_news_scores["One_Class_NN_explicit-Sigmoid-Test"] = result[1]


    # if(dataset=="SPAM_Vs_HAM" ):
    #     result = One_Class_NN_explicit_linear(data_train,data_test)
    #     df_spam_vs_ham_scores["One_Class_NN_explicit-Linear-Train"] = result[0] 
    #     df_spam_vs_ham_scores["One_Class_NN_explicit-Linear-Test"] = result[1]
        
    #     result = One_Class_NN_explicit_sigmoid(data_train,data_test)
    #     df_spam_vs_ham_scores["One_Class_NN_explicit-Sigmoid-Train"] = result[0]
    #     df_spam_vs_ham_scores["One_Class_NN_explicit-Sigmoid-Test"] = result[1]


    if(dataset=="CIFAR-10" ):
        result = One_Class_NN_explicit_linear(data_train,data_test)
        df_cifar_10_scores["One_Class_NN_explicit-Linear-Train"] = result[0]
        df_cifar_10_scores["One_Class_NN_explicit-Linear-Test"] = result[1]
        
        result = One_Class_NN_explicit_sigmoid(data_train,data_test)
        df_cifar_10_scores["One_Class_NN_explicit_Sigmoid-Train"] = result[0]
        df_cifar_10_scores["One_Class_NN_explicit_Sigmoid-Test"] = result[1]

    return [df_usps_scores,df_fake_news_scores,df_spam_vs_ham_scores,df_cifar_10_scores]




