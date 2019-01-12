import numpy as np
import pandas as pd
from sklearn import utils
import matplotlib

from scipy.optimize import minimize

dataPath = './data/'



# Create empty dataframe with given column names.
df_usps_scores = {}
df_fake_news_scores = {}
df_spam_vs_ham_scores = {}
df_cifar_10_scores = {}
nu = 0.04

def relu(x):
    y = x
    y[y < 0] = 0
    return y

def dRelu(x):
    y = x
    y[x <= 0] = 0
    y[x > 0]  = np.ones((len(x[x > 0]),))
    return y

def svmScore(X, w,g):
    return g(X.dot(w))

def ocsvm_obj(theta, X, nu, D,g,dG):
    
    w = theta[:D]
    r = theta[D:]
    
    term1 = 0.5 * np.sum(w**2)
    term2 = 1/nu * np.mean(relu(r - svmScore(X, w, g)))
    term3 = -r
    
    return term1 + term2 + term3

def ocsvm_grad(theta, X, nu, D,g,dG):
    
    w = theta[:D]
    r = theta[D:]
    
    deriv = dRelu(r - svmScore(X, w,g))

    term1 = np.append(w, 0)
    term2 = np.append(1/nu * np.mean(deriv[:,np.newaxis] * (-X), axis = 0),
                      1/nu * np.mean(deriv))
    term3 = np.append(0*w, -1)

    grad = term1 + term2 + term3
    
    return grad



def sklearn_OCSVM_explicit_linear(data_train,data_test):



    X  = data_train
    D  = X.shape[1]

    g  = lambda x : x
    dG = lambda x : np.ones(x.shape)

    np.random.seed(42);
    theta0 = np.random.normal(0, 1, D + 1);

    from scipy.optimize import check_grad
    print('Gradient error: %s' % check_grad(ocsvm_obj, ocsvm_grad, theta0, X, nu, D, g,dG));

    res = minimize(ocsvm_obj, theta0, method = 'L-BFGS-B', jac = ocsvm_grad, args = (X, nu, D, g, dG),
                   options = {'gtol': 1e-8, 'disp': True, 'maxiter' : 50000, 'maxfun' : 10000});

    pos_decisionScore = svmScore(data_train, res.x[0:-1],g) - res.x[-1];
    neg_decisionScore = svmScore(data_test, res.x[0:-1],g) - res.x[-1];


    return [pos_decisionScore,neg_decisionScore]

def sklearn_OCSVM_explicit_sigmoid(data_train,data_test):


    X  = data_train
    D  = X.shape[1]

    g   = lambda x : 1/(1 + np.exp(-x))
    dG  = lambda x : 1/(1 + np.exp(-x)) * 1/(1 + np.exp(+x))

    np.random.seed(42);
    theta0 = np.random.normal(0, 1, D + 1);

    print("Inside sklearn_OCSVM_explicit_sigmoid.....")

    from scipy.optimize import check_grad
    print('Gradient error: %s' % check_grad(ocsvm_obj, ocsvm_grad, theta0, X, nu, D, g, dG));

    res = minimize(ocsvm_obj, theta0, method = 'L-BFGS-B', jac = ocsvm_grad, args = (X, nu, D, g, dG),
                   options = {'gtol': 1e-8, 'disp': True, 'maxiter' : 50000, 'maxfun' : 10000});

    pos_decisionScore = svmScore(data_train, res.x[0:-1],g) - res.x[-1];
    neg_decisionScore = svmScore(data_test, res.x[0:-1],g) - res.x[-1];

    return [pos_decisionScore,neg_decisionScore]



def func_getDecision_Scores_sklearn_OCSVM_explicit(dataset,data_train,data_test):


    # print "Decision_Scores_sklearn_OCSVM Using Linear and RBF Kernels....."

    if(dataset=="USPS" ):
        
        result = sklearn_OCSVM_explicit_linear(data_train,data_test)
        df_usps_scores["sklearn-OCSVM-explicit-Linear-Train"] = result[0]
        df_usps_scores["sklearn-OCSVM-explicit-Linear-Test"] =  result[1]
 
        result = sklearn_OCSVM_explicit_sigmoid(data_train,data_test)
        df_usps_scores["sklearn-OCSVM-explicit-Sigmoid-Train"] = result[0]
        df_usps_scores["sklearn-OCSVM-explicit-Sigmoid-Test"] = result[1]

    if(dataset=="FAKE_NEWS" ):   
        result = sklearn_OCSVM_explicit_linear(data_train,data_test)
        df_fake_news_scores["sklearn-OCSVM-explicit-Linear-Train"] = result[0]
        df_fake_news_scores["sklearn-OCSVM-explicit-Linear-Test"] = result[1]
        
        result = sklearn_OCSVM_explicit_sigmoid(data_train,data_test)
        df_fake_news_scores["sklearn-OCSVM-explicit-Sigmoid-Train"] = result[0]
        df_fake_news_scores["sklearn-OCSVM-explicit-Sigmoid-Test"] = result[1]

    # if(dataset=="SPAM_Vs_HAM" ):
    #     result = sklearn_OCSVM_explicit_linear(data_train,data_test)
    #     df_spam_vs_ham_scores["sklearn-OCSVM-explicit-Linear-Train"] = result[0] 
    #     df_spam_vs_ham_scores["sklearn-OCSVM-explicit-Linear-Test"] = result[1]
        
    #     result = sklearn_OCSVM_explicit_sigmoid(data_train,data_test)
    #     df_spam_vs_ham_scores["sklearn-OCSVM-explicit-Sigmoid-Train"] = result[0]
    #     df_spam_vs_ham_scores["sklearn-OCSVM-explicit-Sigmoid-Test"] = result[1]

    if(dataset=="CIFAR-10" ):
        result = sklearn_OCSVM_explicit_linear(data_train,data_test)
        df_cifar_10_scores["sklearn-OCSVM-explicit-Linear-Train"] = result[0]
        df_cifar_10_scores["sklearn-OCSVM-explicit-Linear-Test"] = result[1]
        
        result = sklearn_OCSVM_explicit_sigmoid(data_train,data_test)
        df_cifar_10_scores["sklearn-OCSVM-explicit-Sigmoid-Train"] = result[0]
        df_cifar_10_scores["sklearn-OCSVM-explicit-Sigmoid-Test"] = result[1]

    return [df_usps_scores,df_fake_news_scores,df_spam_vs_ham_scores,df_cifar_10_scores]




