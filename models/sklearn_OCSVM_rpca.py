from r_pca import R_pca
from sklearn import svm

import time

def sklearn__RPCA_OCSVM(data_train,data_test,nu):

    print "sklearn__RPCA_OCSVM========",data_train.shape
    # Obtain the projections for training test
    rpca_train = R_pca(data_train)
    L_train, S = rpca_train.fit(max_iter=10, iter_print=100)
    ## Obtain the projections fro data_test
    rpca_test = R_pca(data_test)
    L_test, S = rpca_test.fit(max_iter=10, iter_print=100)

    print L_train.shape
    print L_test.shape
    nu = 0.05

    clf = svm.OneClassSVM(nu=nu, kernel="rbf", gamma=0.1)
    start_time = time.time()
    clf.fit(L_train)
    trainTime = time.time() - start_time

    start_time = time.time()
    pos_decisionScore = clf.predict(L_train)
    neg_decisionScore = clf.predict(L_test)
    testTime = time.time() - start_time

    return [pos_decisionScore, neg_decisionScore,trainTime,testTime]


def sklearn_RPCA_OCSVM_Linear(data_train,data_test,nu):

    print "sklearn__RPCA_OCSVM========",data_train.shape
    # Obtain the projections for training test
    rpca_train = R_pca(data_train)
    L_train, S = rpca_train.fit(max_iter=10000, iter_print=100)
    ## Obtain the projections fro data_test
    rpca_test = R_pca(data_test)
    L_test, S = rpca_test.fit(max_iter=10000, iter_print=100)

    clf = svm.OneClassSVM(nu=nu, kernel="linear", gamma=0.1)
    start_time = time.time()
    clf.fit(L_train)
    trainTime = time.time() - start_time

    start_time = time.time()
    pos_decisionScore = clf.predict(L_train)
    neg_decisionScore = clf.predict(L_test)
    testTime = time.time() - start_time

    return [pos_decisionScore, neg_decisionScore,trainTime,testTime]


def sklearn_RPCA_OCSVM_rbf(data_train,data_test,nu):

    print "sklearn__RPCA_OCSVM========",data_train.shape
    # Obtain the projections for training test
    rpca_train = R_pca(data_train)
    L_train, S = rpca_train.fit(max_iter=10000, iter_print=100)
    ## Obtain the projections fro data_test
    rpca_test = R_pca(data_test)
    L_test, S = rpca_test.fit(max_iter=10000, iter_print=100)

    clf = svm.OneClassSVM(nu=nu, kernel="rbf", gamma=0.1)
    start_time = time.time()
    clf.fit(L_train)
    trainTime = time.time() - start_time

    start_time = time.time()
    pos_decisionScore = clf.predict(L_train)
    neg_decisionScore = clf.predict(L_test)
    testTime = time.time() - start_time

    return [pos_decisionScore, neg_decisionScore,trainTime,testTime]
