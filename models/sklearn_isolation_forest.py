import numpy as np
from sklearn.ensemble import IsolationForest
rng = np.random.RandomState(42)
import time

def sklearn_IsolationForest(data_train,data_test):

    print "sklearn_IsolationForest====", data_train.shape
    clf = IsolationForest(max_samples=50, random_state=rng)
    start_time = time.time()
    clf.fit(data_train)
    trainTime = time.time() - start_time

    start_time = time.time()
    pos_decisionScore = clf.predict(data_train)
    neg_decisionScore = clf.predict(data_test)
    testTime = time.time() - start_time
    return [pos_decisionScore, neg_decisionScore,trainTime,testTime]

