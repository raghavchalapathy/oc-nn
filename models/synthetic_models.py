from  sklearn_OCSVM_model import sklearn_OCSVM_linear,sklearn_OCSVM_rbf
from  OneClass_NN_model import One_Class_NN_explicit_linear,One_Class_NN_explicit_sigmoid
from  sklearn_OCSVM_explicit_model import sklearn_OCSVM_explicit_linear,sklearn_OCSVM_explicit_sigmoid
from tf_OneClass_NN_model import tf_OneClass_NN_linear,tf_OneClass_NN_sigmoid,tf_OneClass_NN_Relu
from  sklearn_OCSVM_rpca import sklearn__RPCA_OCSVM, sklearn_RPCA_OCSVM_rbf,sklearn_RPCA_OCSVM_Linear
from sklearn_isolation_forest import sklearn_IsolationForest
from CAE_OCSVM_models import CAE_OCSVM_Linear,CAE_OCSVM_RBF
# from DBN2_OCSVM_models import DBN2_OCSVM_Linear,DBN2_OCSVM_RBF
# from RDA_models import RDA
# from RCAE_models import RCAE

from tflearn_OneClass_NN_model import tflearn_OneClass_NN_linear,tflearn_OneClass_NN_Sigmoid
dataPath = "/Users/raghav/Documents/Uni/oc-nn/data/"
# Declare a dictionary to store the results
df_usps_scores  = {}
import numpy as np
def prepare_usps_mlfetch():

    import tempfile
    import pickle
    # print "importing usps from pickle file ....."

    with open(dataPath + 'usps_data.pkl', "rb") as fp:
        loaded_data1 = pickle.load(fp)

    # test_data_home = tempfile.mkdtemp()
    # from sklearn.datasets.mldata import fetch_mldata
    # usps = fetch_mldata('usps', data_home=test_data_home)
    # print usps.target.shape
    # print type(usps.target)
    labels = loaded_data1['target']
    data = loaded_data1['data']
    # print "******",labels

    k_ones = np.where(labels == 2)
    label_ones = labels[k_ones]
    data_ones = data[k_ones]

    k_sevens = np.where(labels == 8)
    label_sevens = labels[k_sevens]
    data_sevens = data[k_sevens]
    #
    # print "data_sevens:",data_sevens.shape
    # print "label_sevens:",label_sevens.shape
    # print "data_ones:",data_ones.shape
    # print "label_ones:",label_ones.shape
    #
    data_ones = data_ones[:220]
    label_ones = label_ones[:220]
    data_sevens = data_sevens[:11]
    label_sevens = label_sevens[:11]

    data = np.concatenate((data_ones, data_sevens), axis=0)
    label = np.concatenate((label_ones, label_sevens), axis=0)
    label[0:220] = 1
    label[220:231] = -1
    # print "1-s",data[0]
    # print label
    # print "7-s",data[230]
    # print label
    # print "data:",data.shape
    # print "label:",label.shape

    # import matplotlib.pyplot as plt
    # plt.hist(label,bins=5)
    # plt.title("Count of  USPS Normal(1's) and Anomalous datapoints(7's) in training set")
    # plt.show()

    return [data, label]
import csv
from itertools import izip_longest
decision_scorePath = "/Users/raghav/Documents/Uni/oc-nn/Decision_Scores/usps/"
def write_decisionScores2Csv(path, filename, positiveScores, negativeScores):

    newfilePath = path+filename
    print "Writing file to ", path+filename
    poslist = positiveScores.tolist()
    neglist = negativeScores.tolist()

    # rows = zip(poslist, neglist)
    d = [poslist, neglist]
    export_data = izip_longest(*d, fillvalue='')
    with open(newfilePath, 'w') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(("Normal", "Anomaly"))
        wr.writerows(export_data)
    myfile.close()

    return



def func_getDecision_Scores_synthetic(dataset,data_train,data_test,labels_train,autoencoder="no"):



	#sklearn_OCSVM
	nu= 0.05
	result = sklearn_OCSVM_linear(data_train,data_test,nu)
	df_usps_scores["sklearn-OCSVM-Linear-Train"] = result[0]
	df_usps_scores["sklearn-OCSVM-Linear-Test"] =  result[1]
	print ("Finished sklearn_OCSVM_linear")

	result = sklearn_OCSVM_rbf(data_train,data_test,nu)
	df_usps_scores["sklearn-OCSVM-RBF-Train"] = result[0]
	df_usps_scores["sklearn-OCSVM-RBF-Test"] = result[1]
	print ("Finished sklearn_OCSVM_RBF")

	# sklearn _OCSVM_explicit
	result = sklearn_OCSVM_explicit_linear(data_train,data_test)
	df_usps_scores["sklearn-OCSVM-explicit-Linear-Train"] = result[0]
	df_usps_scores["sklearn-OCSVM-explicit-Linear-Test"] =  result[1]

	result = sklearn_OCSVM_explicit_sigmoid(data_train,data_test)
	df_usps_scores["sklearn-OCSVM-explicit-Sigmoid-Train"] = result[0]
	df_usps_scores["sklearn-OCSVM-explicit-Sigmoid-Test"] = result[1]

	print ("Finished sklearn _OCSVM_explicit")


	#One Class NN Explicit
	result = One_Class_NN_explicit_linear(data_train,data_test)
	df_usps_scores["One_Class_NN_explicit-Linear-Train"] = result[0]
	df_usps_scores["One_Class_NN_explicit-Linear-Test"] =  result[1]

	result = One_Class_NN_explicit_sigmoid(data_train,data_test)
	df_usps_scores["One_Class_NN_explicit-Sigmoid-Train"] = result[0]
	df_usps_scores["One_Class_NN_explicit-Sigmoid-Test"] = result[1]

	print ("Finished One Class NN Explicit")


	result = tf_OneClass_NN_linear(data_train,data_test)
	df_usps_scores["tf_OneClass_NN-Linear-Train"] = result[0]
	df_usps_scores["tf_OneClass_NN-Linear-Test"] =  result[1]
	print ("Finished tf_OneClass_NN_linear")

	result = tf_OneClass_NN_sigmoid(data_train,data_test)
	df_usps_scores["tf_OneClass_NN-Sigmoid-Train"] = result[0]
	df_usps_scores["tf_OneClass_NN-Sigmoid-Test"] = result[1]

	print ("Finished tf_OneClass_NN_sigmoid")

	result = tf_OneClass_NN_Relu(data_train,data_test)
	df_usps_scores["tf_OneClass_NN-Relu-Train"] = result[0]
	df_usps_scores["tf_OneClass_NN-Relu-Test"] = result[1]

	print ("Finished tf_OneClass_NN_sigmoid")

	result = sklearn__RPCA_OCSVM(data_train,data_test,nu)
	df_usps_scores["rpca_ocsvm-Train"] = result[0]
	df_usps_scores["rpca_ocsvm-Test"] =  result[1]
	print ("Finished rpca_ocsvm")

	result = sklearn_IsolationForest(data_train,data_test)
	df_usps_scores["isolation-forest-Train"] = result[0]
	df_usps_scores["isolation-forest-Test"] = result[1]
	print ("Finished isolation-forest")


	result = CAE_OCSVM_Linear(data_train,data_test,nu)
	df_usps_scores["cae_ocsvm-linear-Train"] = result[0]
	df_usps_scores["cae_ocsvm-linear-Test"] =  result[1]
	print ("Finished cae_ocsvm-linear")

	result = CAE_OCSVM_RBF(data_train,data_test,nu)
	df_usps_scores["cae_ocsvm-rbf-Train"] = result[0]
	df_usps_scores["cae_ocsvm-rbf-Test"] = result[1]
	print ("Finished cae_ocsvm-sigmoid")


	result = CAE_OCSVM_Linear(data_train,data_test,nu)
	df_usps_scores["ae_svdd-linear-Train"] = result[0]
	df_usps_scores["ae_svdd-linear-Test"] =  result[1]
	print ("Finished ae_ocsvm-linear")

	result = CAE_OCSVM_RBF(data_train,data_test,nu)
	df_usps_scores["ae_svdd-rbf-Train"] = result[0]
	df_usps_scores["ae_svdd-rbf-Test"] = result[1]
	print ("Finished ae_ocsvm-sigmoid")


	# result = DBN2_OCSVM_Linear(data_train,data_test,dataset)
	# df_usps_scores["cae_ocsvm-linear-Train"] = result[0]
	# df_usps_scores["cae_ocsvm-linear-Test"] =  result[1]
	# print ("Finished cae_ocsvm-linear")
    #
	# result = DBN2_OCSVM_RBF(data_train,data_test,dataset)
	# df_usps_scores["cae_ocsvm-sigmoid-Train"] = result[0]
	# df_usps_scores["cae_ocsvm-sigmoid-Test"] = result[1]
	# print ("Finished cae_ocsvm-sigmoid")
    #
    #
	# result = RCAE(data_train,data_test,dataset)
	# df_usps_scores["rcae-Train"] = result[0]
	# df_usps_scores["rcae-Test"] =  result[1]
	# print ("Finished rcae")
    #
	# result = RDA(data_train,data_test,dataset)
	# df_usps_scores["RDA-Train"] = result[0]
	# df_usps_scores["RDA-Test"] = result[1]
	# print ("Finished RDA")



	return df_usps_scores
