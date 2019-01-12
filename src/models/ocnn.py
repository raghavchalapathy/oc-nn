# Import libraries for data wrangling, preprocessing and visualization
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin
import csv

class OCNN(BaseEstimator, ClassifierMixin):  
    """An example of classifier"""
    from sklearn.preprocessing import StandardScaler

    results = "./sanity_results/"
    decision_scorePath = "./scores/"
    df_usps_scores  = {}
    activations = ["Linear","Sigmoid"]
    methods = ["Linear","RBF"]
    path = "./scores/"
    
   
    nu = 0.1
    scaler = StandardScaler()
    h_size = 64
    

    def __init__(self, intValue=0, stringParam="defaultValue", otherParam=None):
        """
        Called when initializing the classifier
        """
        self.intValue = intValue
        self.stringParam = stringParam

        # THIS IS WRONG! Parameters should have same name as attributes
        self.differentParam = otherParam 
               
        self.directory = "../models/supervisedBC/"
        self.ocnnSavedModelPath = "../models/ocnn/"
        self.ocsvmSavedModelPath = "../models/ocsvm/"
  
    def write_Scores2Csv(self,train, trainscore, test, testscore,filename):

            data = np.concatenate((train, test))
            score= np.concatenate((trainscore,testscore))
            data = data.tolist()
            score = score.tolist()
            with open(filename, 'a') as myfile:
                wr = csv.writer(myfile)
                wr.writerow(("x", "score"))
            for row in range(0,len(data)):
                with open(filename,
                        'a') as myfile:
                    wr = csv.writer(myfile)

                    wr.writerow((" ".join(str(x) for x in data[row]), " ".join(str(x) for x in score[row])))
    def write_decisionScores2Csv(self,path, filename, positiveScores, negativeScores):

            newfilePath = path+filename
            print("Writing file to ", path+filename)
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



    def train_OCNN_Classifier(self,X_train,nu,activation,epochs):

        RANDOM_SEED = 42
        tf.reset_default_graph()
        train_X = X_train
        tf.set_random_seed(RANDOM_SEED)
        outfile = self.ocnnSavedModelPath
        oCSVMweights = self.ocsvmSavedModelPath
        import time

        # Layer's sizes
        x_size = train_X.shape[1]   # Number of input nodes: 4 features and 1 bias
        h_size = self.h_size             # Number of hidden nodes
        y_size = 1   # Number of outcomes (3 iris flowers)
        D = x_size
        K = h_size
        theta = np.random.normal(0, 1, K + K*D + 1)
        rvalue = np.random.normal(0,1,(len(train_X),y_size))
        g   = lambda x : (1/np.sqrt(h_size) )*tf.cos(x/0.02)

        def init_weights(shape):
            """ Weight initialization """
            weights = tf.random_normal(shape,mean=0, stddev=0.5)
            return tf.Variable(weights,trainable=False)

            def forwardprop(X, w_1, w_2):
                """
                Forward-propagation.
                IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
                """
                X = tf.cast(X, tf.float32)
                w_1 = tf.cast(w_1, tf.float32)
                w_2 = tf.cast(w_2, tf.float32)
                h    = tf.nn.sigmoid(tf.matmul(X, w_1))  # The \sigma function
                yhat = tf.matmul(h, w_2)  # The \varphi function
                return yhat
         
        def nnScore(X, w, V, g,bias1,bias2):
            X = tf.cast(X, tf.float32)
            w = tf.cast(w, tf.float32)
            V = tf.cast(V, tf.float32)
            y_hat =tf.matmul(g((tf.matmul(X, w)+bias1)), V) +bias2

            return y_hat
        
        def relu(x):
            y = x
            y[y < 0] = 0
            return y
        
        # For testing the algorithm
        def compute_LossValue(X, nu, w1, w2, g, r,bias1,bias2):
            w = w1
            V = w2

            X = tf.cast(X, tf.float32)
            w = tf.cast(w1, tf.float32)
            V = tf.cast(w2, tf.float32)
            term1 = 0.5 * tf.reduce_sum(tf.square(w))
            term2 = 0.5 * tf.reduce_sum(tf.square(V))


            
            term3 = 1 / nu * tf.reduce_mean(tf.nn.relu(r - nnScore(X, w, V, g,bias1,bias2)))
            term4 = -r
            
            y_hat = nnScore(X, w, V, g,bias1,bias2)
            
            totalCost = term1 + term2 + term3 + term4
                
            loss=   [term1,term2,term3,term4,totalCost,y_hat]
            
            return loss
            
            
        def ocnn_obj(theta, X, nu, w1, w2, g,r,bias1,bias2):

            w = w1
            V = w2
     
            X = tf.cast(X, tf.float32)
            w = tf.cast(w1, tf.float32)
            V = tf.cast(w2, tf.float32)


            term1 = 0.5  * tf.reduce_sum(w**2)
            term2 = 0.5  * tf.reduce_sum(V**2)
            term3 = 1/nu * tf.reduce_mean(tf.nn.relu(r - nnScore(X, w, V, g,bias1,bias2)))
            term4 = -r

            return term1 + term2 + term3 + term4





            # Symbols
        X = tf.placeholder("float32", shape=[None, x_size])

        r = tf.get_variable("r", dtype=tf.float32,shape=())

        # Weight initializations
        w_1 = init_weights((x_size, h_size))
        w_2 = init_weights((h_size, y_size))
        # weights = tf.random_normal((h_size, y_size),mean=0, stddev=0.1)
           
        # ocsvm_wt = np.load(oCSVMweights+"ocsvm_wt.npy")
        # w_2 =tf.get_variable("tf_var_initialized_ocsvm",
        #                         initializer=ocsvm_wt)
            
        bias1 = tf.Variable(initial_value=[[1.0]], dtype=tf.float32,trainable=False)
        bias2 = tf.Variable(initial_value=[[0.0]], dtype=tf.float32,trainable=False)


        cost    = ocnn_obj(theta, X, nu, w_1, w_2, g,r,bias1,bias2)
        #updates = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
        updates = tf.train.AdamOptimizer(4.7 * 1e-1).minimize(cost)

        # Run SGD
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        rvalue = 0.1
        start_time = time.time()
        print("Training OC-NN started for epochs: ",epochs)
        for epoch in range(epochs):
                    # Train with each example
            trainX = OCNN.image_to_feature_vector(trainX, 28, 28)
            sess.run(updates, feed_dict={X: train_X})

                    
            with sess.as_default():
                svalue = nnScore(train_X, w_1, w_2, g,bias1,bias2)  
                rval = svalue.eval()
                rvalue = np.percentile(rval,q=100*nu)
                            

                costvalue = compute_LossValue(train_X, nu, w_1, w_2, g, rvalue,bias1,bias2)
                term1 = costvalue[0].eval()
                term2 = costvalue[1].eval()
                term3 = costvalue[2].eval()
                term4 = costvalue[3]
                term5 = costvalue[4].eval()
                yval = costvalue[5].eval()
                # print ("================================")
                print ("Epoch = %d, r = %f, Cost = %f"
                        % (epoch + 1,rvalue,np.mean(term5)))
                # print ("================================")
                # print ("Total Cost: ",np.mean(term5))
                        
                import time
                trainTime = time.time() - start_time
                # print("Training Time taken,",trainTime)
          
            
            
                with sess.as_default():
                    np_w_1= w_1.eval()
                    np_w_2= w_2.eval()
                    np_bias1= bias1.eval()
                    np_bias2= bias2.eval()
            
                rstar =rvalue

            # save the w_1 and bias1 to numpy array
            # print("Saving the trained Model weights ... @",outfile)
            # print("The optimized value of r found is",rstar)
            np.save(outfile+"w_1", np_w_1)
            np.save(outfile+"w_2", np_w_2)
            np.save(outfile+"bias1",np_bias1)
            np.save(outfile+"bias2",np_bias2)

    def fit(self,X,nu,activation,epochs):
  
        print("Training the OCNN classifier.....")
        self.train_OCNN_Classifier(X,nu,activation,epochs)

        return   
    
    def compute_au_roc(self,y_true, df_score):
        y_scores_pos = df_score[0]
        y_scores_neg = df_score[1]
        y_score = np.concatenate((y_scores_pos, y_scores_neg))
        from sklearn.metrics import roc_auc_score
        roc_score = roc_auc_score(y_true, y_score)
 
        return roc_score
    
    def decision_function(self,X, w_1, w_2, g,bias1,bias2):   
        score =np.matmul(g((np.matmul(X, w_1)+bias1)), w_2) +bias2
        return score

    def score(self,Xtest_Pos,Xtest_Neg):
        
        ## Load the saved model and compute the decision score
        model_weights = self.ocnnSavedModelPath
        w_1 = np.load(model_weights+"/w_1.npy")
        w_2 = np.load(model_weights+"/w_2.npy")
        bias1 = np.load(model_weights+"/bias1.npy")
        bias2 = np.load(model_weights+"/bias2.npy")
        
        
        g   = lambda x : (1/np.sqrt(self.h_size) )*np.cos(x/0.02)

        decisionScore_POS= self.decision_function(Xtest_Pos, w_1, w_2, g,bias1,bias2)
        decisionScore_Neg = self.decision_function(Xtest_Neg, w_1, w_2, g,bias1,bias2)
   
        df_score = [decisionScore_POS, decisionScore_Neg]
        
        ## y_true
        y_true_pos = np.ones(Xtest_Pos.shape[0])
        y_true_neg = np.zeros(Xtest_Neg.shape[0])
        y_true = np.concatenate((y_true_pos, y_true_neg))

        plt.hist(decisionScore_POS, bins = 25, label = 'Normal')
        plt.hist(decisionScore_Neg, bins = 25, label = 'Anomaly')
        plt.legend(loc = 'upper right')
        plt.title('OC-NN Normalised Decision Score')

        result = self.compute_au_roc(y_true,df_score)
        return result
    
    def predict(self, X, y=None):
        # counts number of values bigger than mean
        print(" predict  function is not implemented for OCNN")
        return



