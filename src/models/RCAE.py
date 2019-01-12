# import the necessary packages
import numpy as np
from src.data.preprocessing import learn_dictionary
from sklearn.metrics import average_precision_score,mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from skimage import io

from src.config import Configuration as Cfg

# RANDOM_SEED = 42
# np.random.seed(RANDOM_SEED)
import tensorflow as tf

sess = tf.Session()


from keras import backend as K

K.set_session(sess)

import numpy as np

from src.data.main import load_dataset

PROJECT_DIR = "/content/drive/My Drive/2019/testing/oc-nn/"

class RCAE_AD:
    ## Initialise static variables
    INPUT_DIM = 0
    HIDDEN_SIZE = 0
    DATASET = "mnist"
    mean_square_error_dict = {}
    RESULT_PATH = ""

    def __init__(self, dataset, inputdim, hiddenLayerSize, img_hgt, img_wdt,img_channel, modelSavePath, reportSavePath,
                 preTrainedWtPath, seed, intValue=0, stringParam="defaultValue",
                 otherParam=None):
        """
        Called when initializing the classifier
        """
        RCAE_AD.DATASET = dataset
        RCAE_AD.INPUT_DIM = inputdim
        RCAE_AD.HIDDEN_SIZE = hiddenLayerSize
        RCAE_AD.RESULT_PATH = reportSavePath
        RCAE_AD.RANDOM_SEED = seed

        print("RCAE.RESULT_PATH:",RCAE_AD.RESULT_PATH)
        self.intValue = intValue
        self.stringParam = stringParam

        # THIS IS WRONG! Parameters should have same name as attributes
        self.differentParam = otherParam

        self.directory = modelSavePath
        self.results = reportSavePath
        self.pretrainedWts = preTrainedWtPath
        self.model = ""

        self.IMG_HGT = img_hgt
        self.IMG_WDT = img_wdt
        self.channel = img_channel
        self.h_size = RCAE_AD.HIDDEN_SIZE
        global model
        self.r = 1.0
        self.kvar = 0.0
        self.pretrain= True
        self.dataset = dataset.lower()
        
        print("INFO: The dataset is ",self.dataset)
        # load dataset
        load_dataset(self, dataset.lower(), self.pretrain)

        # Depending on the dataset build respective autoencoder architecture
        if (dataset.lower() == "mnist"):
            from src.data.mnist import MNIST_DataLoader
            # Create the robust cae for the dataset passed
            self.nn_model = MNIST_DataLoader()
            self.mnist_savedModelPath= PROJECT_DIR+"/models/MNIST/RCAE/"


        # Depending on the dataset build respective autoencoder architecture
        if (dataset.lower() == "cifar10"):
            from src.data.cifar10 import CIFAR_10_DataLoader
            # Create the robust cae for the dataset passed
            self.nn_model = CIFAR_10_DataLoader()


        # Depending on the dataset build respective autoencoder architecture
        if (dataset.lower() == "gtsrb"):
            from src.data.GTSRB import GTSRB_DataLoader
            # Create the robust cae for the dataset passed
            self.nn_model = GTSRB_DataLoader()

    
    def save_model(self, model,lambdaval):

        ## save the model
        # serialize model to JSON
        if(RCAE_AD.DATASET == "mnist"):
            model_json = model.to_json()
            with open(self.mnist_savedModelPath +lambdaval+ "__DCAE_DIGIT__"+str(Cfg.mnist_normal)+"__model.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights(self.mnist_savedModelPath + lambdaval+"__DCAE_DIGIT__"+str(Cfg.mnist_normal)+"__model.h5")
            print("Saved model to disk....")


        return

    def computePred_Labels(self, X_test,decoded,poslabelBoundary,negBoundary,lamda):

        y_pred = np.ones(len(X_test))
        recon_error = {}
        for i in range(0, len(X_test)):
            recon_error.update({i: np.linalg.norm(X_test[i] - decoded[i])})

        best_sorted_keys = sorted(recon_error, key=recon_error.get, reverse=False)
        worst_sorted_keys = sorted(recon_error, key=recon_error.get, reverse=True)
        anomaly_index = worst_sorted_keys[0:negBoundary]
        print("[INFO:] The anomaly index are ",anomaly_index)
        
       
        worstreconstructed_Top200index = worst_sorted_keys[0:200]
        print("[INFO:] The worstreconstructed_Top200index index are ",worstreconstructed_Top200index)
        
        
        
        for key in anomaly_index:
            if(key >= poslabelBoundary):
                y_pred[key] = -1
        
        top_100_anomalies= []
        for i in worstreconstructed_Top200index:
            top_100_anomalies.append(X_test[i])

        top_100_anomalies = np.asarray(top_100_anomalies)

        top_100_anomalies = np.reshape(top_100_anomalies,(len(top_100_anomalies),28,28))
        
        result = self.tile_raster_images(top_100_anomalies, [28, 28], [10, 10])
        print("[INFO:] Saving Anomalies Found at ..",self.results)
        io.imsave(self.results  + str(lamda)+"_Top100_anomalies.png",result)



        return y_pred

    def load_data(self, data_loader=None, pretrain=False):

        self.data = data_loader()
        return
    
    def scale_to_unit_interval(self,ndar, eps=1e-8):
        """ Scales all values in the ndarray ndar to be between 0 and 1 """
        ndar = ndar.copy()
        ndar -= ndar.min()
        ndar *= 1.0 / (ndar.max() + eps)
        return ndar

    def tile_raster_images(self,X, img_shape, tile_shape, tile_spacing=(0, 0),
                           scale_rows_to_unit_interval=True,
                           output_pixel_vals=True):
        """
        Source : http://deeplearning.net/tutorial/utilities.html#how-to-plot
        Transform an array with one flattened image per row, into an array in
        which images are reshaped and layed out like tiles on a floor.
        """

        assert len(img_shape) == 2
        assert len(tile_shape) == 2
        assert len(tile_spacing) == 2

        # The expression below can be re-written in a more C style as
        # follows :
        #
        # out_shape = [0,0]
        # out_shape[0] = (img_shape[0] + tile_spacing[0]) * tile_shape[0] -
        #                tile_spacing[0]
        # out_shape[1] = (img_shape[1] + tile_spacing[1]) * tile_shape[1] -
        #                tile_spacing[1]
        out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                     in zip(img_shape, tile_shape, tile_spacing)]

        if isinstance(X, tuple):
            assert len(X) == 4
            # Create an output numpy ndarray to store the image
            if output_pixel_vals:
                out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype='uint8')
            else:
                out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype=X.dtype)

            # colors default to 0, alpha defaults to 1 (opaque)
            if output_pixel_vals:
                channel_defaults = [0, 0, 0, 255]
            else:
                channel_defaults = [0., 0., 0., 1.]

            for i in range(4):
                if X[i] is None:
                    # if channel is None, fill it with zeros of the correct
                    # dtype
                    out_array[:, :, i] = np.zeros(out_shape,
                                                  dtype='uint8' if output_pixel_vals else out_array.dtype
                                                  ) + channel_defaults[i]
                else:
                    # use a recurrent call to compute the channel and store it
                    # in the output
                    out_array[:, :, i] = self.tile_raster_images(X[i], img_shape, tile_shape,
                                                            tile_spacing, scale_rows_to_unit_interval,
                                                            output_pixel_vals)
            return out_array

        else:
            # if we are dealing with only one channel
            H, W = img_shape
            Hs, Ws = tile_spacing

            # generate a matrix to store the output
            out_array = np.zeros(out_shape, dtype='uint8' if output_pixel_vals else X.dtype)

            for tile_row in range(tile_shape[0]):
                for tile_col in range(tile_shape[1]):
                    if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                        if scale_rows_to_unit_interval:
                            # if we should scale values to be between 0 and 1
                            # do this by calling the `scale_to_unit_interval`
                            # function
                            this_img = self.scale_to_unit_interval(X[tile_row * tile_shape[1] + tile_col].reshape(img_shape))
                        else:
                            this_img = X[tile_row * tile_shape[1] + tile_col].reshape(img_shape)
                            # add the slice to the corresponding position in the
                            # output array
                        out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                        ] \
                            = this_img * (255 if output_pixel_vals else 1)
            return out_array

    def get_oneClass_testData(self):

        X_train = self.data._X_train
        y_train = self.data._y_train
        
        X_test = self.data._X_test
        y_test = self.data._y_test

        
        ## Combine the positive data
        trainXPos = X_train[np.where(y_train == 0)]
        trainYPos = np.zeros(len(trainXPos))
        
        testXPos = X_test[np.where(y_test == 0)]
        testYPos = np.zeros(len(testXPos))
        
        
        # Combine the negative data
        trainXNeg = X_train[np.where(y_train == 1)]
        trainYNeg = np.ones(len(trainXNeg))
        
        testXNeg = X_test[np.where(y_test == 1)]
        testYNeg = np.ones(len(testXNeg))

     
        X_testPOS = np.concatenate((trainXPos, testXPos))
        y_testPOS = np.concatenate((trainYPos, testYPos))
        
        X_testNEG = np.concatenate((trainXNeg, testXNeg))
        y_testNEG = np.concatenate((trainYNeg, testYNeg))
        
        # Just 0.01 points are the number of anomalies.
        
        # Just 0.01 points are the number of anomalies
        if(self.dataset == "mnist"):
            num_of_anomalies = int(0.01 * len(X_testPOS))
        elif(self.dataset == "cifar10"):
            num_of_anomalies = int(0.1 * len(X_testPOS))
            
        # num_of_anomalies = int(0.01 * len(X_testPOS))
        X_testNEG = X_testNEG[0:num_of_anomalies]
        y_testNEG = y_testNEG[0:num_of_anomalies]
        
        
        X_test = np.concatenate((X_testPOS, X_testNEG))
        y_test = np.concatenate((y_testPOS, y_testNEG))
        
        
        PosBoundary= len(X_testPOS)
        NegBoundary = len(X_testNEG)


        print("[INFO: ] Shape of One Class Input Data used in testing", X_test.shape)
        print("[INFO: ] Shape of (Positive) One Class Input Data used in testing", X_testPOS.shape)
        print("[INFO: ] Shape of (Negative) One Class Input Data used in testing", X_testNEG.shape)

        return [X_test, y_test]

    def get_oneClass_trainData(self):
        # X_train = np.concatenate((self.data._X_train, self.data._X_val))
        # y_train = np.concatenate((self.data._y_train, self.data._y_val))

        X_train = self.data._X_train
        y_train = self.data._y_train
        
        X_test = self.data._X_test
        y_test = self.data._y_test

        
        ## Combine the positive data
        trainXPos = X_train[np.where(y_train == 0)]
        trainYPos = np.zeros(len(trainXPos))
        
        testXPos = X_test[np.where(y_test == 0)]
        testYPos = np.zeros(len(testXPos))
        
        
        # Combine the negative data
        trainXNeg = X_train[np.where(y_train == 1)]
        trainYNeg = np.ones(len(trainXNeg))
        
        testXNeg = X_test[np.where(y_test == 1)]
        testYNeg = np.ones(len(testXNeg))

     
        X_trainPOS = np.concatenate((trainXPos, testXPos))
        y_trainPOS = np.concatenate((trainYPos, testYPos))
        
        X_trainNEG = np.concatenate((trainXNeg, testXNeg))
        y_trainNEG = np.concatenate((trainYNeg, testYNeg))
        
        # Just 0.01 points are the number of anomalies.
        if(self.dataset == "mnist"):
            num_of_anomalies = int(0.01 * len(X_trainPOS))
        elif(self.dataset == "cifar10"):
            num_of_anomalies = int(0.1 * len(X_trainPOS))
        elif(self.dataset == "gtsrb"):
            num_of_anomalies = int(0.1 * len(X_trainPOS))
        
        X_trainNEG = X_trainNEG[0:num_of_anomalies]
        y_trainNEG = y_trainNEG[0:num_of_anomalies]
        
        
        X_train = np.concatenate((X_trainPOS, X_trainNEG))
        y_train = np.concatenate((y_trainPOS, y_trainNEG))
        
    
        print("[INFO: ] Shape of One Class Input Data used in training", X_train.shape)
        print("[INFO: ] Shape of (Positive) One Class Input Data used in training", X_trainPOS.shape)
        print("[INFO: ] Shape of (Negative) One Class Input Data used in training", X_trainNEG.shape)

        return [X_train,y_train]


    def tile_raster_visualise_anamolies_detected(self,testX, worst_top10_keys, lamda, nrows=10, ncols=10):
        #
        # print("[INFO:] The shape of input data  ",testX.shape)
        # print("[INFO:] The shape of decoded  data  ", decoded.shape)


        side = self.IMG_HGT
        channel = self.channel
        # side = 28
        # channel = 1
        # Display the decoded Original, noisy, reconstructed images


        img = np.ndarray(shape=(side * nrows, side * ncols, channel))
        print("img shape:", img.shape)

        worst_top10_keys = list(worst_top10_keys)

        # Display the decoded Original, noisy, reconstructed images for worst
        img = np.ndarray(shape=(side * nrows, side * ncols, channel))
        for i in range(ncols):
            row = i // ncols * nrows
            col = i % ncols
            img[side * row:side * (row + 1), side * col:side * (col + 1), :] = testX[worst_top10_keys[i]]
            img[side * (row + 1):side * (row + 2), side * col:side * (col + 1), :] = testX[worst_top10_keys[i+2*ncols]]
            img[side * (row + 2):side * (row + 3), side * col:side * (col + 1), :] = testX[worst_top10_keys[i+3*ncols]]
            img[side * (row + 3):side * (row + 4), side * col:side * (col + 1), :] = testX[worst_top10_keys[i+4*ncols]]
            img[side * (row + 4):side * (row + 5), side * col:side * (col + 1), :] = testX[worst_top10_keys[i+5*ncols]]
            img[side * (row + 5):side * (row + 6), side * col:side * (col + 1), :] = testX[worst_top10_keys[i+6*ncols]]
            img[side * (row + 6):side * (row + 7), side * col:side * (col + 1), :] = testX[worst_top10_keys[i+7*ncols]]
            img[side * (row + 7):side * (row + 8), side * col:side * (col + 1), :] = testX[worst_top10_keys[i+8*ncols]]
            img[side * (row + 8):side * (row + 9), side * col:side * (col + 1), :] = testX[worst_top10_keys[i+9*ncols]]
            img[side * (row + 9):side * (row + 10), side * col:side * (col + 1), :] = testX[worst_top10_keys[i+10*ncols]]

        img *= 255
        img = img.astype(np.uint8)

        # # Save the image decoded
        # print('Saving Top-200'+ str(lamda) + "most anomalous digit: @")
        # io.imsave(self.results + str(lamda)  + 'Class_'+ str(Cfg.cifar10_normal) + "_Top100.png", img)
        
        if(self.dataset == "cifar10"):
             # Save the image decoded
            print('Saving Top-200'+ str(lamda) + "most anomalous digit: @")
            io.imsave(self.results + str(lamda)  + 'Class_'+ str(Cfg.cifar10_normal) + "_Top100.png", img)
        if(self.dataset == "mnist"):
             # Save the image decoded
            img = np.reshape(img, (side * nrows, side * ncols))
            print('Saving Top-200'+ str(lamda) + "most anomalous digit: @")
            io.imsave(self.results + str(lamda)  + 'Class_'+ str(Cfg.mnist_normal) + "_Top100.png", img)

        return
    

    def save_Most_Normal(self,X_test,scores):

        worst_sorted_keys= np.argsort(scores)
        most_anomalous_index = worst_sorted_keys[0:100]
        top_100_anomalies = []

        for i in most_anomalous_index:
            top_100_anomalies.append(X_test[i])

        top_100_anomalies = np.asarray(top_100_anomalies)
        
        print("[INFO:] The  top_100_anomalies",top_100_anomalies.shape)
        self.tile_raster_visualise_anamolies_detected(X_test,worst_sorted_keys,"most_normal")

        return 
    
     
   
    
    def save_Most_Anomalous(self,X_test,scores):

        worst_sorted_keys= np.argsort(-scores)
        most_anomalous_index = worst_sorted_keys[0:100]
        top_100_anomalies = []

        for i in most_anomalous_index:
            top_100_anomalies.append(X_test[i])

        top_100_anomalies = np.asarray(top_100_anomalies)
        
        print("[INFO:] The  top_100_anomalies",top_100_anomalies.shape)
        self.tile_raster_visualise_anamolies_detected(X_test,worst_sorted_keys,"most_anomalous")

        return 

    def fit_and_predict(self):

      
        
        if(self.dataset == "cifar10"):
            X_train = self.data._X_train
            y_train = self.data._y_train
            
            X_test = X_train
            y_test = y_train
            
        elif(self.dataset == "mnist"):
            X_train,y_train = self.get_oneClass_trainData()
            #testing data
            X_test,y_test = self.get_oneClass_trainData()
             
        
        elif (RCAE_AD.DATASET == "gtsrb"):

            X_train = np.concatenate((self.data._X_train, self.data._X_test))
            y_train = np.concatenate((self.data._y_train, self.data._y_test))

            # X_test = self.data._X_test
            # y_test = self.data._y_test

            trainXPos = X_train[np.where(y_train == 0)]
            trainYPos = np.zeros(len(trainXPos))
            trainXNeg = X_train[np.where(y_train == 1)]
            trainYNeg = 1 * np.ones(len(trainXNeg))

            PosBoundary = len(trainXPos)
            NegBoundary = len(trainXNeg)

            # print("[INFO:]  Length of Positive data", len(trainXPos))
            # print("[INFO:]  Length of Negative data", len(trainXNeg))

            X_train = np.concatenate((trainXPos, trainXNeg))
            y_train = np.concatenate((trainYPos, trainYNeg))
            
            X_test = X_train
            y_test = y_train
            
            # Make sure the axis dimensions are aligned for training convolutional autoencoders
            X_train = np.moveaxis(X_train, 1, 3)
            X_test = np.moveaxis(X_test, 1, 3)
           

            X_train = X_train/255.0
            X_test = X_test / 255.0

            print("[INFO:] X_train.shape", X_train.shape)
            print("[INFO:] y_train.shape", y_train.shape)

            print("[INFO:] X_test.shape", X_test.shape)
            print("[INFO:] y_test.shape", y_test.shape)
        

      
        # best_top10_keys = [1,2,3,4,5,6,7,8,9,10]
        # worst_top10_keys = [5001, 5002, 5003, 5004, 5005, 5006, 5007, 5008, 5009, 5010]
        # debug_visualise_anamolies_detected(X_test, X_test, X_test, X_test, best_top10_keys, worst_top10_keys, 0.0)
        # exit()
        # define lamda set
        lamda_set = [0.0, 0.5, 1.0,  100.0]
        # lamda_set = [0.0,  0.5]
        
        # lamda_set = [0.0]
        # lamda_set = [0.5]
        # mue = 0.0
        # TRIALS = 2
        TRIALS = 4
        ap = np.zeros((TRIALS,))
        auc = np.zeros((TRIALS,))
        prec = np.zeros((TRIALS,))
        # outer loop for lamda
        for l in range(0, len(lamda_set)):
            # Learn the N using softthresholding technique
            N = 0
            lamda = lamda_set[l]
            XTrue = X_train
            YTrue = y_train

            # Capture the structural Noise
            self.nn_model.compute_softhreshold(XTrue, N, lamda, XTrue)
            N = self.nn_model.Noise
            
         
            decoded = self.nn_model.cae.predict(XTrue)
             # rank the best and worst reconstructed images
            [best_top10_keys, worst_top10_keys] = self.nn_model.compute_best_worst_rank(XTrue, decoded)

            self.nn_model.visualise_anamolies_detected(XTrue, XTrue, decoded, N, best_top10_keys, worst_top10_keys, lamda)

            
            
            
            # compute MeanSqared error metric
            # self.nn_model.compute_mse(X_test, decoded, lamda)
            # print("[INFO:] The anomaly threshold computed is ", self.nn_model.anomaly_threshold)

            
            # decoded 
            decoded = self.nn_model.cae.predict(X_test)
            XPred = decoded
            
            if(self.dataset == "mnist"):
                decoded = np.reshape(decoded, (len(decoded), 784))
                X_test_for_roc = np.reshape(X_test, (len(X_test), 784))
            
            elif(self.dataset == "cifar10"):
                decoded = np.reshape(decoded, (len(decoded), 3072))
                X_test_for_roc = np.reshape(X_test, (len(X_test), 3072))
            
            elif(self.dataset == "gtsrb"):
                decoded = np.reshape(decoded, (len(decoded), 3072))
                X_test_for_roc = np.reshape(X_test, (len(X_test), 3072))
                
            recErr = ((decoded - X_test_for_roc) ** 2).sum(axis = 1)
            
            self.save_Most_Anomalous(X_test,recErr)
            self.save_Most_Normal(X_test,recErr)
            
            auc[l] = roc_auc_score(y_test, recErr)
            import pandas as pd 
            df = pd.DataFrame(recErr)
            df.to_csv(self.results  +"recErr.csv")
            
            print("=====================")
            print("AUROC", lamda, auc[l])
            auc_roc = auc[l]
            print("=======================")
            
            self.save_model(self.nn_model.cae,str(lamda))


        return auc_roc




def debug_visualise_anamolies_detected(testX, noisytestX, decoded, N, best_top10_keys, worst_top10_keys, lamda):

        #
        # print("[INFO:] The shape of input data  ",testX.shape)
        # print("[INFO:] The shape of decoded  data  ", decoded.shape)


        side =32
        channel = 3
        N = np.reshape(N, (len(N), 32, 32, 3))
        # Display the decoded Original, noisy, reconstructed images
        print("[INFO:] The shape of N  data  ", N.shape)

        img = np.ndarray(shape=(side * 4, side * 10, channel))
        print("img shape:", img.shape)


        best_top10_keys = list(best_top10_keys)
        worst_top10_keys = list(worst_top10_keys)

        for i in range(10):
            row = i // 10 * 4
            col = i % 10
            img[side * row:side * (row + 1), side * col:side * (col + 1), :] = testX[best_top10_keys[i]]
            img[side * (row + 1):side * (row + 2), side * col:side * (col + 1), :] = noisytestX[best_top10_keys[i]]
            img[side * (row + 2):side * (row + 3), side * col:side * (col + 1), :] = decoded[best_top10_keys[i]]
            img[side * (row + 3):side * (row + 4), side * col:side * (col + 1), :] = N[best_top10_keys[i]]

        # img *= 255
        img = img.astype(np.uint8)

        # Save the image decoded
        print("\nSaving results for best after being encoded and decoded: @")
        save_results = PROJECT_DIR+ "/figures/cifar10/RCAE/"
        print(save_results+"/best/")
        io.imsave(save_results + '/best/' + str(lamda) + '_RCAE.png', img)

        # Display the decoded Original, noisy, reconstructed images for worst
        img = np.ndarray(shape=(side * 4, side * 10, channel))
        for i in range(10):
            row = i // 10 * 4
            col = i % 10
            img[side * row:side * (row + 1), side * col:side * (col + 1), :] = testX[worst_top10_keys[i]]
            img[side * (row + 1):side * (row + 2), side * col:side * (col + 1), :] = noisytestX[worst_top10_keys[i]]
            img[side * (row + 2):side * (row + 3), side * col:side * (col + 1), :] = decoded[worst_top10_keys[i]]
            img[side * (row + 3):side * (row + 4), side * col:side * (col + 1), :] = N[worst_top10_keys[i]]

        # img *= 255
        img = img.astype(np.uint8)

        # Save the image decoded
        print("\nSaving results for worst after being encoded and decoded: @")
        print(save_results + '/worst/')
        io.imsave(save_results + '/worst/' + str(lamda) + '_RCAE.png', img)

        return

