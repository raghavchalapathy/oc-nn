from img_to_vec import Img2Vec
from PIL import Image
import scipy.io
w1 = "/Users/raghav/Documents/Uni/oc-nn/models/X.mat"
w1 = scipy.io.loadmat(w1)
print w1.keys()
print w1['X'].shape

w2 = "/Users/raghav/Documents/Uni/oc-nn/models/X_test.mat"
w2 = scipy.io.loadmat(w2)
print w2.keys()
print w2['X_test'].shape
import numpy as np

X1 = w1['X']
Y1 = np.ones(len(X1))

X2 = w2['X_test']
Y2 = np.zeros(len(X2))




from sklearn.manifold import TSNE


import numpy as np



# For speed of computation, only run on a subset



# perform t-SNE embedding
tsne =  TSNE(n_components=2, random_state=0)
print "tsne.fit_transform(X1)"
X1_2d = tsne.fit_transform(X1)
print "tsne.fit_transform(X2)"
X2_2d = tsne.fit_transform(X2)

vis_data= X1_2d[:]
y_data = Y1[:]
# plot the result
vis_x = vis_data[:, 0]
vis_y = vis_data[:, 1]


vis_data2= X2_2d[:]
y_data2 = Y2[:]
# plot the result
vis_x2 = vis_data2[:, 0]
vis_y2 = vis_data2[:, 1]

import matplotlib.pyplot as plt
print "PLotting the graph......"
plt.scatter(vis_x, vis_y, c=y_data, cmap=plt.cm.get_cmap("jet", 10))
plt.scatter(vis_x2, vis_y2, c=y_data2, cmap=plt.cm.get_cmap("Oranges_r", 10))
# plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)
plt.show()




# y = [0,1]
# target_ids = range(len(y))
#
# from matplotlib import pyplot as plt
# plt.figure(figsize=(6, 5))
# colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
# for i, c, label in zip(target_ids, colors, y):
#     plt.scatter(X_2d[ 0], X_2d[ 1], c=c, label=label)
# plt.legend()
# plt.show()



import matplotlib.pyplot as plt
import math
import numpy as np
def plotNNFilter(units):
    filters = 4
    plt.figure(1, figsize=(20,20))
    n_columns = 4
    n_rows = math.ceil(filters / n_columns) + 1
    print n_rows,n_columns
    print units[0]

    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        # plt.title('Cat ' + str(i))
        plt.imshow(np.reshape(units[i],(4,8)), interpolation="nearest")
        plt.axis('off')
        plt.savefig("/Users/raghav/Documents/Uni/oc-nn/models/test.png", bbox_inches='tight')

plotNNFilter(w1['X'])
plotNNFilter(w2['X_test'])
plt.show()
# plotNNFilter(w2['data'])

# Read in an image
# img = Image.open('/Users/raghav/Documents/Uni/oc-nn/data/cifar-10_data/train/dogs/dog.0.jpg')
# # Get a vector from img2vec
# vec = img2vec.get_vec(img)
#
# print vec.shape
# print vec





# from keras.applications.resnet50 import ResNet50
# from keras.preprocessing import image
# from keras.applications.resnet50 import preprocess_input, decode_predictions
# import numpy as np
#
# model = ResNet50(weights='imagenet')
#
# # img_path = '/Users/raghav/Documents/Uni/oc-nn/data/cifar-10_data/train/dogs/dog.0.jpg'
# img_path ="/Users/raghav/Documents/Uni/oc-nn/data/cifar-10_data/test/cat.1.jpg"
# img = image.load_img(img_path, target_size=(224, 224))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)
# print model.summary()
# print x.shape
# preds = model.predict(x)
# # decode the results into a list of tuples (class, description, probability)
# # (one such list for each sample in the batch)
# print('Predicted:', decode_predictions(preds, top=3)[0])


