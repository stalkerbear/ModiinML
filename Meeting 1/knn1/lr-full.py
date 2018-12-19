# import the necessary packages
# from __future__ import print_function
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import datasets
from skimage import exposure
import numpy as np
import imutils
import cv2
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


# handle older versions of sklearn
if int((sklearn.__version__).split(".")[1]) < 18:
    from sklearn.cross_validation import train_test_split
# otherwise we're using at lease version 0.18
else:
    from sklearn.model_selection import train_test_split

# load the MNIST digits dataset
import pickle, os
file = "mnist_784"
if os.path.exists("./"+file+".pickle"):
	with open("./"+file+".pickle", "rb") as file_input:
		mnist = pickle.load(file_input)
	print("DB Loaded from cache")
else:
	mnist = datasets.fetch_openml(file, cache=True)
	with open(file+".pickle", "wb") as file_output:
		accuracies = pickle.dump(mnist, file_output)
	print("DB Loaded and Cached")


# take the MNIST data and construct the training and testing split, using 75% of the
# data for training and 25% for testing
(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(mnist.data),
                                                                  mnist.target, test_size=0.20, random_state=42)
print("First split finished")
# now, let's take 10% of the training data and use that for validation
(trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels,
                                                                test_size=0.1, random_state=84)
print("Second split finished")

# show the sizes of each data split
print("training data points: {}".format(len(trainLabels)))
print("validation data points: {}".format(len(valLabels)))
print("testing data points: {}".format(len(testLabels)))

# re-train our classifier using the best k value and predict the labels of the
# test data
model = LogisticRegression()
model.fit(trainData, trainLabels)


# import pickle
# with open("knn-full.model", "rb") as file_input:
# 	model = pickle.load(file_input)
# with open("knn-full.accuracies", "rb") as file_input:
#     accuracies = pickle.load(file_input)

predictions = model.predict(testData)
acc_sgd = accuracy_score(testLabels, predictions)
print("stochastic gradient descent accuracy: ",acc_sgd)
# show a final classification report demonstrating the accuracy of the classifier
# for each of the digits
print("EVALUATION ON TESTING DATA")
print(classification_report(testLabels, predictions))

# loop over a few random digits
for i in list(map(int, np.random.randint(0, high=len(testLabels), size=(15,)))):
	# grab the image and classify it
	image = testData[i]
	prediction = model.predict(image.reshape(1, -1))[0]

	# convert the image for a 64-dim array to an 8 x 8 image compatible with OpenCV,
	# then resize it to 32 x 32 pixels so we can see it better
	image = image.reshape((28, 28)).astype("uint8")
	image = exposure.rescale_intensity(image, out_range=(0, 255))
	image = imutils.resize(image, width=56, inter=cv2.INTER_CUBIC)

	# show the prediction
	print("I think that digit is: {}".format(prediction), "; real label :", testLabels[i])
	cv2.imshow("Image", image)
	cv2.waitKey(0)