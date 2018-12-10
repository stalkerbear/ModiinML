# import the necessary packages
# from __future__ import print_function
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn import datasets
from skimage import exposure
import numpy as np
import imutils
import cv2
import sklearn
import matplotlib.pyplot as plt


# handle older versions of sklearn
if int((sklearn.__version__).split(".")[1]) < 18:
    from sklearn.cross_validation import train_test_split
# otherwise we're using at lease version 0.18
else:
    from sklearn.model_selection import train_test_split

# load the MNIST digits dataset
mnist = datasets.load_digits()

# take the MNIST data and construct the training and testing split, using 75% of the
# data for training and 25% for testing
(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(mnist.data),
                                                                  mnist.target, test_size=0.20, random_state=42)

# now, let's take 10% of the training data and use that for validation
(trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels,
                                                                test_size=0.1, random_state=84)

# show the sizes of each data split
print("training data points: {}".format(len(trainLabels)))
print("validation data points: {}".format(len(valLabels)))
print("testing data points: {}".format(len(testLabels)))

# re-train our classifier using the best k value and predict the labels of the
# test data
model = RandomForestClassifier()
model.fit(trainData, trainLabels)

# import pickle
# with open("knn-small.model", "wb") as output:
# 	pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)
# with open("knn-small.accuracies", "wb") as output:
# 	pickle.dump(accuracies, output, pickle.HIGHEST_PROTOCOL)

predictions = model.predict(testData)
acc_rf = accuracy_score(testLabels, predictions)
print("Random Forest Accuracy :", acc_rf)
# show a final classification report demonstrat ing the accuracy of the classifier
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
	image = image.reshape((8, 8)).astype("uint8")
	image = exposure.rescale_intensity(image, out_range=(0, 255))
	image = imutils.resize(image, width=32, inter=cv2.INTER_CUBIC)

	# show the prediction
	print("I think that digit is: {}".format(prediction), "; while real label was : {}".format(testLabels[i]))
	cv2.imshow("Image", image)
	cv2.waitKey(0)

