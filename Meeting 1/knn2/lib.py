from skimage import exposure
import numpy as np
import imutils
import cv2
from sklearn import datasets


def dataset_names():
    return ["mnist_small", "mnist_full"]


def get_dataset(name):
    if name == "mnist_small":
        return datasets.load_digits()
    elif name == "mnist_full":
        return datasets.fetch_openml("mnist_784")
    else:
        raise ("We cannot find your dataset")


def proceed_and_show_image(model, testData, testLabels, amount=5):
    # loop over a few random digits
    for i in list(map(int, np.random.randint(0, high=len(testLabels), size=(amount,)))):
        # grab the image and classify it
        _image = testData[i]
        _prediction = model.predict(_image.reshape(1, -1))[0]

        # convert the image for a 64-dim array to an 8 x 8 image compatible with OpenCV,
        # then resize it to 32 x 32 pixels so we can see it better
        _image = _image.reshape((28, 28)).astype("uint8")
        _image = exposure.rescale_intensity(_image, out_range=(0, 255))
        _image = imutils.resize(_image, width=56, inter=cv2.INTER_CUBIC)

        # show the prediction
        print("I think that digit is: {}".format(_prediction))
        cv2.imshow("Image", _image)
        cv2.waitKey(0)


def get_dataset_cache(file):
    # load the MNIST digits dataset
    import pickle, os
    if os.path.exists("./" + file + ".pickle"):
        with open("./" + file + ".pickle", "rb") as file_input:
            mnist = pickle.load(file_input)
        print("DB Loaded from cache")
    else:
        mnist = datasets.fetch_openml(file, cache=True)
        with open(file + ".pickle", "wb") as file_output:
            accuracies = pickle.dump(mnist, file_output)
        print("DB Loaded and Cached")
