import cv2
from sklearn.preprocessing import LabelEncoder
import numpy as np
import tensorflow as tf
from keras.utils.np_utils import to_categorical



def process_x(files,dsize = (256,256) , con = "Normal"):


    x = []
    for f in files:

        img =  cv2.imread(f,1)

        if con == "YUV":

            img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

        elif con == "HSV":

            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        elif con == "LUV":

            img = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)

        elif con == "LAB":

            img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        elif con == "HLS":

            img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

        elif con == "GRAY":

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = cv2.resize(img, dsize= dsize, interpolation = cv2.INTER_NEAREST)/255.0
        x.append(img)

    return np.array(x).astype("float32")



def process_x2(files,dsize = (256,256) , con = "Normal" ):

    x = []
    X = []

    for f in files:

        img =  cv2.imread(f,1)

        if con == "YUV":

            img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

        elif con == "HSV":

            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        elif con == "LUV":

            img = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)

        elif con == "LAB":

            img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        elif con == "HLS":

            img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

        elif con == "GRAY":

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = cv2.resize(img, dsize= dsize, interpolation = cv2.INTER_NEAREST)/255.0

        x.append(img)

    for i in x :

        image = tf.cast(tf.convert_to_tensor(i), tf.float32)
        X.append(image)

    return np.array(X)