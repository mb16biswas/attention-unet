import cv2
from sklearn.preprocessing import LabelEncoder
import numpy as np
import tensorflow as tf
from keras.utils.np_utils import to_categorical


def process_y(files,dsize = (256,256),cat = False , n_classes = 5 ):


    masks = []
    for f in files:
        img  = cv2.imread(f,0)
        res = cv2.resize(img, dsize= dsize, interpolation = cv2.INTER_NEAREST)
        masks.append(res)
    masks = np.array(masks)
    labelencoder = LabelEncoder()
    n, h, w = masks.shape
    train_masks_reshaped = masks.reshape(-1,1)
    train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
    train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)
    train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)

    if(cat):
        train_masks_cat = to_categorical(train_masks_input, num_classes=n_classes)
        y_train_cat = train_masks_cat.reshape((train_masks_input.shape[0], train_masks_input.shape[1], train_masks_input.shape[2], n_classes))
        return y_train_cat

    else:
        return train_masks_input




def process_y2(files,dsize = (256,256) , cat = False , n_classes = 5 ):

    masks = []
    Y = []
    for f in files:
        img  = cv2.imread(f,0)
        res = cv2.resize(img, dsize= dsize, interpolation = cv2.INTER_NEAREST)
        masks.append(tf.convert_to_tensor(res))
    masks = np.array(masks)
    labelencoder = LabelEncoder()
    n, h, w = masks.shape
    train_masks_reshaped = masks.reshape(-1,1)
    train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
    train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)
    if(cat == False):

        # train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)
        for mask in  train_masks_encoded_original_shape:
            m = tf.expand_dims(tf.convert_to_tensor(mask), axis=-1)
            m = tf.cast(m, tf.float32)
            Y.append(m)
        return Y
    else:
        train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)
        train_masks_cat = to_categorical(train_masks_input, num_classes=n_classes)
        y_train_cat = train_masks_cat.reshape((train_masks_input.shape[0], train_masks_input.shape[1], train_masks_input.shape[2], n_classes))
        for mask in y_train_cat:
             m = tf.cast(tf.convert_to_tensor(mask), tf.float32)
             Y.append(m)
        return Y