import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def show_predictions(X,y, model ):

    pred_mask = model.predict(X)

    display([X, y, create_mask(pred_mask)])



def display(display_list):

    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):

        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        if(i == 0):

            plt.imshow(tf.keras.utils.array_to_img(display_list[i][0]))

        else:

            plt.imshow(tf.keras.utils.array_to_img(display_list[i]))

        plt.axis('off')
        
    plt.show()



def pred(X,y,model,num):

  #returns the y preds and y accutual

    y_acc = []
    y_pred = []
    for i in range(0,num):
        y_acc.append(y[i].reshape(256,256,1))
        y_pred.append(create_mask(model.predict(X[i:i+1])) )

    y_acc = np.array(y_acc)
    y_pred = np.array(y_pred)

    return (y_acc,y_pred)