import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Concatenate, Input, UpSampling2D
from tensorflow.keras.models import Model
from tf.keras.losses import SparseCategoricalCrossentropy
from unet.attention import ChannelAttention, SpatialAttention

def conv_block(inputs, filters, pool=True,num = 0):

    dilation_rate=(1, 1)

    if num == 1 :
        dilation_rate=(4, 4)

    elif num == 2 :
        dilation_rate=(3, 3)

    elif num == 3 :
        dilation_rate = (2,2)

    elif num == 4 :
        dilation_rate=(1, 1)

    x = Conv2D(filters, 3, padding="same",dilation_rate = dilation_rate )(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)


    x = Conv2D(filters, 3, padding="same",dilation_rate = dilation_rate)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)


    if pool == True:

        p = MaxPool2D((2, 2))(x)
        return x, p

    else:

        return x



def build_unet(shape, num_classes = 5 ,activation = "softmax" ,loss = SparseCategoricalCrossentropy ,optimizer = tf.keras.optimizers.Adam ,  lr = 0.0001, compile = True , metrics = ["accuracy"]   ):

    print("--------config-------------------------")
    print("shape " , shape)
    print("num_classes " , num_classes)
    print("activation ",activation)
    print("optimizer ",optimizer)
    print("lr ",lr)
    print("compile ", compile)
    print("loss ", loss)
    print("metics" , metrics )
    print("----------------------------------------")

    inputs = Input(shape)

    """ Encoder """
    x1, p1 = conv_block(inputs, 16, pool=True,num =1)
    x1 = ChannelAttention(16)(x1)
    x1 = SpatialAttention()(x1)
    x2, p2 = conv_block(p1, 32, pool=True,num =2)
    x2 = ChannelAttention(32)(x2)
    x2 = SpatialAttention()(x2)
    x3, p3 = conv_block(p2, 48, pool=True,num =3)
    x3 = ChannelAttention(48)(x3)
    x3 = SpatialAttention()(x3)
    x4, p4 = conv_block(p3, 64, pool=True,num =4)
    x4 = ChannelAttention(64)(x4)
    x4 = SpatialAttention()(x4)
    x4 = tf.keras.layers.Dropout(0.2)(x4)

    """ Bridge """
    b1 = conv_block(p4, 128, pool=False)

    """ Decoder """
    u1 = UpSampling2D((2, 2), interpolation="bilinear")(b1)
    c1 = Concatenate()([u1, x4])
    x5 = conv_block(c1, 64, pool=False)
    # x5 = tf.keras.layers.Dropout(0.2)(x5)

    u2 = UpSampling2D((2, 2), interpolation="bilinear")(x5)
    c2 = Concatenate()([u2, x3])
    x6 = conv_block(c2, 48, pool=False)
    # x6 = tf.keras.layers.Dropout(0.2)(x6)

    u3 = UpSampling2D((2, 2), interpolation="bilinear")(x6)
    c3 = Concatenate()([u3, x2])
    x7 = conv_block(c3, 32, pool=False)

    u4 = UpSampling2D((2, 2), interpolation="bilinear")(x7)
    c4 = Concatenate()([u4, x1])
    x8 = conv_block(c4, 16, pool=False)


    """ Output layer """
    output = Conv2D(num_classes, 1, padding="same", activation= activation)(x8)

    model = Model(inputs, output)

    if compile:

        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=lr) ,loss = loss , metrics = metrics)

    return model
