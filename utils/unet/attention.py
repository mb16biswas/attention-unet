import tensorflow as tf

class ChannelAttention(tf.keras.layers.Layer):

    def __init__(self, filters, ratio = 8 ):
        super(ChannelAttention, self).__init__()
        self.filters = filters
        self.ratio = ratio

        def build(self, input_shape):

            self.conv=tf.keras.layers.Conv1D(filters=32,kernel_size=1)
            self.LSTM = tf.keras.layers.LSTM(64,return_sequences=True)

        def call(self, inputs):

            avg_pool = tf.keras.layers.GlobalAveragePooling2D()(inputs)
            max_pool = tf.keras.layers.GlobalMaxPooling2D()(inputs)
            max_pool = tf.keras.layers.Reshape((1,1,filters))(max_pool)

            layer = tf.stack([avg_pool,max_pool],axis=2)
            layer = self.conv(layer)
            layer= self.LSTM(layer)
            attention = tf.keras.layers.GlobalAveragePooling1D()(layer)

            attention = tf.keras.layers.Activation('sigmoid')(attention)

            return tf.keras.layers.Multiply()([inputs, attention])





class SpatialAttention(tf.keras.layers.Layer):

    def __init__(self, kernel_size = 7 ):

        super(SpatialAttention, self).__init__()

        def build(self, input_shape):
            self.conv1 = tf.keras.layers.Conv2D(1, kernel_size = kernel_size, use_bias=True)

            self.avg_deep=  tf.keras.Sequential([tf.keras.layers.Conv2D(16,kernel_size=1, use_bias =False , activation = "relu"),
                                            tf.keras.layers.Conv2D(1,kernel_size=1, use_bias =False , activation = "softmax")]
                                            )

            self.max_deep=  tf.keras.Sequential([tf.nn.conv2d(16,kernel_size=1, use_bias =False , activation = "relu"),
                                            tf.keras.layers.Conv2D(1,kernel_size=1, use_bias =False , activation = "softmax")]
                                            )


        def call(self, x):

            avg_out = tf.math.reduce_mean(x, dim=1, keepdim=True)
            max_out, _ = tf.math.reduce_max(x, dim=1, keepdim=True)

            avg = self.avg_deep(avg_out)
            max = self.max_deep(max_out)

            avg_out = avg*avg_out
            max_out = max*max_out

            x = tf.concat([avg_out, max_out], dim=1)
            x = self.conv1(x)

            return tf.keras.layers.Activation('sigmoid')(x)