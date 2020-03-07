import tensorflow as tf

# Leaky relu
def lrelu(inputs, leak=0.2, scope="lrelu"):
        with tf.variable_scope(scope):
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            return f1 * inputs + f2 * abs(inputs)


def convLayer(inputs, filters, kernel_size, strides, padding, activation, batch_normalization, training):
    conv = tf.layers.conv2d(inputs=inputs,
                            filters=filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding=padding)
    if activation == "relu" and batch_normalization == True:
        return tf.nn.relu(tf.layers.batch_normalization(conv, training=training))
    elif activation == "relu" and batch_normalization == False:
        return tf.nn.relu(conv)
    elif activation == "lrelu" and batch_normalization == True:
        return lrelu(tf.layers.batch_normalization(conv, training=training), 0.2)
    elif activation == "lrelu" and batch_normalization == False:
        return lrelu(conv, 0.2)
    else:
        return conv


def deConvLayer(inputs, filters, kernel_size, strides, padding, activation, batch_normalization, training):
    deConv = tf.layers.conv2d_transpose(inputs=inputs, 
                                    filters=filters, 
                                    kernel_size=kernel_size, 
                                    strides=strides, 
                                    padding=padding)
    if activation == "relu" and batch_normalization == True:
        return tf.nn.relu(tf.layers.batch_normalization(deConv, training=training))
    elif activation =="relu" and batch_normalization == False:
        return tf.nn.relu(deConv)
    elif activation =="lrelu" and batch_normalization == True:
        return lrelu(tf.layers.batch_normalization(deConv, training=training), 0.2)
    elif activation =="lrelu" and batch_normalization == False:
        return lrelu(deConv,0.2)
    else:
        return deConv
