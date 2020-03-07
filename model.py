import tensorflow as tf
from layers import deConvLayer, convLayer

def generator(x, out_channel_dim, isTrain=True, reuse=False):
    with tf.variable_scope('G', reuse=reuse):
        h1 = deConvLayer(inputs=x, filters=1024, kernel_size=[4, 4], strides=(1, 1), padding='VALID',
                            activation='lrelu', batch_normalization=True, training=isTrain)

        h2 = deConvLayer(inputs=h1, filters=512, kernel_size=[4, 4], strides=(2, 2), padding='SAME',
                            activation='lrelu', batch_normalization=True, training=isTrain)

        h3 = deConvLayer(inputs=h2, filters=256, kernel_size=[4, 4], strides=(2, 2), padding='SAME',
                            activation='lrelu', batch_normalization=True, training=isTrain)

        h4 = deConvLayer(inputs=h3, filters=128, kernel_size=[4, 4], strides=(2, 2), padding='SAME',
                            activation='lrelu', batch_normalization=True, training=isTrain)
        # out layer
        h5 = tf.layers.conv2d_transpose(inputs=h4, filters=out_channel_dim,
                                        kernel_size=[4, 4], strides=(2, 2),
                                        padding='SAME')
        o = tf.nn.tanh(h5)
        tf.summary.image("Generated Images", o, 9)
        return o

def discriminator(x, isTrain=True, reuse=False):
    with tf.variable_scope('D', reuse=reuse):
        
        h1 = convLayer(inputs=x, filters=128, kernel_size=[4, 4], strides=(2, 2), padding='SAME', 
                        activation='lrelu', batch_normalization=True, training=isTrain)
        
        h2 = convLayer(inputs=h1, filters=256, kernel_size=[4, 4], strides=(2, 2), padding='SAME', 
                        activation='lrelu', batch_normalization=True, training=isTrain)
        
        h3 = convLayer(inputs=h2, filters=512, kernel_size=[4, 4], strides=(2, 2), padding='SAME', 
                        activation='lrelu', batch_normalization=True, training=isTrain)
        
        h4 = convLayer(inputs=h3, filters=1024, kernel_size=[4, 4], strides=(2, 2), padding='SAME', 
                        activation='lrelu', batch_normalization=True, training=isTrain)
        # out Layer
        h5 = tf.layers.conv2d(inputs=h4, filters=1, kernel_size=[4, 4], strides=(1, 1), padding='VALID')
        o = tf.nn.sigmoid(h5)
        return o, h5
