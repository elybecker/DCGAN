import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import hops.hdfs as hdfs
from hops import tensorboard
import sys
import time
import os

from model import generator, discriminator
from utils import reScale, normalizeImages, getNextBatch

project_DIR = hdfs.project_path()
faces = "hdfs:///Projects/DCGAN/_data/celeb/celeb_64x64.npy"
catizh = "hdfs:///Projects/DCGAN/_data/cats/Cats_64x64.npy"

# Load data from HDFS


def wrapper(d_lr=2e-4, g_lr=2e-4):
    with hdfs.get_fs().open_file(catizh, "r") as someFile:
        importNumpyCats = np.load(someFile)
   
    CatImages = np.reshape(importNumpyCats, (-1, 64, 64, 3))
    org_CatImages = CatImages

    logdir = tensorboard.logdir()
    if not os.path.exists(logdir + '/train'):
        os.mkdir(logdir + '/train')

    image = CatImages[0]
    IMAGE_XDIM = image.shape[0]
    IMAGE_YDIM = image.shape[1]
    IMAGE_ZDIM = image.shape[2]

    shape = len(CatImages[0]), IMAGE_XDIM, IMAGE_YDIM, IMAGE_ZDIM
    out_channel_dim = shape[3]

    tf.reset_default_graph()
    # Defining Placeholder
    input_real_images = tf.placeholder(shape=[None, 64, 64, 3], dtype=tf.float32, name='input_real_images')
    input_z = tf.placeholder(dtype=tf.float32, shape=[None, 1, 1, 100], name='input_z')
    X_viz = tf.placeholder(tf.float32, shape=[3, 64, 64, 3])  # input Image
    isTrain = tf.placeholder(dtype=tf.bool)

    label_smoothing = 0.9

    # Defining the Networks
    g_model = generator(input_z, out_channel_dim, isTrain)
    d_model_real, d_logits_real = discriminator(input_real_images, isTrain)
    d_model_fake, d_logits_fake = discriminator(g_model, isTrain, reuse=True)

    # Defining Loss
    # Discriminator Loss
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_logits_real, labels=tf.ones_like(d_model_real) * label_smoothing))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_logits_fake, labels=tf.zeros_like(d_model_fake)))
    d_loss = d_loss_real + d_loss_fake
    # Generator Loss
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_logits_fake, labels=tf.ones_like(d_model_fake) * label_smoothing))

    # Gathering Variables
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if var.name.startswith('D')]
    g_vars = [var for var in t_vars if var.name.startswith('G')]

    # Defining Optimizer
    d_beta1 = 0.5
    g_beta1 = 0.5
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        d_train_opt = tf.train.AdamOptimizer(d_lr, beta1=d_beta1).minimize(d_loss, var_list=d_vars)
        g_train_opt = tf.train.AdamOptimizer(g_lr, beta1=g_beta1).minimize(g_loss, var_list=g_vars)

    # Training Parameters
    steps = 1
    batch_size = 64
    z_dim = 100
    epochs = 1000
    iteration = int(CatImages.shape[0] / batch_size)
    save_step = 10
    # Normalise to -1.0 to 1.0, tanH
    CatImages = normalizeImages(org_CatImages)
    # Start Training Session
    with tf.Session() as sess:
        # Values to be added to tensorboard
        tf.summary.scalar('Generator_loss', g_loss)
        tf.summary.scalar('Discriminator_loss', d_loss)
        tf.summary.image("Input_images", X_viz)
        merged = tf.summary.merge_all()
        # Define writer
        writer_train = tf.summary.FileWriter(logdir + "/train", sess.graph)
        sess.run(tf.global_variables_initializer())
        for epoch_i in range(epochs):
            for i in range(iteration):
                batch_images = getNextBatch(
                    CatImages, batch_size)  # MiniBatch real img
                # MiniBatch noise
                batch_z = np.random.uniform(-1.0, 1.0,
                                            size=(batch_size, 1, 1, z_dim))
                _ = sess.run(d_train_opt, feed_dict={
                    input_real_images: batch_images, input_z: batch_z, isTrain: True})  # train Discriminator
                _ = sess.run(g_train_opt, feed_dict={
                    input_real_images: batch_images, input_z: batch_z, isTrain: True})  # train Generator
                if steps == 1 or steps % save_step == 0:  # record losses and retrevie images every ...save_step
                    # MiniBatch noise
                    z_batch = np.random.uniform(-1.0, 1.0,
                                                size=[batch_size, 1, 1, z_dim])
                    # Example training images from current batch
                    x_vizu_3 = reScale(getNextBatch(batch_images, 3))
                    summary_train = sess.run(merged, feed_dict={
                        X_viz: x_vizu_3, input_real_images: batch_images, input_z: z_batch, isTrain: False})
                    writer_train.add_summary(summary_train, steps)
                steps += 1

if __name__ == "__main__":
    wrapper()
    
