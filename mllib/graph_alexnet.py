#!/usr/bin/env python
"""
Build graph for mllib.

__author__ = "Hide Inada"
__copyright__ = "Copyright 2018, Hide Inada"
__license__ = "The MIT License"
__email__ = "hideyuki@gmail.com"
"""

import os
import logging

from pathlib import Path
import tensorflow as tf
import numpy as np

import keras

log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))  # Change the 2nd arg to INFO to suppress debug logging


def build_alexnet(h, w, channels, classes):
    """
    Build TensorFlow graph

    Parameters
    ----------
    h: int
        Height of the image
    w: int
        Width of the image
    channels: int
        Number of channels of the image
    classes: int
        Number of classes of the dataset

    Returns
    -------
    init_op: tensor
        Operation to initialize all variables
    objective: tensor
        Optimization objective
    cost: tensor
        Per sample cost
    x_placeholder: tensor
        Placeholder to feed x input
    y_placeholder: tensor
        Placeholder to feed y input

    Raises
    ------
    ValueError
        If an image is larger than 256 high x 256 wide
    """
    x_placeholder = tf.placeholder(tf.float32, shape=(None, h, w, channels))
    y_placeholder = tf.placeholder(tf.float32, shape=(None, classes))

    input_to_this_layer = x_placeholder
    channels_prev = channels
    channel_count_multiplier = 1

    if h > 256 or w > 256:
        raise ValueError('Image size larger than 256 high x 256 wide is not supported.')

    if h > 128:
        with tf.variable_scope("cv1") as scope:
            channels_this_layer = 96
            weights = tf.get_variable("weights", [11, 11, channels_prev, channels_this_layer], dtype=tf.float32,
                                      initializer=tf.contrib.layers.xavier_initializer(seed=0))
            bias = tf.get_variable("bias", [channels_this_layer], dtype=tf.float32)
            z = tf.add(tf.nn.conv2d(input_to_this_layer, weights, strides=[1, 4, 4, 1], padding='VALID'), bias)
            activation = tf.nn.relu(z, name=scope.name)

            input_to_this_layer = tf.nn.max_pool(activation, ksize=(1, 3, 3, 1), strides=[1, 2, 2, 1], padding='VALID')
            channels_prev = channels_this_layer
            channel_count_multiplier *= 2

            # to 55x55

    if h > 64:
        with tf.variable_scope("cv2") as scope:
            channels_this_layer = 256
            weights = tf.get_variable("weights", [5, 5, channels_prev, channels_this_layer], dtype=tf.float32,
                                      initializer=tf.contrib.layers.xavier_initializer(seed=0))
            bias = tf.get_variable("bias", [channels_this_layer], dtype=tf.float32)
            z = tf.add(tf.nn.conv2d(input_to_this_layer, weights, strides=[1, 1, 1, 1], padding='SAME'), bias)
            activation = tf.nn.relu(z, name=scope.name)

            # Apply pooling with ksize: batch, h, w, channel
            input_to_this_layer = tf.nn.max_pool(activation, ksize=(1, 3, 3, 1), strides=[1, 2, 2, 1], padding='VALID')
            channels_prev = channels_this_layer
            channel_count_multiplier *= 2
            # to 27

    if h > 32:
        with tf.variable_scope("cv3") as scope:
            channels_this_layer = 384
            weights = tf.get_variable("weights", [3, 3, channels_prev, channels_this_layer], dtype=tf.float32,
                                      initializer=tf.contrib.layers.xavier_initializer(seed=0))
            bias = tf.get_variable("bias", [channels_this_layer], dtype=tf.float32)
            z = tf.add(tf.nn.conv2d(input_to_this_layer, weights, strides=[1, 1, 1, 1], padding='SAME'), bias)
            activation = tf.nn.relu(z, name=scope.name)
            input_to_this_layer = activation
            channels_prev = channels_this_layer

        with tf.variable_scope("cv4") as scope:
            channels_this_layer = 384
            weights = tf.get_variable("weights", [3, 3, channels_prev, channels_this_layer], dtype=tf.float32,
                                      initializer=tf.contrib.layers.xavier_initializer(seed=0))
            bias = tf.get_variable("bias", [channels_this_layer], dtype=tf.float32)
            z = tf.add(tf.nn.conv2d(input_to_this_layer, weights, strides=[1, 1, 1, 1], padding='SAME'), bias)
            activation = tf.nn.relu(z, name=scope.name)
            input_to_this_layer = activation
            channels_prev = channels_this_layer

        with tf.variable_scope("cv5") as scope:
            channels_this_layer = 256
            weights = tf.get_variable("weights", [3, 3, channels_prev, channels_this_layer], dtype=tf.float32,
                                      initializer=tf.contrib.layers.xavier_initializer(seed=0))
            bias = tf.get_variable("bias", [channels_this_layer], dtype=tf.float32)
            z = tf.add(tf.nn.conv2d(input_to_this_layer, weights, strides=[1, 1, 1, 1], padding='SAME'), bias)
            activation = tf.nn.relu(z, name=scope.name)
            input_to_this_layer = activation
            channels_prev = channels_this_layer


    with tf.variable_scope("cv1") as scope:
        p1 = tf.nn.max_pool(input_to_this_layer, ksize=(1, 3, 3, 1), strides=[1, 2, 2, 1], padding='VALID')
        input_to_this_layer = p1

    flat = tf.contrib.layers.flatten(input_to_this_layer)
    input_to_this_layer = flat

    fc6 = tf.contrib.layers.fully_connected(input_to_this_layer, activation_fn=tf.nn.relu, num_outputs=4096)
    input_to_this_layer = fc6

    fc7 = tf.contrib.layers.fully_connected(input_to_this_layer, activation_fn=tf.nn.relu, num_outputs=4096)
    input_to_this_layer = fc7

    fc8 = tf.contrib.layers.fully_connected(input_to_this_layer, activation_fn=tf.nn.relu, num_outputs=classes)
    input_to_this_layer = fc8

    y_hat_softmax = tf.nn.softmax(input_to_this_layer)  # you can just use fc2 for prediction if you want to further optimize

    # Set up optimizer & cost
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc8, labels=y_placeholder))
    tf.summary.scalar('Cost', cost)
    objective = optimizer.minimize(cost)

    init_op = tf.global_variables_initializer()  # Set up operator to assign all init values to variables

    return init_op, objective, cost, x_placeholder, y_placeholder, y_hat_softmax
