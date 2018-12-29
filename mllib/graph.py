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


def build_graph(h, w, channels, classes):
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
        with tf.variable_scope("cv256") as scope:
            channels_this_layer = 16
            weights = tf.get_variable("weights", [5, 5, channels_prev, channels_this_layer], dtype=tf.float32,
                                      initializer=tf.contrib.layers.xavier_initializer(seed=0))
            bias = tf.get_variable("bias", [channels_this_layer], dtype=tf.float32)
            z = tf.add(tf.nn.conv2d(input_to_this_layer, weights, strides=[1, 1, 1, 1], padding='SAME'), bias)
            activation = tf.nn.relu(z, name=scope.name)
            input_to_this_layer = tf.nn.max_pool(activation, ksize=(1, 2, 2, 1), strides=[1, 2, 2, 1], padding='VALID')
            channels_prev = channels_this_layer
            channel_count_multiplier *= 2
            # 256x256 3 channels to 128x128, 16 channels

    if h > 64:
        with tf.variable_scope("cv128") as scope:
            channels_this_layer = 8 * channel_count_multiplier
            weights = tf.get_variable("weights", [5, 5, channels_prev, channels_this_layer], dtype=tf.float32,
                                      initializer=tf.contrib.layers.xavier_initializer(seed=0))
            bias = tf.get_variable("bias", [channels_this_layer], dtype=tf.float32)
            z = tf.add(tf.nn.conv2d(input_to_this_layer, weights, strides=[1, 1, 1, 1], padding='SAME'), bias)
            activation = tf.nn.relu(z, name=scope.name)
            # Apply pooling with ksize: batch, h, w, channel
            input_to_this_layer = tf.nn.max_pool(activation, ksize=(1, 2, 2, 1), strides=[1, 2, 2, 1], padding='VALID')
            channels_prev = channels_this_layer
            channel_count_multiplier *= 2
            # to 64x64, 32 channels

    if h > 32:
        with tf.variable_scope("cv64") as scope:
            channels_this_layer = 8 * channel_count_multiplier
            weights = tf.get_variable("weights", [5, 5, channels_prev, channels_this_layer], dtype=tf.float32,
                                      initializer=tf.contrib.layers.xavier_initializer(seed=0))
            bias = tf.get_variable("bias", [channels_this_layer], dtype=tf.float32)
            z = tf.add(tf.nn.conv2d(input_to_this_layer, weights, strides=[1, 1, 1, 1], padding='SAME'), bias)
            activation = tf.nn.relu(z, name=scope.name)
            # Apply pooling with ksize: batch, h, w, channel
            input_to_this_layer = tf.nn.max_pool(activation, ksize=(1, 2, 2, 1), strides=[1, 2, 2, 1], padding='VALID')
            channels_prev = channels_this_layer
            channel_count_multiplier *= 2
            # to 32x32, 64 channels

    with tf.variable_scope("cv1") as scope:
        channels_this_layer = 8 * channel_count_multiplier  # 128
        weights = tf.get_variable("weights", [5, 5, channels_prev, channels_this_layer], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer(seed=0))
        bias = tf.get_variable("bias", [channels_this_layer], dtype=tf.float32)
        z = tf.add(tf.nn.conv2d(input_to_this_layer, weights, strides=[1, 1, 1, 1], padding='SAME'), bias)
        activation = tf.nn.relu(z, name=scope.name)
        # Apply pooling with ksize: batch, h, w, channel
        p1 = tf.nn.max_pool(activation, ksize=(1, 2, 2, 1), strides=[1, 2, 2, 1], padding='VALID')
        channels_prev = channels_this_layer
        channel_count_multiplier *= 2
        # to 16x16, 128 channels

    with tf.variable_scope("cv2") as scope:
        channels_this_layer = 8 * channel_count_multiplier  # 256
        weights = tf.get_variable("weights", [3, 3, channels_prev, channels_this_layer], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer(seed=0))
        bias = tf.get_variable("bias", [channels_this_layer], dtype=tf.float32)
        z = tf.add(tf.nn.conv2d(p1, weights, strides=[1, 1, 1, 1], padding='SAME'), bias)
        activation = tf.nn.relu(z, name=scope.name)
        p2 = tf.nn.max_pool(activation, ksize=(1, 2, 2, 1), strides=[1, 2, 2, 1], padding='VALID')
        channels_prev = channels_this_layer
        channel_count_multiplier *= 2
        # to 8x8, 256 channels

    flat = tf.contrib.layers.flatten(p2)
    fc1 = tf.contrib.layers.fully_connected(flat, activation_fn=tf.nn.relu, num_outputs=2048)
    fc2 = tf.contrib.layers.fully_connected(fc1, activation_fn=None, num_outputs=classes)
    y_hat_softmax = tf.nn.softmax(fc2)  # you can just use fc2 for prediction if you want to further optimize

    # Set up optimizer & cost
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc2, labels=y_placeholder))
    objective = optimizer.minimize(cost)

    init_op = tf.global_variables_initializer()  # Set up operator to assign all init values to variables

    return init_op, objective, cost, x_placeholder, y_placeholder, y_hat_softmax
