#!/usr/bin/env python
"""
Entry point for mllib

The goal is to allow the user design a generic convnet and wrap in a library.

__author__ = "Hide Inada"
__copyright__ = "Copyright 2018, Hide Inada"
__license__ = "The MIT License"
__email__ = "hideyuki@gmail.com"
"""

import os
import logging

import tensorflow as tf
import numpy as np
import sys

import keras

log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))  # Change the 2nd arg to INFO to suppress debug logging

log.info("mllib loaded")

EPOCH_SIZE = 1

import os
import logging

import tensorflow as tf
import numpy as np
from pathlib import Path
import keras

log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))  # Change the 2nd arg to INFO to suppress debug logging

EPOCH_SIZE = 2
BATCH_SIZE = 32

WEIGHT_DIR = "/tmp/mllib/weights"


def build_graph():
    """
    Build TensorFlow graph

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
    """
    x_placeholder = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y_placeholder = tf.placeholder(tf.float32, shape=(None, 10))

    # Set up conv net
    with tf.variable_scope("cv1") as scope:
        weights = tf.get_variable("weights", [5, 5, 1, 8], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer(seed=0))
        bias = tf.get_variable("bias", [8], dtype=tf.float32)
        z = tf.add(tf.nn.conv2d(x_placeholder, weights, strides=[1, 1, 1, 1], padding='SAME'), bias)
        activation = tf.nn.relu(z, name=scope.name)
        # Apply pooling with ksize: batch, h, w, channel
        p1 = tf.nn.max_pool(activation, ksize=(1, 2, 2, 1), strides=[1, 2, 2, 1], padding='VALID')  # to 14x14

    with tf.variable_scope("cv2") as scope:
        weights = tf.get_variable("weights", [3, 3, 8, 16], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer(seed=0))
        bias = tf.get_variable("bias", [16], dtype=tf.float32)
        z = tf.add(tf.nn.conv2d(p1, weights, strides=[1, 1, 1, 1], padding='SAME'), bias)
        activation = tf.nn.relu(z, name=scope.name)
        p2 = tf.nn.max_pool(activation, ksize=(1, 2, 2, 1), strides=[1, 2, 2, 1], padding='VALID')  # to 7x7

    flat = tf.contrib.layers.flatten(p2)
    fc1 = tf.contrib.layers.fully_connected(flat, activation_fn=tf.nn.relu, num_outputs=128)
    fc2 = tf.contrib.layers.fully_connected(fc1, activation_fn=None, num_outputs=10)
    y_hat_softmax = tf.nn.softmax(fc2)  # you can just use fc2 for prediction if you want to further optimize

    # Set up optimizer & cost
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc2, labels=y_placeholder))
    objective = optimizer.minimize(cost)

    init_op = tf.global_variables_initializer()  # Set up operator to assign all init values to variables

    return init_op, objective, cost, x_placeholder, y_placeholder, y_hat_softmax


def train(project_name, x_train, y_train, epoch_size=EPOCH_SIZE):
    """
    Train the model

    Parameters
    ----------
    project_name:
        Name of the project.  This is used to tag weights.
    x_train: ndarray
        Training dataset
    y_train: ndarray
        Ground truth for training dataset

    Returns
    -------
    cost: float
        Cost
    """
    tf.reset_default_graph()

    weight_dir = Path(WEIGHT_DIR) / Path(project_name.replace(" ", "_"))
    if weight_dir.exists() is False:
        weight_dir.mkdir(parents=True, exist_ok=True)
        log.info("Created %s" % (weight_dir))

    # Change the value from 0<= x <= 255 in UINT8 to 0 <= x <= 1 in float
    x_train = x_train / 255.0
    x_train = (x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)).astype(np.float32)

    y_train_one_hot = y_train.reshape(y_train.shape[0], 1)
    y_train_one_hot = keras.utils.to_categorical(y_train_one_hot, 10).astype(np.float32)

    init_op, objective, cost, x_placeholder, y_placeholder, y_hat_softmax = build_graph()

    saver = tf.train.Saver()

    with tf.Session() as s:

        if Path(WEIGHT_DIR).exists():
            try:
                saver.restore(s, str(weight_dir / Path("model.ckpt")))
                log.info("Loaded weight from: %s" % str(weight_dir / Path("model.ckpt")))
            except:
                log.info("Weight could not be loaded. Proceeding.")
        else:
            log.info("Weights not found. Proceeding.")
        s.run(init_op)  # Actually assign initial value to variables

        dataset_size = y_train.shape[0]

        for i in range(epoch_size):
            next_k = 0
            loop_count = int(
                dataset_size / BATCH_SIZE)  # for m = 5, batch_size = 2, this results in [0, 1]
            current_batch_size = 0

            for j in range(loop_count):
                current_batch_size = BATCH_SIZE
                k = j * current_batch_size
                next_k = k + current_batch_size

                o, c = s.run([objective, cost],
                             feed_dict={x_placeholder: x_train[k:next_k], y_placeholder: y_train_one_hot[k:next_k]})

                log.info("Epoch: %d.  Batch: %d Cost:%f, Batch size: %d" % (i, j, c, current_batch_size))

            # remainder
            last_batch_size = x_train.shape[0] - next_k
            if last_batch_size > 0:
                k = next_k

                o, c = s.run([objective, cost],
                             feed_dict={x_placeholder: x_train[k:k + last_batch_size],
                                        y_placeholder: y_train_one_hot[k:k + last_batch_size]})

                log.info("Epoch: %d.  Batch: %d Cost:%f, Batch size: %d" % (i, j, c, last_batch_size))

        log.info("Training completed.")

        weight_path = saver.save(s, str(weight_dir / Path("model.ckpt")))
        log.info("Saved model in: %s" % weight_path)

        return c


def test(project_name, x_test, y_test):
    """
    Test accuracy using test dataset.

    Parameters
    ----------
    project_name:
        Name of the project.  This is used to tag weights.
    x_test: ndarray
        Test dataset
    y_test: ndarray
        Ground truth for test dataset

    Returns
    -------
    accuracy: float
        accuracy
    """
    weight_dir = Path(WEIGHT_DIR) / Path(project_name.replace(" ", "_"))

    tf.reset_default_graph()

    x_test = x_test / 255.0
    x_test = (x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)).astype(np.float32)

    init_op, objective, cost, x_placeholder, y_placeholder, y_hat_softmax = build_graph()

    saver = tf.train.Saver()

    with tf.Session() as s:

        if weight_dir.exists():
            saver.restore(s, str(weight_dir / Path("model.ckpt")))
            log.info("Loaded weight from: %s" % str(Path(WEIGHT_DIR) / Path("model.ckpt")))
        else:
            log.fatal("Weights not found.")
            raise Exception("Weights not found.")

        y_hat_test_one_hot = s.run(y_hat_softmax, feed_dict={x_placeholder: x_test})

        total_size = y_hat_test_one_hot.shape[0]
        y_hat_test_one_hot_int = np.argmax(y_hat_test_one_hot, axis=1)  # to int from one-hot vector

        matched_indices = (y_hat_test_one_hot_int == y_test)
        matched_count = y_test[matched_indices].shape[0]
        accuracy = matched_count / total_size
        log.info(
            "Matched: %d out of Total: %d (%f percent)" % (matched_count, total_size, accuracy * 100))

    return accuracy
