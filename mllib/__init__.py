#!/usr/bin/env python
"""
Main code for mllib.

The goal is to allow the user to use ML as a black box.

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

EPOCH_SIZE = 2
BATCH_SIZE = 16

WEIGHT_DIR = "/tmp/mllib/weights"


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
    """
    x_placeholder = tf.placeholder(tf.float32, shape=(None, h, w, channels))
    y_placeholder = tf.placeholder(tf.float32, shape=(None, classes))

    x_input = x_placeholder
    channels_prev = channels
    channels_this_layer = 8
    channel_count_multiplier = 1

    if h > 128:
        # Set up conv net
        with tf.variable_scope("cv256") as scope:
            channels_this_layer = 16
            weights = tf.get_variable("weights", [5, 5, channels_prev, channels_this_layer], dtype=tf.float32,
                                      initializer=tf.contrib.layers.xavier_initializer(seed=0))
            bias = tf.get_variable("bias", [channels_this_layer], dtype=tf.float32)
            z = tf.add(tf.nn.conv2d(x_input, weights, strides=[1, 1, 1, 1], padding='SAME'), bias)
            activation = tf.nn.relu(z, name=scope.name)
            # Apply pooling with ksize: batch, h, w, channel
            x_input = tf.nn.max_pool(activation, ksize=(1, 2, 2, 1), strides=[1, 2, 2, 1], padding='VALID')  # to 128
            channels_prev = channels_this_layer
            channel_count_multiplier *= 2

    if h > 64:
        # Set up conv net
        with tf.variable_scope("cv128") as scope:
            channels_this_layer = 8 * channel_count_multiplier  # 32
            weights = tf.get_variable("weights", [5, 5, channels_prev, channels_this_layer], dtype=tf.float32,
                                      initializer=tf.contrib.layers.xavier_initializer(seed=0))
            bias = tf.get_variable("bias", [channels_this_layer], dtype=tf.float32)
            z = tf.add(tf.nn.conv2d(x_input, weights, strides=[1, 1, 1, 1], padding='SAME'), bias)
            activation = tf.nn.relu(z, name=scope.name)
            # Apply pooling with ksize: batch, h, w, channel
            x_input = tf.nn.max_pool(activation, ksize=(1, 2, 2, 1), strides=[1, 2, 2, 1], padding='VALID')  # to 64
            channels_prev = channels_this_layer
            channel_count_multiplier *= 2

    if h > 32:
        # Set up conv net
        with tf.variable_scope("cv64") as scope:
            channels_this_layer = 8 * channel_count_multiplier  # 64
            weights = tf.get_variable("weights", [5, 5, channels_prev, channels_this_layer], dtype=tf.float32,
                                      initializer=tf.contrib.layers.xavier_initializer(seed=0))
            bias = tf.get_variable("bias", [channels_this_layer], dtype=tf.float32)
            z = tf.add(tf.nn.conv2d(x_input, weights, strides=[1, 1, 1, 1], padding='SAME'), bias)
            activation = tf.nn.relu(z, name=scope.name)
            # Apply pooling with ksize: batch, h, w, channel
            x_input = tf.nn.max_pool(activation, ksize=(1, 2, 2, 1), strides=[1, 2, 2, 1], padding='VALID')  # to 32
            channels_prev = channels_this_layer
            channel_count_multiplier *= 2

    # Set up conv net
    with tf.variable_scope("cv1") as scope:
        channels_this_layer = 8 * channel_count_multiplier  # 128
        weights = tf.get_variable("weights", [5, 5, channels_prev, channels_this_layer], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer(seed=0))
        bias = tf.get_variable("bias", [channels_this_layer], dtype=tf.float32)
        z = tf.add(tf.nn.conv2d(x_input, weights, strides=[1, 1, 1, 1], padding='SAME'), bias)
        activation = tf.nn.relu(z, name=scope.name)
        # Apply pooling with ksize: batch, h, w, channel
        p1 = tf.nn.max_pool(activation, ksize=(1, 2, 2, 1), strides=[1, 2, 2, 1], padding='VALID')  # to 16x16
        channels_prev = channels_this_layer
        channel_count_multiplier *= 2

    with tf.variable_scope("cv2") as scope:
        channels_this_layer = 8 * channel_count_multiplier  # 256
        weights = tf.get_variable("weights", [3, 3, channels_prev, channels_this_layer], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer(seed=0))
        bias = tf.get_variable("bias", [channels_this_layer], dtype=tf.float32)
        z = tf.add(tf.nn.conv2d(p1, weights, strides=[1, 1, 1, 1], padding='SAME'), bias)
        activation = tf.nn.relu(z, name=scope.name)
        p2 = tf.nn.max_pool(activation, ksize=(1, 2, 2, 1), strides=[1, 2, 2, 1], padding='VALID')  # to 8x8
        channels_prev = channels_this_layer
        channel_count_multiplier *= 2

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


def train(project_name, x_train, y_train, num_classes, num_epochs=EPOCH_SIZE, x_test=None, y_test=None):
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
    num_classes: int
        Number of classes in dataset.  For example, 10 for MNIST.
    num_epochs: int
        Number of epochs
    Returns
    -------
    cost: float
        Cost
    """
    tf.reset_default_graph()

    weight_dir = Path(WEIGHT_DIR) / Path(project_name)
    if weight_dir.exists() is False:
        weight_dir.mkdir(parents=True, exist_ok=True)
        log.info("Created %s" % (weight_dir))

    if len(x_train.shape) == 3:  # grayscale image missing the channels
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))

    dataset_size = x_train.shape[0]
    h = x_train.shape[1]
    w = x_train.shape[2]
    channels = x_train.shape[3]

    # Randomly sort
    r = np.arange(0, dataset_size, 1)  # For dataset size = 5, generate 0, 1, 2, 3, 4
    p = np.random.permutation(r)  # Shuffle, 4, 2, 0, 1, 3
    x_train = x_train[p]  # Apply the new sequence above
    y_train = y_train[p]

    # Change the value from 0<= x <= 255 in UINT8 to 0 <= x <= 1 in float
    x_train = x_train / 255.0
    x_train = (x_train.reshape(dataset_size, h, w, channels)).astype(np.float32)

    y_train_one_hot = y_train.reshape(y_train.shape[0], 1)
    y_train_one_hot = keras.utils.to_categorical(y_train_one_hot, num_classes).astype(np.float32)

    init_op, objective, cost, x_placeholder, y_placeholder, y_hat_softmax = build_graph(h, w, channels, num_classes)

    saver = tf.train.Saver()

    with tf.Session() as s:

        if Path(WEIGHT_DIR).exists():
            try:
                saver.restore(s, str(weight_dir / Path("model.ckpt")))
                log.info("Loaded weight from: %s" % str(weight_dir / Path("model.ckpt")))
            except:
                log.info("Weight could not be loaded. Proceeding.")
                s.run(init_op)
        else:
            log.info("Weights not found. Proceeding.")
            s.run(init_op)  # Actually assign initial value to variables

        dataset_size = y_train.shape[0]

        for i in range(num_epochs):
            next_k = 0
            loop_count = int(
                dataset_size / BATCH_SIZE)  # for m = 5, batch_size = 2, this results in [0, 1]
            current_batch_size = 0
            batch_id = 1

            for j in range(loop_count):
                current_batch_size = BATCH_SIZE
                k = j * current_batch_size
                next_k = k + current_batch_size

                o, c = s.run([objective, cost],
                             feed_dict={x_placeholder: x_train[k:next_k], y_placeholder: y_train_one_hot[k:next_k]})

                log.info("Epoch: %d/%d.  Batch: %d Cost:%f, Batch size: %d" % (
                i + 1, num_epochs, batch_id, c, current_batch_size))
                batch_id += 1

            # remainder
            last_batch_size = x_train.shape[0] - next_k
            if last_batch_size > 0:
                k = next_k

                o, c = s.run([objective, cost],
                             feed_dict={x_placeholder: x_train[k:k + last_batch_size],
                                        y_placeholder: y_train_one_hot[k:k + last_batch_size]})

                log.info("Epoch: %d/%d.  Batch: %d Cost:%f, Batch size: %d" % (
                i + 1, num_epochs, batch_id, c, last_batch_size))

            # Save weight
            weight_path = saver.save(s, str(weight_dir / Path("model.ckpt")))
            log.info("Saved model in: %s" % weight_path)

            # Display stats
            if x_test is not None and y_test is not None:
                y_hat_test_one_hot = s.run(y_hat_softmax, feed_dict={x_placeholder: x_test})

                total_size = y_hat_test_one_hot.shape[0]
                y_hat_test_one_hot_int = np.argmax(y_hat_test_one_hot, axis=1)  # to int from one-hot vector

                matched_indices = (y_hat_test_one_hot_int == y_test)
                matched_count = y_test[matched_indices].shape[0]
                accuracy = matched_count / total_size
                log.info(
                    "Matched: %d out of Total: %d (%f percent)" % (matched_count, total_size, accuracy * 100))

        log.info("Training completed.")

        return c


def test(project_name, x_test, y_test, num_classes):
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
    dataset_size = x_test.shape[0]
    h = x_test.shape[1]
    w = x_test.shape[2]

    if len(x_test.shape) == 3:  # grayscale image missing the channels
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

    channels = x_test.shape[3]

    weight_dir = Path(WEIGHT_DIR) / Path(project_name.replace(" ", "_"))

    tf.reset_default_graph()

    x_test = x_test / 255.0
    x_test = (x_test.reshape(dataset_size, h, w, channels)).astype(np.float32)

    init_op, objective, cost, x_placeholder, y_placeholder, y_hat_softmax = build_graph(h, w, channels, num_classes)

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
