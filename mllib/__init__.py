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

from .graph import build_graph

log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))  # Change the 2nd arg to INFO to suppress debug logging

EPOCH_SIZE = 2
BATCH_SIZE = 16
PREDICTION_BATCH_SIZE=16

WEIGHT_DIR = "/tmp/mllib/weights"

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
    x_test: ndarray
        Test dataset used for validation test
    y_test: ndarray
        Ground truth for test dataset used for validation test
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

    if x_test is not None:
        test_dataset_size = x_test.shape[0]
        x_test = x_test / 255.0
        x_test = (x_test.reshape(test_dataset_size, h, w, channels)).astype(np.float32)

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

            total_cost = 0
            total_samples = 0 # for epoch
            for j in range(loop_count):
                current_batch_size = BATCH_SIZE
                k = j * current_batch_size
                next_k = k + current_batch_size

                total_samples += current_batch_size

                o, c = s.run([objective, cost],
                             feed_dict={x_placeholder: x_train[k:next_k], y_placeholder: y_train_one_hot[k:next_k]})

                total_cost += c * current_batch_size
                log.info("Epoch: %d/%d.  Batch: %d Cost for batch:%f, Cost for epoch: %f. Batch size: %d" % (
                    i + 1, num_epochs, batch_id, c, total_cost/total_samples, current_batch_size))
                batch_id += 1

            # remainder
            last_batch_size = x_train.shape[0] - next_k
            if last_batch_size > 0:
                k = next_k
                total_samples += last_batch_size

                o, c = s.run([objective, cost],
                             feed_dict={x_placeholder: x_train[k:k + last_batch_size],
                                        y_placeholder: y_train_one_hot[k:k + last_batch_size]})

                total_cost += c * last_batch_size
                log.info("Epoch: %d/%d.  Batch: %d Cost for batch:%f, Cost for epoch: %f. Batch size: %d" % (
                    i + 1, num_epochs, batch_id, c, total_cost/total_samples, last_batch_size))

            # Save weight
            weight_path = saver.save(s, str(weight_dir / Path("model.ckpt")))
            log.info("Saved model in: %s" % weight_path)

            # Display validation test stats
            if x_test is not None and y_test is not None:
                accuracy = test_accuracy(s, x_test, y_test, x_placeholder, y_hat_softmax)
                log.info("Validation test accuracy: %f percent" % (accuracy * 100))


        log.info("Training completed.")

        return total_cost/total_samples


def test_accuracy(s, x_test, y_test, x_placeholder, y_hat_softmax):
    """
    Test accuracy

    Parameters
    ----------
    s: tfSession
        An open tfSession
    x_placeholder: tensor
        Placeholder for input
    x_test: ndarray
        Test dataset
    y_test: ndarray
        Ground truth for test dataset
    y_hat_softmax: tensor
        Predicted value for y
    """

    dataset_size = x_test.shape[0]
    total_matched_count = 0
    next_k = 0
    loop_count = int(dataset_size / PREDICTION_BATCH_SIZE)  # for m = 5, batch_size = 2, this results in [0, 1]
    batch_id = 1

    for j in range(loop_count):
        current_batch_size = BATCH_SIZE
        k = j * current_batch_size
        next_k = k + current_batch_size

        y_hat_test_one_hot = s.run(y_hat_softmax, feed_dict={x_placeholder: x_test[k:next_k]})

        batch_id += 1

        y_hat_test_one_hot_int = np.argmax(y_hat_test_one_hot, axis=1)  # to int from one-hot vector
        y_sub = y_test[k:next_k]
        matched_indices = (y_hat_test_one_hot_int == y_sub)
        matched_count = y_sub[matched_indices].shape[0]
        total_matched_count += matched_count

    # remainder
    last_batch_size = dataset_size - next_k
    if last_batch_size > 0:
        k = next_k
        y_hat_test_one_hot = s.run(y_hat_softmax, feed_dict={x_placeholder: x_test[k:k + last_batch_size]})

        y_hat_test_one_hot_int = np.argmax(y_hat_test_one_hot, axis=1)  # to int from one-hot vector
        y_sub = y_test[k:k + last_batch_size]
        matched_indices = (y_hat_test_one_hot_int == y_sub)
        matched_count = y_sub[matched_indices].shape[0]
        total_matched_count += matched_count

    accuracy = total_matched_count / dataset_size
    log.info(
        "Matched: %d out of Total: %d (%f percent)" % (total_matched_count, dataset_size, accuracy * 100))

    return accuracy

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
    num_classes: int
        Number of classes in dataset

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

        accuracy = test_accuracy(s, x_test, y_test, x_placeholder, y_hat_softmax)
        log.info("Accuracy: %f percent" % (accuracy * 100))

    return accuracy

def predict(project_name, samples, num_classes):
    """
    Predict the class of each sample in samples.

    Parameters
    ----------
    project_name:
        Name of the project.  This is used to tag weights.
    samples: ndarray
        Sample data
    num_classes: int
        Number of classes in dataset

    Returns
    -------
    class_id: int
        Predicted class ID of each sample
    """
    dataset_size = samples[0]
    h = samples.shape[1]
    w = samples.shape[2]

    if len(samples.shape) == 3:  # grayscale image missing the channels
        samples = samples.reshape((samples.shape[0], samples.shape[1], samples.shape[2], 1))

    channels = samples.shape[3]

    weight_dir = Path(WEIGHT_DIR) / Path(project_name.replace(" ", "_"))

    tf.reset_default_graph()

    samples = samples / 255.0
    samples = samples.astype(np.float32)

    init_op, objective, cost, x_placeholder, y_placeholder, y_hat_softmax = build_graph(h, w, channels, num_classes)

    saver = tf.train.Saver()

    with tf.Session() as s:

        if weight_dir.exists():
            saver.restore(s, str(weight_dir / Path("model.ckpt")))
            log.info("Loaded weight from: %s" % str(Path(WEIGHT_DIR) / Path("model.ckpt")))
        else:
            log.fatal("Weights not found.")
            raise Exception("Weights not found.")

        y_hat_test_one_hot = s.run(y_hat_softmax, feed_dict={x_placeholder: samples})
        y_hat_test_one_hot_int = np.argmax(y_hat_test_one_hot, axis=1)  # to int from one-hot vector

    return y_hat_test_one_hot_int