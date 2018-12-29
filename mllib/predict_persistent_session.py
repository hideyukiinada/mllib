#!/usr/bin/env python
"""
Main code for mllib.

This class provides a faster prediction call as tfSession is kept alive.
To use, wrap in the with statement block:
    with Predictor as p:
        p.predict()

__author__ = "Hide Inada"
__copyright__ = "Copyright 2018, Hide Inada"
__license__ = "The MIT License"
__email__ = "hideyuki@gmail.com"
"""

import os
import logging
from contextlib import ContextDecorator

from pathlib import Path
import tensorflow as tf
import numpy as np

from .graph import build_graph

log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))  # Change the 2nd arg to INFO to suppress debug logging

WEIGHT_DIR = "/tmp/mllib/weights"


# class Predictor(ContextDecorator):
class Predictor():

    def __init__(self, project_name, shape, num_classes):
        """
        Initialize and set up Predictor class with image data.

        Parameters
        ----------
        project_name:
            Name of the project.  This is used to tag weights.
        shape: tuple
            Shape of a single image
        num_classes: int
            Number of classes in dataset

        Returns
        -------
        class_id: int
            Predicted class ID of each sample
        """

        self.h = shape[0]
        self.w = shape[1]
        self.channels = shape[2]
        self.num_classes = num_classes
        self.weight_dir = Path(WEIGHT_DIR) / Path(project_name.replace(" ", "_"))

    def start_session(self):
        tf.reset_default_graph()

        self.init_op, self.objective, self.cost, self.x_placeholder, self.y_placeholder, self.y_hat_softmax = \
            build_graph(self.h, self.w, self.channels, self.num_classes)

        self.session = tf.Session()

        self.saver = tf.train.Saver()

        if self.weight_dir.exists():
            self.saver.restore(self.session, str(self.weight_dir / Path("model.ckpt")))
            log.info("Loaded weight from: %s" % str(Path(WEIGHT_DIR) / Path("model.ckpt")))
        else:
            log.fatal("Weights not found.")
            raise Exception("Weights not found.")

    def __enter__(self):
        logging.debug('Predictor.__enter__ called.')
        self.start_session()
        return self  # returns value to with statement

    def __exit__(self, exc_type, exc, exc_tb):
        self.session.close()
        logging.debug('Exiting Predictor. Session closed.')

    def predict(self, samples):
        """
        Predict the class of each sample in samples.

        Parameters
        ----------
        samples: ndarray
            Sample data

        Returns
        -------
        class_id: int
            Predicted class ID of each sample

        Raises
        ------
        ValueError
            Unknown data format is detected
        """
        samples = samples / 255.0
        samples = samples.astype(np.float32)
        dataset_size = samples.shape[0]
        if len(samples.shape) == 3:  # grayscale image missing the channels
            samples = samples.reshape((dataset_size, self.h, self.w, self.channels, 1))

        y_hat_test_one_hot = self.session.run(self.y_hat_softmax, feed_dict={self.x_placeholder: samples})
        y_hat_test_one_hot_int = np.argmax(y_hat_test_one_hot, axis=1)  # to int from one-hot vector

        return y_hat_test_one_hot_int


import logging

logging.basicConfig(level=logging.INFO)


class track_entry_and_exit(ContextDecorator):
    def __init__(self, name):
        self.name = name
