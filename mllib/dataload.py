#!/usr/bin/env python
'''Load image files containing images.

__author__ = "Hide Inada"
__copyright__ = "Copyright 2018, Hide Inada"
__license__ = "The MIT license"
__email__ = "hideyuki@gmail.com"
'''

import os
import logging

from pathlib import Path
from PIL import Image
import numpy as np
import random
import cv2 as cv

log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))  # Change the 2nd arg to INFO to suppress debug logging

MAX_NUM_CLASS = 10 # Use this flag to limit the number of classes to load

def load_image_data(image_root_dir, test_data_ratio=0.2):
    """Load image files from the dataset directory

    Parameters
    ----------
    image_root_dir: Path
        A Pathlib containing the name of the directory that holds sub-directories where each image set is located.

        For example, if the directory structure is below:
        /
        /usr/jack_canine
        /usr/jack_canine/data
        /usr/jack_canine/data/dogs_images
        /usr/jack_canine/data/dogs_images/st_bernard/*.jpg
        /usr/jack_canine/data/dogs_images/old_english_sheepdog/*.jpg

        you will pass Path("/usr/jack_canine/data/dogs_images") to this function

        You need to resize images to have the same height, width and the number of channels before calling this method.
        Dimension of the first image is used to determine these three parameters and if an image with non-matching
        dimension is found, the image is ignored.

    test_data_ratio: float
        Ratio to allocate to test dataset
    Returns
    -------
    (x_train, y_train): (ndarray, ndarray)
        Numpy array of training dataset and groundtruth of type int.
    (x_test, y_test): (ndarray, ndarray)
        Numpy array of test dataset and groundtruth of type int.
    class_id_to_name_dic: dict
        Class ID to class name mapping
    """

    train_image_list = list()
    train_class_id_list = list()
    test_image_list = list()
    test_class_id_list = list()

    image_shape = None

    class_id_to_name_dict = dict()
    current_class_id = 0

    # Loop for dirs
    for i, x in enumerate([x for x in image_root_dir.iterdir() if x.is_dir()]):
        label = x.stem
        log.debug(label)

        tmp_image_list = list()
        tmp_class_id_list = list()

        # Loop for files
        for file_type in ("*.jpeg", "*.JPG", "*.jpg", "*.png"):
            for f in x.glob(file_type):

                log.debug("Reading %s" % (f))
                image = cv.imread(str(f))
                if image_shape is None:
                    image_shape = image.shape

                if image_shape != image.shape:
                    log.warning("Skipping image.  Expecting shape: %s.  Found: %s" % (str(image_shape), str(image.shape)))
                    continue

                image = cv.cvtColor(image, cv.COLOR_BGR2RGB) # Switch to RGB
                image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]) # h, w, channels
                tmp_image_list.append(image)
                tmp_class_id_list.append(current_class_id)

        num_images = len(tmp_image_list)
        min_images_required = 50
        if num_images < min_images_required :
            log.debug("Skipping %s as there is only %d image" % (label, min_images_required))
            continue

        class_id_to_name_dict[current_class_id] = label

        num_test_images = max(int(num_images*test_data_ratio), 1)
        num_train_images = num_images - num_test_images

        # Append
        train_image_list += tmp_image_list[:num_train_images ]
        test_image_list += tmp_image_list[num_train_images:]
        train_class_id_list += tmp_class_id_list[:num_train_images ]
        test_class_id_list += tmp_class_id_list[num_train_images:]

        assert len(train_image_list) == len(train_class_id_list)
        assert len(test_image_list) == len(test_class_id_list)

        current_class_id += 1
        if current_class_id + 1 >= MAX_NUM_CLASS:
            break

    # Convert to numpy array
    x_train = np.concatenate(train_image_list, axis=0)
    y_train = np.array(train_class_id_list)
    x_test = np.concatenate(test_image_list, axis=0)
    y_test = np.array(test_class_id_list)

    log.info("Number of train images: %d" % (x_train.shape[0]))
    log.info("Number of test images: %d" % (x_test.shape[0]))

    return (x_train, y_train), (x_test, y_test), class_id_to_name_dict