'''
This code was adapted from
https://medium.com/@joeyism/creating-alexnet-on-tensorflow-from-scratch-part-1-getting-cifar-10-data-46d349a4282f
and was needed to work with the Cifar-10 dataset.
'''

import numpy as np
import pickle
import os
import math

def __extract_file__(fname):
    print(fname)
    with open(fname, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d


def __unflatten_image__(img_flat):
    # Breaks image up into three different color channels and then adds those together
    # [32, 32, 3]
    img_R = img_flat[0:1024].reshape((32,32))
    img_G = img_flat[1024:2048].reshape((32,32))
    img_B = img_flat[2048:3072].reshape((32,32))
    img = np.dstack((img_R, img_G, img_B))
    return img

def __extract_reshape_file__(fname):
    # Reshapes the data/labels to be pairs of image-labels
    res = []
    d = __extract_file__(fname)
    images = d[b"data"]
    labels = d[b"labels"]
    for image, label in zip(images, labels):
        res.append((__unflatten_image__(image), label))
    return res

def get_images_from(dir):
    # Extracts data from different files where they are stored
    files = [f for f in os.listdir(dir) if f.startswith("data_batch")]
    res = []
    for f in files:
        res += __extract_reshape_file__(os.path.join(dir, f))
    return res


class Cifar(object):

    def __init__(self, dir="cifar-10-batches-py/", batch_size=1):
        # Initializes variables and sets up how many batches there are and how many images are in each batch
        self.__res__ = get_images_from(dir)
        self.test_set = __extract_reshape_file__(os.path.join(dir, "test_batch"))

    def return_train(self):
        return self.__res__

    def return_test(self):
        return self.test_set