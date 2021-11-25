import os
from os import walk
import shutil
import json
import numpy as np
from PIL import Image, ImageChops
from tqdm import tqdm
from skimage import color
import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_addons as tfa
from helper_funcs import map_to, map_from


# class MyDataSet(object):

#     def __init__(self, flowDir):
#         self.flowDir = flowDir

def parse_to_image(filename):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float16)
    image = tfio.experimental.color.rgb_to_lab(image)
    return tf.expand_dims(image[:,:,0]/100, -1), (image[:,:,1:]/128+1)/2

def parse_and_augment(filename):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float16)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    image = tfio.experimental.color.rgb_to_lab(image)
    
    return tf.expand_dims(image[:,:,0]/100, -1), (image[:,:,1:]/128+1)/2


def parse_to_array(filename):
    filename.as_numpy()
    print(type(filename))
    arr = filename.numpy()
    print(type(arr))
    exit()
    # with tf.io.gfile.GFile(filename, "r") as arr:
    #     print(arr.next())
    #     exit()
    # return tf.expand_dims(arr[:,:,0], -1), arr[:,:,1:]

def augment_image(image):
    return None