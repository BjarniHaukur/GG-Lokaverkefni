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
from helper_funcs import map_to, map_from


# class MyDataSet(object):

#     def __init__(self, flowDir):
#         self.flowDir = flowDir

def parse_to_image(filename):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float16)
    image = tfio.experimental.color.rgb_to_lab(image)
    
    return tf.expand_dims(image[:,:,0], -1)/100, (lambda x: (x+1)/2)(image[:,:,1:]/128)


def augment_image(image):
    return None