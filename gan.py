import numpy as np
import tensorflow as tf
from keras.layers import *
from keras import Sequential
from tensorflow.keras.losses import BinaryCrossentropy

class MyGAN(object):

    def __init__(self, ds_train, ds_val, gen_model, disc_model, norm_size):
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.gen_model = gen_model
        self.disc_model = disc_model
        self.norm_zie = norm_size

    



# def generator():
#     model = Sequential()
#     model.add(InputLayer(input_shape=(128,128,1)))
#     model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
#     model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
#     model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
#     model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
#     model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
#     model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
#     model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
#     model.add(Conv2DTranspose(256, (3, 3), activation='relu', padding='same', strides=2))
#     model.add(Conv2DTranspose(128, (3, 3), activation='relu', padding='same', strides=2))
#     model.add(Conv2DTranspose(64, (3, 3), activation='relu', padding='same', strides=2))
#     model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
#     model.add(Conv2DTranspose(2, (3, 3), activation='sigmoid', padding='same'))
#     return model



