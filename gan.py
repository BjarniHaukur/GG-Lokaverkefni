import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import *
from keras import Sequential
from tensorflow.keras.losses import BinaryCrossentropy

def generator():
    model = Sequential()
    model.add(InputLayer(input_shape=(128,128,1)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
    # model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
    model.add(UpSampling2D((2, 2)))
    # model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(2, (3,3), activation='sigmoid', padding='same'))
    return model

def discriminator():
    model = Sequential()
    model.add(InputLayer(input_shape=(128,128,2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))
    return model



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



