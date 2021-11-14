
from keras.layers import Input, Conv2D, BatchNormalization, UpSampling2D, Dense, InputLayer
from keras.models import Model
from keras.regularizers import l2
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.optimizers import Adam
from keras import Sequential

class NeuralNets(object):

    def __init__(self, norm_size):
        self.norm_size = norm_size
    
    def model_a(self, kernel = (2,2)):
        input_shape = (self.norm_size[0], self.norm_size[1], 1)

        model = Sequential()
        model.add(InputLayer(input_shape=input_shape))
        model.add(Conv2D(64, kernel, activation='relu', padding='same'))
        model.add(Conv2D(64, kernel, activation='relu', padding='same', strides=2))
        model.add(Conv2D(128, kernel, activation='relu', padding='same'))
        model.add(UpSampling2D((2,2)))
        model.add(Conv2D(128, kernel, activation='relu', padding='same', strides=2))
        model.add(Conv2D(256, kernel, activation='relu', padding='same'))
        model.add(Conv2D(256, kernel, activation='relu', padding='same', strides=2))
        model.add(Conv2D(512, kernel, activation='relu', padding='same'))
        model.add(Conv2D(256, kernel, activation='relu', padding='same'))
        model.add(Conv2D(128, kernel, activation='relu', padding='same'))
        model.add(UpSampling2D((2,2)))
        model.add(Conv2D(64, kernel, activation='relu', padding='same'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(32, kernel, activation='relu', padding='same'))
        model.add(Conv2D(2, kernel, activation='sigmoid', padding='same'))

        return model