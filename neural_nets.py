
from keras.layers import *
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

    def model_b(self):
        return self.__ResNet34()


    def model_c(self, kernel=2):
        l2_reg = l2(1e-3)
        input_tensor = Input(shape=(self.norm_size[0], self.norm_size[1], 1))
        x = Conv2D(64, (kernel, kernel), activation='relu', padding='same', name='conv1_1', kernel_initializer="he_normal",
                kernel_regularizer=l2_reg)(input_tensor)
        x = Conv2D(64, (kernel, kernel), activation='relu', padding='same', name='conv1_2', kernel_initializer="he_normal",
                kernel_regularizer=l2_reg, strides=(2, 2))(x)
        x = BatchNormalization()(x)

        x = Conv2D(128, (kernel, kernel), activation='relu', padding='same', name='conv2_1', kernel_initializer="he_normal",
                kernel_regularizer=l2_reg)(x)
        x = Conv2D(128, (kernel, kernel), activation='relu', padding='same', name='conv2_2', kernel_initializer="he_normal",
                kernel_regularizer=l2_reg,
                strides=(2, 2))(x)
        x = BatchNormalization()(x)

        x = Conv2D(256, (kernel, kernel), activation='relu', padding='same', name='conv3_1',
                kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
        x = Conv2D(256, (kernel, kernel), activation='relu', padding='same', name='conv3_2',
                kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
        x = Conv2D(256, (kernel, kernel), activation='relu', padding='same', name='conv3_3', kernel_initializer="he_normal",
                strides=(2, 2))(x)
        x = BatchNormalization()(x)

        x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', name='conv4_1',
                kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
        x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', name='conv4_2',
                kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
        x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', name='conv4_3',
                kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
        x = BatchNormalization()(x)

        x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', dilation_rate=2, name='conv5_1',
                kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
        x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', dilation_rate=2, name='conv5_2',
                kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
        x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', dilation_rate=2, name='conv5_3',
                kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
        x = BatchNormalization()(x)

        x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', dilation_rate=2, name='conv6_1',
                kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
        x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', dilation_rate=2, name='conv6_2',
                kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
        x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', dilation_rate=2, name='conv6_3',
                kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
        x = BatchNormalization()(x)

        x = Conv2D(256, (kernel, kernel), activation='relu', padding='same', name='conv7_1',
                kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
        x = Conv2D(256, (kernel, kernel), activation='relu', padding='same', name='conv7_2',
                kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
        x = Conv2D(256, (kernel, kernel), activation='relu', padding='same', name='conv7_3',
                kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
        x = BatchNormalization()(x)

        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(128, (kernel, kernel), activation='relu', padding='same', name='conv8_1',
                kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
        x = Conv2D(128, (kernel, kernel), activation='relu', padding='same', name='conv8_2',
                kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
        x = Conv2D(128, (kernel, kernel), activation='relu', padding='same', name='conv8_3',
                kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
        x = BatchNormalization()(x)
        x = (UpSampling2D((2,2)))(x)
        x = (Conv2D(64, kernel, activation='relu', padding='same'))(x)
        x = (UpSampling2D((2, 2)))(x)
        x = (Conv2D(32, kernel, activation='relu', padding='same'))(x)

        outputs = Conv2D(2, (1, 1), activation='softmax', padding='same', name='pred')(x)

        model = Model(inputs=input_tensor, outputs=outputs, name="ColorNet")
        return model

    def __res_block(self, inputs, filters=64):
        x = Conv2D(filters, (3, 3), padding='same')(inputs)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(filters, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Add()([x, inputs])
        x = ReLU()(x)

        return x


    def __ResNet34(self):
        img_input = Input(shape=(self.norm_size[0], self.norm_size[1], 1))

        x = Conv2D(64, (7, 7), strides=2, padding='same')(img_input)
        x = MaxPooling2D()(x)

        x = self.__res_block(x, filters=64)
        x = self.__res_block(x, filters=64)
        x = self.__res_block(x, filters=64)

        x = Conv2D(128, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling2D()(x)
        x = Conv2D(128, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = self.__res_block(x, filters=128)
        x = self.__res_block(x, filters=128)
        x = self.__res_block(x, filters=128)

        x = Conv2D(256, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling2D()(x)
        x = Conv2D(256, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = self.__res_block(x, filters=256)
        x = self.__res_block(x, filters=256)
        x = self.__res_block(x, filters=256)
        x = self.__res_block(x, filters=256)
        x = self.__res_block(x, filters=256)

        x = Conv2D(512, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling2D()(x)
        x = Conv2D(512, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = self.__res_block(x, filters=512)
        x = self.__res_block(x, filters=512)

        return Model(img_input, x)