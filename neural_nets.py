
from keras.layers import *
from keras.models import Model
from keras.regularizers import l2
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.optimizers import Adam
from keras import Sequential

class NeuralNets(object):

        def __init__(self, norm_size):
                self.norm_size = norm_size
    
        def model_a(self, kernel = (2,2), sampling = (2,2)):
                input_shape = (self.norm_size[0], self.norm_size[1], 1)
                num_nodes = 64
                l2_reg = None #l2(1e-4)

                model = Sequential()
                model.add(InputLayer(input_shape=input_shape))
                model.add(Conv2D(num_nodes, kernel, activation='relu', padding='same', kernel_regularizer=l2_reg))
                model.add(Conv2D(num_nodes, kernel, activation='relu', padding='same', strides=2, kernel_regularizer=l2_reg))
                model.add(Conv2D(num_nodes*2, kernel, activation='relu', padding='same', kernel_regularizer=l2_reg))
                model.add(UpSampling2D(sampling))
                model.add(Conv2D(num_nodes*2, kernel, activation='relu', padding='same', strides=2, kernel_regularizer=l2_reg))
                model.add(Conv2D(num_nodes*4, kernel, activation='relu', padding='same', kernel_regularizer=l2_reg))
                model.add(Conv2D(num_nodes*4, kernel, activation='relu', padding='same', strides=2, kernel_regularizer=l2_reg))
                model.add(Conv2D(num_nodes*8, kernel, activation='relu', padding='same', kernel_regularizer=l2_reg))
                model.add(Conv2D(num_nodes*4, kernel, activation='relu', padding='same', kernel_regularizer=l2_reg))
                model.add(Conv2D(num_nodes*2, kernel, activation='relu', padding='same', kernel_regularizer=l2_reg))
                model.add(UpSampling2D(sampling))
                model.add(Conv2D(num_nodes, kernel, activation='relu', padding='same', kernel_regularizer=l2_reg))
                model.add(UpSampling2D(sampling))
                model.add(Conv2D(32, kernel, activation='relu', padding='same', kernel_regularizer=l2_reg))
                model.add(Conv2D(2, kernel, activation='sigmoid', padding='same', kernel_regularizer=l2_reg))

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

        def model_d(self, kernel = (2,2)):
                model = Sequential()
                model.add(InputLayer(input_shape=(self.norm_size[0], self.norm_size[1], 1)))
                model.add(Conv2D(64, kernel, activation='relu', padding='same', name='conv_1'))
                model.add(Conv2D(64, kernel, activation='relu', padding='same', name='conv_2', strides=2))
                model.add(Conv2D(128, kernel, activation='relu', padding='same', name='conv_3'))
                model.add(Conv2D(128, kernel, activation='relu', padding='same', name='conv_4', strides=2))
                model.add(Conv2D(256, kernel, activation='relu', padding='same', name='conv_5'))
                model.add(Conv2D(256, kernel, activation='relu', padding='same', name='conv_6', strides=2))
                model.add(Conv2D(512, kernel, activation='relu', padding='same', name='conv_7'))
                model.add(Conv2D(512, kernel, activation='relu', padding='same', name='conv_8', strides = 2))
                model.add(Conv2D(1024, kernel, activation='relu', padding='same', name='conv_9'))
                model.add(Conv2D(512, kernel, activation='relu', padding='same', name='conv_10'))
                model.add(Conv2D(256, kernel, activation='relu', padding='same', name='conv_11'))
                model.add(Conv2D(128, kernel, activation='relu', padding='same', name='conv_12'))
                model.add(Conv2D(64, kernel, activation='relu', padding='same', name='conv_13'))
                model.add(Conv2D(32, kernel, activation='relu', padding='same', name='conv_14'))
                model.add(Conv2D(2, kernel, activation='sigmoid', padding='same', name='conv_15'))
                return model

        def model_e(self):
                model = Sequential()
                model.add(InputLayer(input_shape=(None, None, 1)))
                model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
                model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
                model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
                model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
                model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
                model.add(UpSampling2D((2, 2)))
                model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
                model.add(UpSampling2D((2, 2)))
                model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
                model.add(UpSampling2D((2, 2)))
                model.add(Conv2D(2, (3,3), activation='tanh', padding='same'))
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