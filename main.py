from sklearn.model_selection import train_test_split
from keras.layers import Input, Conv2D, BatchNormalization, UpSampling2D, Dense, InputLayer
from keras.models import Model
from keras.regularizers import l2
from sklearn.utils import validation
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
from keras import Sequential
import numpy as np
from PIL import Image

from data_loader import MyDataLoader
from helper_funcs import *
from neural_nets import NeuralNets
from skimage import color

from tensorflow.keras.callbacks import EarlyStopping

import tensorflow as tf
import sys


def main():
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu,True)

    norm = (128,128)
    dl = MyDataLoader("combine2", norm)
    nn = NeuralNets(norm)

    #dl.normalize_train_data(norm_size=norm)

    #X, y = dl.get_lab_data()

    #dl.numpy_dump(X, "X")
    #dl.numpy_dump(y, "y")

    X = dl.numpy_load("X")
    y = dl.numpy_load("y")


    X_lab = X
    y_lab = map_to(y)

    X_train, X_test, y_train, y_test = train_test_split(X_lab, y_lab, test_size = 0.3, random_state = 3)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = 0.33, random_state = 3)

    # print(X_lab.shape)
    # print(y_lab.shape)
    # print(np.max(X_lab))
    # print(np.max(y_lab))
    # print(np.min(X_lab))
    # print(np.min(y_lab))

    # decay_schedule = PolynomialDecay(initial_learning_rate=1e-3,
    #                                 decay_steps=30000,
    #                                 end_learning_rate=1e-5)
    model = load_model("model_e")
    #model = nn.model_e()
    model.compile(optimizer=RMSprop(), loss=MeanAbsoluteError())
    # (optimizer=Adam(learning_rate=decay_schedule), loss='mse')
    # model = load_model("model_d")
    

    num_epochs = 30

    # percent = 0.6
    # size = X_train.shape[0]
    # patience = 5
    # callback = EarlyStopping(monitor="val_loss", patience=patience)

    log_dir = "logs/fit/" + "model_e"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

 
    nr = 101
    test_image = X_test[nr]
    test_image = np.expand_dims(test_image, axis=0)
    for i in range(num_epochs):

        model.fit(X_train, y_train, batch_size = 32, epochs = 5, validation_data=(X_val, y_val), verbose=2, callbacks=[tensorboard_callback])#, callbacks=[callback])
        save_model(model, "model_e", brave=True)

        pred = model.predict(test_image)
        save_images(test_image, map_from(pred), name="epoch", enumerate=i)
    

if __name__ == '__main__':
    main()