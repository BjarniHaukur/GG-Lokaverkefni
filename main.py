from sklearn.model_selection import train_test_split
from keras.layers import Input, Conv2D, BatchNormalization, UpSampling2D, Dense, InputLayer
from keras.models import Model
from keras.regularizers import l2
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.optimizers import Adam
from keras import Sequential
import numpy as np
from PIL import Image

from data_loader import MyDataLoader
from nn_helper import save_model, load_model
from neural_nets import NeuralNets
from skimage import color

import tensorflow as tf
import sys


def main():

    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.8
    # tf.Session(config=config)


    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu,True)

    norm = (128,128)
    dl = MyDataLoader("combine2", norm)
    nn = NeuralNets(norm)

    X = dl.numpy_load("X")
    y = dl.numpy_load("y")

    X_lab = X/100
    y_lab = y/128

    X_train, X_test, y_train, y_test = train_test_split(X_lab, y_lab, test_size = 0.3, random_state = 3)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = 0.33, random_state = 3)

    # print(X_lab.shape)
    # print(y_lab.shape)
    # print(np.max(X_lab))
    # print(np.max(y_lab))
    # print(np.min(X_lab))
    # print(np.min(y_lab))


    model = nn.model_c()
    model.compile(optimizer=Adam(1e-3), loss='msle')

    # model = load_model("model_c")
    

    num_epochs = 20

    percent = 0.6
    size = X_train.shape[0]

    canvas = np.zeros((norm[0],norm[1],3))
    nr = 69
    test_image = X_test[nr]
    test_image = np.expand_dims(test_image, axis=0)
    for i in range(num_epochs):
        canvas = np.zeros((norm[0],norm[1],3))

        index_array = np.random.choice(size, int(size*percent))
        X_batch = X_train[index_array]
        y_batch = y_train[index_array]

        model.fit(X_batch, y_batch, batch_size = 32, epochs = 5)
        save_model(model, "model_c", brave=True)

        pred = model.predict(test_image)
        canvas[:,:,0] = test_image[0,:,:,0]*100
        canvas[:,:,1:] = pred*128
        canvas = color.lab2rgb(canvas)*255
        canvas = canvas.astype(np.uint8)
        img = Image.fromarray(canvas)
        img.save(f'mynd{i}.png')
    
    true_image = np.zeros((norm[0],norm[1],3))
    true_image[:,:,0] = X_test[nr,:,:,0]*100
    true_image[:,:,1:] = y_test[nr,:,:,1:]*128
    true_image = color.lab2rgb(true_image)*255
    true_image = true_image.astype(np.uint8)
    true_image = Image.fromarray(true_image)
    true_image.save("true.png")


if __name__ == '__main__':
    main()