from data_loader import MyDataLoader
from sklearn.model_selection import train_test_split
from keras.layers import Input, Conv2D, BatchNormalization, UpSampling2D, Dense, InputLayer
from keras.models import Model
from keras.regularizers import l2
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.optimizers import Adam
from keras import Sequential
from nn_helper import save_model, load_model
import numpy as np
from PIL import Image
import os

print()
print(os.getcwd())
print()

dl = MyDataLoader("combine2", (128,128))


X, y = dl.get_lab_data()


X_lab = X/100
y_lab = y/128



X_train, X_test, y_train, y_test = train_test_split(X_lab, y_lab, test_size = 0.3, random_state = 3)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = 0.33, random_state = 3)



input_shape = (128, 128, 1)
kernel = (2, 2)

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

model.compile(optimizer='rmsprop', loss='mse')

# model = load_model("model_a")


num_epochs = 10
nr = 42
canvas = np.zeros((128,128,3))
for i in range(num_epochs):
    model.fit(X_train, y_train, batch_size = 64, epochs = 10)
    save_model(model, "model_a", brave=True)
