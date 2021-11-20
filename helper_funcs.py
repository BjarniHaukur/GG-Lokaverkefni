import os
import shutil
from PIL import ImageEnhance
from PIL import Image
from skimage import color
import numpy as np
from tqdm import tqdm
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

def __prompt():
    val = input("y/n?")
    if val=='y':
        return True
    if val=='n':
        return False
    else:
        __prompt()

def save_model(model, dirName, brave=False):
    root = os.getcwd()
    myDir = os.path.join(root, dirName)
    if os.path.isdir(myDir):
        if not brave: print("Directory already exists, overwrite?")
        if brave or __prompt():
            shutil.rmtree(myDir)
            os.mkdir(myDir)
            model.save(dirName)
        else:
            return
    else:
        os.mkdir(myDir)
        model.save(dirName)


def load_model(dirName):
    root = os.getcwd()
    myDir = os.path.join(root, dirName)
    if os.path.isdir(myDir):
        return keras.models.load_model(dirName)
    else:
        print("No such directory")

def delete_model(dirName):
    root = os.getcwd()
    myDir = os.path.join(root, dirName)
    shutil.rmtree(myDir)

def show_images(X, predictions, enhance=False):
    num_images = predictions.shape[0]
    height = predictions.shape[1]
    width = predictions.shape[2]

    if not X.shape[0]==num_images:
        print("Arrays do not have the same dimensions")
        return
    
    for i in range(num_images):
        canvas = np.zeros((height, width, 3))
        canvas[:,:,0] = X[i,:,:,0]*100
        canvas[:,:,1:] = predictions[i]*128
        canvas = (color.lab2rgb(canvas)*255).astype(np.uint8)
        img = Image.fromarray(canvas)
        if enhance:
            converter = ImageEnhance.Color(img)
            img = converter.enhance(1.5)
        img.show()

def save_images(X, predictions, name="mynd", enhance=False, enumerate=None):
    num_images = predictions.shape[0]
    height = predictions.shape[1]
    width = predictions.shape[2]

    if not X.shape[0]==num_images:
        print("Arrays do not have the same dimensions")
        return

    root = os.getcwd()
    writePath = os.path.join(root, "myndir")
    if not os.path.isdir(writePath):
        os.mkdir(writePath)

    for i in range(num_images):
        canvas = np.zeros((height, width, 3))
        canvas[:,:,0] = X[i,:,:,0]*100
        canvas[:,:,1:] = predictions[i]*128
        canvas = (color.lab2rgb(canvas)*255).astype(np.uint8)
        img = Image.fromarray(canvas)
        if enhance:
            converter = ImageEnhance.Color(img)
            img = converter.enhance(1.5)
        if enumerate:
            img.save(os.path.join(writePath, f"{name}{enumerate+1}_{i+1}.jpg"), 'JPEG') 
        else:
            img.save(os.path.join(writePath, f"{name}{i+1}.jpg"), 'JPEG')

def map_to(arr):
    new_arr = np.zeros(shape=arr.shape, dtype=np.float16)
    for i in range(arr.shape[3]):
        arr_min = np.min(arr[:,:,:,i])
        arr_max = np.max(arr[:,:,:,i])
        new_arr[:,:,:,i] = (arr[:,:,:,i]-arr_min)/(arr_max-arr_min)
    return new_arr

def map_from(arr, old):
    new_arr = np.zeros(shape=arr.shape, dtype=np.float16)
    for i in range(arr.shape[3]):
        old_min = np.min(old[:,:,:,i])
        old_max = np.max(old[:,:,:,i])
        new_arr[:,:,:,i] = arr[:,:,:,i]*(old_max-old_min)+old_min
    return new_arr

# def map_to(arr):
#     func = lambda x: (x+1)/2
#     return func(arr)
#     #np.vectorize(func)(arr)

# def map_from(arr):
#     func = lambda x: 2*x-1
#     return func(arr)