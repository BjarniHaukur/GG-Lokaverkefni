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
from matplotlib import pyplot as plt

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
    modelDir = os.path.join(root, "models")

    if not os.path.isdir(modelDir):
        os.mkdir(modelDir)

    myDir = os.path.join(modelDir, dirName)

    if os.path.isdir(myDir):
        if not brave: print("Directory already exists, overwrite?")
        if brave or __prompt():
            shutil.rmtree(myDir)
            os.mkdir(myDir)
            model.save(myDir)
        else:
            return
    else:
        os.mkdir(myDir)
        model.save(myDir)


def load_model(dirName, custom=None):
    root = os.getcwd()
    modelDir = os.path.join(root, "models")
    myDir = os.path.join(modelDir, dirName)

    if os.path.isdir(myDir):
        return keras.models.load_model(myDir, custom_objects=custom)
    else:
        print("No such directory")

def delete_model(dirName):
    root = os.getcwd()
    modelDir = os.path.join(root, "models")
    myDir = os.path.join(modelDir, dirName)
    if os.path.isdir(myDir):
        shutil.rmtree(myDir)
    else:
        print("No such directory")
    

def show_images(X, predictions, enhance=0):
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
            img = converter.enhance(enhance)
        img.show()

def save_images(X, predictions, name="mynd", enhance=0, enumerate=None):
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
            img = converter.enhance(enhance)
        if enumerate:
            img.save(os.path.join(writePath, f"{name}{enumerate+1}_{i+1}.jpg"), 'JPEG') 
        else:
            img.save(os.path.join(writePath, f"{name}{i+1}.jpg"), 'JPEG')

def plot_acc_and_loss(history_arr, title, name, save=False, figsize=(5, 10), brave=False):
    acc, val_acc, loss, val_loss = __data_from_history_arr(history_arr)
    fig, axs = plt.subplots(2, 1, figsize=figsize)
    axs[0].plot(acc, label="accuracy")
    axs[0].plot(val_acc, label="validation accuracy")
    axs[1].plot(loss, label="loss")
    axs[1].plot(val_loss, label="validation loss")

    axs[0].legend()
    axs[1].legend()
    
    axs[0].set(ylabel="Accuracy")
    axs[1].set(ylabel="Loss")

    fig.suptitle(title, fontsize=20)
    plt.xlabel("Epochs", fontsize=12)
    
    if save:
        root = os.getcwd()
        plotDir = os.path.join(root, "plot")

        if not os.path.isdir(plotDir):
            os.mkdir(plotDir)

        writePath = os.path.join(plotDir, f'{name}.png')

        if os.path.isfile(writePath):
            if not brave: print("Plot already exists, overwrite?")
            if brave or __prompt():
                fig.savefig(writePath)
            else:
                return
        else:
            fig.savefig(writePath)
    


def __data_from_history_arr(history_arr):
    """
    acc, val_acc, loss, val_loss
    """
    val_acc = []
    val_loss = []
    acc = []
    loss = []
    for x in history_arr:
        if x == None: continue
        val_acc = val_acc + x.history["val_accuracy"]
        val_loss = val_loss + x.history["val_loss"]
        acc = acc + x.history["accuracy"]
        loss = loss + x.history["loss"]
    return acc, val_acc, loss, val_loss


def map_to(arr):
    return (arr+1)/2
    #np.vectorize(func)(arr)

def map_from(arr):
    return arr*2-1

# def map_to(arr):
#     new_arr = np.zeros(shape=arr.shape, dtype=np.float16)
#     for i in range(arr.shape[3]):
#         arr_min = np.min(arr[:,:,:,i])
#         arr_max = np.max(arr[:,:,:,i])
#         new_arr[:,:,:,i] = (arr[:,:,:,i]-arr_min)/(arr_max-arr_min)
#     return new_arr

# def map_from(arr, old):
#     new_arr = np.zeros(shape=arr.shape, dtype=np.float16)
#     for i in range(arr.shape[3]):
#         old_min = np.min(old[:,:,:,i])
#         old_max = np.max(old[:,:,:,i])
#         new_arr[:,:,:,i] = arr[:,:,:,i]*(old_max-old_min)+old_min
#     return new_arr

