import os
import shutil
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

def save_model(model, dirName):
    root = os.getcwd()
    myDir = os.path.join(root, dirName)
    if os.path.isdir(myDir):
        print("Directory already exists, overwrite?")
        if __prompt():
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

#def image_plot():