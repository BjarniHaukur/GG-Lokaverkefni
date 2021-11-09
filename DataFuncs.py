import os
from os import walk
import shutil
import random
import json
from PIL import Image


def resizable(img_size, aspect_ratio, tolerance):
    width = img_size[0]
    height = img_size[1]

    if width >= height:
        return aspect_ratio - tolerance <= width/height <= aspect_ratio + tolerance
    return aspect_ratio - tolerance <= height/width <= aspect_ratio + tolerance

def normalize_train_data(data_dir, grayScale = False, norm_size=(500,500), aspect_ratio = 1.5, tolerance = 0.5):
    rootPath = os.getcwd()

    dataPath = os.path.join(rootPath, data_dir)

    newName = "train"
    newPath = os.path.join(rootPath, newName)
    failPath = os.path.join(newPath, "fail")

    with open('cat_to_name.json') as json_file:
        nameDict = json.load(json_file)

    try:
        shutil.rmtree(newPath)
    except:
        pass

    os.mkdir(newPath)
    os.mkdir(failPath)

    for (_, dirNames, _) in walk(dataPath):

        for category in dirNames:
            readPath = os.path.join(dataPath, category)
            writePath = os.path.join(newPath, category)
            os.mkdir(writePath)

            for (_,_,fileNames) in walk(readPath):

                imgNumber = 1
                for imgName in fileNames:
                    img = Image.open(readPath+"\\"+imgName)
                    
                    if resizable(img.size, aspect_ratio, tolerance):
                        img = img.resize(norm_size)
                        img.save(os.path.join(writePath, nameDict[category]+str(imgNumber)+".jpg"), "JPEG")
                    else:
                        img.save(os.path.join(failPath, nameDict[category]+str(imgNumber)+".jpg"), "JPEG")
                    imgNumber = imgNumber + 1

def train_test_split(ratio_test):
    rootPath = os.getcwd()
    trainDir = os.path.join(rootPath, "train")
    testDir = os.path.join(rootPath, "test")

    try:
        shutil.rmtree(testDir)
    except:
        pass

    os.mkdir(testDir)

    for (_, dirNames, _) in walk(trainDir):
        for dirName in dirNames:
            newTrainDir = os.path.join(trainDir, dirName)
            newTestDir = os.path.join(testDir, dirName)
            os.mkdir(newTestDir)
            for (_, _, filenames) in walk(newTrainDir):
                for name in filenames:
                    if random.random() < ratio_test:
                        oldpath = os.path.join(newTrainDir, name)
                        newpath = os.path.join(newTestDir, name)
                        shutil.move(oldpath, newpath)


