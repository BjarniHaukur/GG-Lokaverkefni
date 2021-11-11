import os
from os import walk
import shutil
import random
import json
import numpy as np
from PIL import Image



class MyDataLoader(object):   

    def __init__(self, dirName, jsonName="cat_to_name.json", trainName="train", testName="test"):
        self.dirName = dirName
        self.jsonName = jsonName
        self.trainName = trainName
        self.testName = testName

        self.rootPath  = os.getcwd()
        self.dataPath  = os.path.join(self.rootPath, self.dirName)
        self.trainPath = os.path.join(self.rootPath, self.trainName)
        self.failPath  = os.path.join(self.trainPath, "fail")
        self.testPath  = os.path.join(self.rootPath, self.testName)

        with open(self.jsonName) as json_file:
            self.nameDict = json.load(json_file)
        
        # TODO 1 ekki gera svona (fer eftir röð aðgerða hvort þetta virki)
        self.numTrainImages = 0
        self.numTestImages = 0

        self.norm_size = 0
        

    def __resizable(self, img_size, aspect_ratio, tolerance):
        width = img_size[0]
        height = img_size[1]

        if width >= height:
            return aspect_ratio - tolerance <= width/height <= aspect_ratio + tolerance
        return aspect_ratio - tolerance <= height/width <= aspect_ratio + tolerance


    def normalize_train_data(self, norm_size=(500,500), aspect_ratio = 1.5, tolerance = 0.5):
        self.norm_size=norm_size

        try:
            shutil.rmtree(self.trainPath)
        except:
            pass

        os.mkdir(self.trainPath)
        os.mkdir(self.failPath)

        for (_, dirNames, _) in walk(self.dataPath):

            for category in dirNames:
                readPath = os.path.join(self.dataPath, category)
                writePath = os.path.join(self.trainPath, category)
                os.mkdir(writePath)

                for (_,_,fileNames) in walk(readPath):

                    imgNumber = 1
                    for imgName in fileNames:
                        img = Image.open(readPath+"\\"+imgName).convert('L')
                        if self.__resizable(img.size, aspect_ratio, tolerance):
                            img = img.resize(norm_size)
                            img.save(os.path.join(writePath, self.nameDict[category]+str(imgNumber)+".jpg"), "JPEG")
                        else:
                            img.save(os.path.join(self.failPath, self.nameDict[category]+str(imgNumber)+".jpg"), "JPEG")
                        imgNumber = imgNumber + 1

    def train_test_split(self, ratio_test):
        try:
            shutil.rmtree(self.testPath)
        except:
            pass

        os.mkdir(self.testPath)

        for (_, dirNames, _) in walk(self.trainPath):
            for dirName in dirNames:
                newTrainDir = os.path.join(self.trainPath, dirName)
                newTestDir = os.path.join(self.testPath, dirName)
                os.mkdir(newTestDir)

                for (_, _, fileNames) in walk(newTrainDir):
                    for name in fileNames:
                        if random.random() < ratio_test:
                            oldPath = os.path.join(newTrainDir, name)
                            newPath = os.path.join(newTestDir, name)
                            shutil.move(oldPath, newPath)
                            # TODO 1
                            self.numTestImages = self.numTestImages + 1
                        else:
                            # TODO 1
                            self.numTrainImages = self.numTrainImages + 1

    def __read_to_arrays(self, path, size):
        X = np.ndarray(shape=(size, self.norm_size[0], self.norm_size[1]))
        y = np.ndarray(shape=size)
        iter = 0
        for (_, dirNames, _) in walk(path):
            for dirName in dirNames:
                if dirName=="fail": continue
                readPath = os.path.join(path, dirName)
                for (_, _, fileNames) in walk(readPath):
                    for name in fileNames:
                        img = Image.open(readPath+"\\"+name)
                        X[iter] = np.array(img)
                        y[iter] = dirName
                        iter = iter + 1
        return X, y

    def to_numpy_array(self):
        X_train, y_train = self.__read_to_arrays(self.trainPath, self.numTrainImages)                
        X_test, y_test = self.__read_to_arrays(self.testPath, self.numTestImages)        
        return X_train, y_train, X_test, y_test

    

