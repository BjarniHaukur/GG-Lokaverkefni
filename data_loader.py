import os
from os import walk
import shutil
import json
import numpy as np
from PIL import Image



class MyDataLoader(object):   

    def __init__(self, dirName, jsonName="cat_to_name.json", trainName="train"):
        self.dirName = dirName
        self.jsonName = jsonName
        self.trainName = trainName

        self.rootPath  = os.getcwd()
        self.dataPath  = os.path.join(self.rootPath, self.dirName)

        self.trainPath = os.path.join(self.rootPath, self.trainName)
        self.XPath = os.path.join(self.trainPath, "X")
        self.yPath = os.path.join(self.trainPath, "y")
        
        self.failPath  = os.path.join(self.trainPath, "fail")

        with open(self.jsonName) as json_file:
            self.nameDict = json.load(json_file)
        
        self.norm_size = (500,500)

    def normalize_train_data(self, norm_size=(500,500), aspect_ratio = 1.5, tolerance = 0.5):

        self.norm_size = norm_size

        try:
            shutil.rmtree(self.trainPath)
        except:
            pass

        os.mkdir(self.trainPath)
        os.mkdir(self.yPath)
        os.mkdir(self.failPath)

        for (_, dirNames, _) in walk(self.dataPath):

            for category in dirNames:
                readPath = os.path.join(self.dataPath, category)
                writePath = os.path.join(self.yPath, category)
                os.mkdir(writePath)

                for (_,_,fileNames) in walk(readPath):

                    imgNumber = 1
                    for imgName in fileNames:
                        img = Image.open(readPath+"\\"+imgName)
                        if self.__resizable(img.size, aspect_ratio, tolerance):
                            img = img.resize(norm_size)
                            img.save(os.path.join(writePath, self.nameDict[category]+str(imgNumber)+".jpg"), "JPEG")
                        else:
                            img.save(os.path.join(self.failPath, self.nameDict[category]+str(imgNumber)+".jpg"), "JPEG")
                        imgNumber = imgNumber + 1

    def make_colorization_data(self):
        if os.path.isdir(self.yPath):
            os.mkdir(self.XPath)
            for (_, dirNames, _) in walk(self.yPath):

                for category in dirNames:
                    readPath = os.path.join(self.yPath, category)
                    writePath = os.path.join(self.XPath, category)
                    os.mkdir(writePath)

                    for (_,_,fileNames) in walk(readPath):

                        for imgName in fileNames:
                            img = Image.open(readPath+"\\"+imgName).convert('L')
                            img.save(os.path.join(writePath, imgName), "JPEG")
        else:
            print("No such directory of normalized images. Have you executed normalize_train_data?")
    


    def get_classification_data(self):                 
        return self.__read_classification(self.__dir_size(self.trainPath))   


    def get_colorization_data(self):                  
        return self.__read_colorization(self.__dir_size(self.trainPath))    

    
    def __resizable(self, img_size, aspect_ratio, tolerance):
        width = img_size[0]
        height = img_size[1]

        if width >= height:
            return aspect_ratio - tolerance <= width/height <= aspect_ratio + tolerance
        return aspect_ratio - tolerance <= height/width <= aspect_ratio + tolerance

    def __dir_size(self, path):
        if(os.path.isdir(path)):
            size = 0
            for (_, dirNames, _) in walk(path):
                for dirName in dirNames:
                    if dirName == "fail": continue
                    readPath = os.path.join(self.trainPath, dirName)
                    for (_, _, fileNames) in walk(readPath):
                        size = size + len(fileNames)
            return size-1
        else:
            print("Not a valid path to directory")
            return 0

    def __read_classification(self, size):
        X = np.ndarray(shape=(size, self.norm_size[0], self.norm_size[1]), dtype=np.uint8)
        y = np.ndarray(shape=size)
        iter = 0
        for (_, dirNames, _) in walk(self.XPath):
            for dirName in dirNames:
                readPath = os.path.join(self.XPath, dirName)
                for (_, _, fileNames) in walk(readPath):
                    for name in fileNames:
                        img = Image.open(readPath+"\\"+name)
                        X[iter] = np.ndarray(img, dtype=np.uint8)
                        y[iter] = dirName
                        iter = iter + 1
        return X, y
    
    def __read_colorization(self, size):
        X = np.ndarray(shape=(size, self.norm_size[0], self.norm_size[1]), dtype=np.uint8)
        y = np.ndarray(shape=(size, self.norm_size[0], self.norm_size[1], 3), dtype=np.uint8)
        iter = 0
        for (_, dirNames, _) in walk(self.XPath):
            for dirName in dirNames:
                if dirName=="fail": continue
                XreadPath = os.path.join(self.XPath, dirName)
                yreadPath = os.path.join(self.yPath, dirName)

                tempIter = iter
                for (_, _, fileNames) in walk(XreadPath):
                    for name in fileNames:
                        img = Image.open(XreadPath+"\\"+name)
                        X[tempIter] = np.ndarray(img, dtype=np.uint8)
                        tempIter = tempIter + 1

                tempIter = iter
                for (_, _, fileNames) in walk(yreadPath):
                    for name in fileNames:
                        img = Image.open(yreadPath+"\\"+name)
                        y[tempIter] = np.ndarray(img, dtype=np.uint8)
                        tempIter = tempIter + 1
                iter = tempIter
        return X, y

    

# def train_test_val_split(self, ratio_test):
    #     try:
    #         shutil.rmtree(self.testPath)
    #     except:
    #         pass

    #     os.mkdir(self.testPath)

    #     for (_, dirNames, _) in walk(self.trainPath):
    #         for dirName in dirNames:
    #             newTrainDir = os.path.join(self.trainPath, dirName)
    #             newTestDir = os.path.join(self.testPath, dirName)
    #             os.mkdir(newTestDir)

    #             for (_, _, fileNames) in walk(newTrainDir):
    #                 for name in fileNames:
    #                     if random.random() < ratio_test:
    #                         oldPath = os.path.join(newTrainDir, name)
    #                         newPath = os.path.join(newTestDir, name)
    #                         shutil.move(oldPath, newPath)