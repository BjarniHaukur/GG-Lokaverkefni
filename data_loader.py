import os
from os import walk
import shutil
import random
import json
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
        

    def __resizable(self, img_size, aspect_ratio, tolerance):
        width = img_size[0]
        height = img_size[1]

        if width >= height:
            return aspect_ratio - tolerance <= width/height <= aspect_ratio + tolerance
        return aspect_ratio - tolerance <= height/width <= aspect_ratio + tolerance


    def normalize_train_data(self, norm_size=(500,500), aspect_ratio = 1.5, tolerance = 0.5):
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
                for (_, _, filenames) in walk(newTrainDir):
                    for name in filenames:
                        if random.random() < ratio_test:
                            oldPath = os.path.join(newTrainDir, name)
                            newPath = os.path.join(newTestDir, name)
                            shutil.move(oldPath, newPath)

    

