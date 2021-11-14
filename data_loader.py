import os
from os import walk
import shutil
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage import color


class MyDataLoader(object):   

    def __init__(self, dirName, norm_size, jsonName="cat_to_name.json", trainName="train", dumpName="arrays"):
        self.dirName = dirName
        self.jsonName = jsonName
        self.trainName = trainName
        self.norm_size = norm_size
        self.dumpName = dumpName

        self.rootPath  = os.getcwd()
        self.dataPath  = os.path.join(self.rootPath, self.dirName)

        self.trainPath = os.path.join(self.rootPath, self.trainName)
        self.XPath = os.path.join(self.trainPath, "X")
        self.failPath  = os.path.join(self.trainPath, "fail")
        self.yPath = os.path.join(self.trainPath, "y")

        self.dumpPath = os.path.join(self.rootPath, dumpName)
        

        with open(self.jsonName) as json_file:
            self.nameDict = json.load(json_file)
        
        

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
            if dirNames==[]: continue
            for category in tqdm(dirNames):
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

    def make_grayscale_data(self):
        if os.path.isdir(self.yPath):
            os.mkdir(self.XPath)
            for (_, dirNames, _) in walk(self.yPath):
                if dirNames==[]: continue
                for category in tqdm(dirNames):
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
        return self.__read_classification(self.__dir_size(self.XPath))   


    def get_rgb_data(self):                  
        return self.__read_colorization(self.__dir_size(self.XPath))

    def get_lab_data(self):
        lab = self.__read_lab(self.__dir_size(self.yPath))
        X = np.expand_dims(lab[:,:,:,0], -1)
        y = lab[:,:,:,1:]
        return X, y

    def numpy_dump(self, arr, name):
        if not os.path.isdir(self.dumpPath):
            os.mkdir(self.dumpPath)
        
        arrPath = os.path.join(self.dumpPath, name)
        try:
            os.rmdir(arrPath)
        except:
            pass

        np.save(arrPath, arr)
    
    def numpy_load(self, name):
        if os.path.isdir(self.dumpPath):
            read = os.path.join(self.dumpPath, name)
            if os.path.isfile(read):
                return np.load(read)
            read = os.path.join(self.dumpPath, name+".npy")
            if os.path.isfile(read):
                return np.load(read)
            else:
                print("No such file")
        else:
            print("No such directory")
        

    
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
                    readPath = os.path.join(path, dirName)
                    for (_, _, fileNames) in walk(readPath):
                        size = size + len(fileNames)
            return size
        else:
            print("Not a valid path to directory")
            return 0

    def __read_lab(self, size):
        lab = np.ndarray(shape=(size, self.norm_size[0], self.norm_size[1], 3), dtype=np.int8)
        iter = 0
        for (_, dirNames, _) in walk(self.yPath):
            if dirNames==[]: continue
            for dirName in tqdm(dirNames):
                readPath = os.path.join(self.yPath, dirName)
                for (_, _, fileNames) in walk(readPath):
                    for name in fileNames:
                        img = color.rgb2lab(Image.open(readPath+"\\"+name))
                        lab[iter] = np.array(img, dtype=np.int8)
                        iter = iter + 1
        return lab


    def __read_classification(self, size):
        X = np.ndarray(shape=(size, self.norm_size[0], self.norm_size[1]), dtype=np.uint8)
        y = np.ndarray(shape=size)
        iter = 0
        for (_, dirNames, _) in walk(self.XPath):
            if dirNames==[]: continue
            for dirName in tqdm(dirNames):
                readPath = os.path.join(self.XPath, dirName)
                for (_, _, fileNames) in walk(readPath):
                    for name in fileNames:
                        img = Image.open(readPath+"\\"+name)
                        X[iter] = np.array(img, dtype=np.uint8)
                        y[iter] = dirName
                        iter = iter + 1
        return X, y
    
    def __read_rgb(self, size):
        X = np.ndarray(shape=(size, self.norm_size[0], self.norm_size[1]), dtype=np.uint8)
        y = np.ndarray(shape=(size, self.norm_size[0], self.norm_size[1], 3), dtype=np.uint8)
        iter = 0
        for (_, dirNames, _) in walk(self.XPath):
            if dirNames==[]: continue
            for dirName in tqdm(dirNames):
                if dirName=="fail": continue
                XreadPath = os.path.join(self.XPath, dirName)
                yreadPath = os.path.join(self.yPath, dirName)

                tempIter = iter
                for (_, _, fileNames) in walk(XreadPath):
                    for name in fileNames:
                        img = Image.open(XreadPath+"\\"+name)
                        X[tempIter] = np.array(img, dtype=np.uint8)
                        tempIter = tempIter + 1

                tempIter = iter
                for (_, _, fileNames) in walk(yreadPath):
                    for name in fileNames:
                        img = Image.open(yreadPath+"\\"+name)
                        y[tempIter] = np.array(img, dtype=np.uint8)
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