# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras import layers
# from tensorflow.keras.models import Sequential


# class ImageClassifier(object):
#     """
#     A two-layer fully-connected neural network. The net has an input dimension of
#     N, a hidden layer dimension of H, and performs classification over C classes.
#     We train the network with a softmax loss function and L2 regularization on the
#     weight matrices. The network uses a ReLU nonlinearity after the first fully
#     connected layer.

#     In other words, the network has the following architecture:

#     input - fully connected layer - ReLU - fully connected layer - softmax

#     The outputs of the second fully-connected layer are the scores for each class.
#     """

#     def __init__(self, input_size, hidden_size, output_size, std=1e-4):
#         """
#         Initialize the model. Weights are initialized to small random values and
#         biases are initialized to zero. Weights and biases are stored in the
#         variable self.params, which is a dictionary with the following keys:

#         W1: First layer weights; has shape (D, H)
#         b1: First layer biases; has shape (H,)
#         W2: Second layer weights; has shape (H, C)
#         b2: Second layer biases; has shape (C,)

#         Inputs:
#         - input_size: The dimension D of the input data.
#         - hidden_size: The number of neurons H in the hidden layer.
#         - output_size: The number of classes C.
#         """
#         self.params = {}
#         self.params['W1'] = std * np.random.randn(input_size, hidden_size)
#         self.params['b1'] = np.zeros(hidden_size)
#         self.params['W2'] = std * np.random.randn(hidden_size, output_size)
#         self.params['b2'] = np.zeros(output_size)

# def train_test_split(ratio_test):
#     rootPath = os.getcwd()
#     trainDir = os.path.join(rootPath, "train")
#     testDir = os.path.join(rootPath, "test")

#     try:
#         shutil.rmtree(testDir)
#     except:
#         pass

#     os.mkdir(testDir)

#     for (_, dirNames, _) in walk(trainDir):
#         for dirName in dirNames:
#             newTrainDir = os.path.join(trainDir, dirName)
#             newTestDir = os.path.join(testDir, dirName)
#             os.mkdir(newTestDir)
#             for (_, _, filenames) in walk(newTrainDir):
#                 for name in filenames:
#                     if random.random() < ratio_test:
#                         oldpath = os.path.join(newTrainDir, name)
#                         newpath = os.path.join(newTestDir, name)
#                         shutil.move(oldpath, newpath)