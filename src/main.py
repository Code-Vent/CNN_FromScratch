from array import array
import numpy as np
import math
import pickle
import copy
from typing import List
from numpy.lib.function_base import rot90

from numpy.random.mtrand import randn
from nnet import *

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

train_dataset_batch = unpickle('./cifar-10-batches-py/data_batch_1')
train_data = train_dataset_batch[b'data'] / 255
train_labels = train_dataset_batch[b'labels']

input2d = Input2D(3)
conv2d = Conv2D(8, 3, input2d)        
conv2d2 = Conv2D(6, 2, conv2d)
maxp2d = Maxp2D(2, conv2d2)
bridge1d = Bridge1D(maxp2d)
dense1d = Dense1D(10, bridge1d, relu, deriv_relu)
error1d = Error1D(10, dense1d, softmax, deriv_softmax)
layers = [input2d, conv2d, conv2d2, maxp2d, bridge1d, dense1d, error1d]
cnn = CNN(layers)
cnn.train(train_data, train_labels, 0.01)

