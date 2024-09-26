from array import array
import numpy as np
import math
import pickle
import copy
from typing import List
from numpy.lib.function_base import rot90

from numpy.random.mtrand import randn

class Tensor:
    def from_list(arr, depth, height, width) -> np.array:
        assert(len(arr) == (depth*height*width))
        return np.reshape(arr, (depth, height, width))

    def from_random(depth, height, width) -> np.array:
        n = (depth*height*width)
        arr = np.random.randint(-10000, 10000, n)/20000
        return Tensor.from_list(arr, depth, height, width)

    def make_kernels(depth, height, width, count=1) -> np.array:
        n = depth*height*width*count
        arr = np.random.randint(-10000, 10000, n)/20000
        return np.reshape(arr, (count, depth, height, width))

    def padding(tensor, depth_t, height_t, width_t, p):
        h = 2*p + height_t
        temp = np.zeros((depth_t, h, h))
        for k in range(depth_t):
            for i in range(height_t):
                for j in range(width_t):
                    temp[k][i+p][j+p] = tensor[k][i][j]
        return temp

    def tensor_tensor_corr(tensor, tensor2, out):
        (depth_k, height_k, width_k) = np.shape(tensor2)
        (depth_t, height_t, width_t) = np.shape(tensor)
        (height_o, width_o) = np.shape(out)
        assert(depth_k == depth_t)
        for x in range(height_o):
            for y in range(width_o):
                sum = 0
                for k in range(depth_k):
                    for i in range(height_k):
                        for j in range(width_k):
                            sum += tensor2[k][i][j] * tensor[k][i+x][j+y]
                out[x][y] = sum

    def tensor_matrix_corr(tensor, matrix, out):
        (height_k, width_k) = np.shape(matrix)
        (depth_t, height_t, width_t) = np.shape(tensor)
        (depth_o, height_o, width_o) = np.shape(out)
        for z in range(depth_o):
            for x in range(height_o):
                for y in range(width_o):
                    sum = 0
                    for i in range(height_k):
                        for j in range(width_k):
                            sum += matrix[i][j] * tensor[z][i+x][j+y]
                    out[z][x][y] = sum

    def tensor_kernel_corr(tensor, kernel, out, l):
        (depth_t,height_t, width_t) = np.shape(tensor)
        (n_k, depth_k, height_k, width_k) = np.shape(kernel)
        (height_o, width_o) = np.shape(out)
        #assert(height_t == height_o)
        for x in range(height_o):
            for y in range(width_o):
                sum = 0
                for k in range(n_k):
                    for i in range(height_k):
                        for j in range(width_k):
                            sum += kernel[k][l][i][j] * tensor[k][i+x][j+y]
                out[x][y] = sum

    def rot90(kernels):
        (n_k, depth_k, height_k, width_k) = np.shape(kernels)
        temp = np.zeros((n_k, depth_k, height_k, width_k))
        for l in range(n_k):
            for i in range(depth_k):
                for k in range(height_k):
                    for j in range(width_k):
                        temp[l][i][width_k - j - 1][k] = kernels[l][i][k][j]
        return temp

    def correlation(opr1, opr2, mode, opr='tensor-kernel') -> np.array:
        out = None
        (n_k, depth_k, height_k, width_k) = (0,0,0,0)
        (depth_t, height_t, width_t) = np.shape(opr1)
        assert(height_t == width_t)

        if opr == 'tensor-kernel' or opr == 'kernel-tensor':
            (n_k, depth_k, height_k, width_k) = np.shape(opr2)
        elif opr == 'tensor-tensor':
            (depth_k, height_k, width_k) = np.shape(opr2)

        if mode == 'same':
            p = height_k - 1
            opr1 = Tensor.padding(opr1, depth_t, height_t, width_t, p)
            (depth_t, height_t, width_t) = np.shape(opr1)
        elif mode == 'valid':
            dummy = 0
        else:
            return None

        if opr == 'tensor-kernel':
            #(n_k, depth_k, height_k, width_k) = np.shape(opr2)
            s = height_t - height_k + 1
            out = np.zeros((n_k, s, s))
            for i in range(n_k):
                Tensor.tensor_tensor_corr(opr1, opr2[i], out[i])
        elif opr == 'tensor-tensor':
            #(depth_k, height_k, width_k) = np.shape(opr2)
            s = height_t - height_k + 1
            out = np.zeros((depth_k, depth_t, s, s))
            for i in range(depth_k):
                Tensor.tensor_matrix_corr(opr1, opr2[i], out[i])
        elif opr == 'kernel-tensor':
            #(n_k, depth_k, height_k, width_k) = np.shape(opr2)
            s = height_t - height_k + 1
            out = np.zeros((depth_k, s, s))
            for i in range(depth_k):
                Tensor.tensor_kernel_corr(opr1, opr2, out[i], i)
        return out

    def convolution(tensor, kernels, mode, opr) -> np.array:
        t = Tensor.rot90(kernels)
        t = Tensor.rot90(t)
        return Tensor.correlation(tensor, t, mode, opr)

    def max_pooling(tensor, kernel_size) -> (np.array, List[int]):
        (depth_t, height_t, width_t) = np.shape(tensor)
        assert(height_t == width_t)
        s = height_t - kernel_size + 1
        out = np.zeros((depth_t,s,s))
        mp_coords = []
        for k in range(depth_t):
            for i in range(s):
                for j in range(s):
                    mp = -math.inf
                    mp_coord = []
                    for x in range(kernel_size):
                        for y in range(kernel_size):
                            if mp < tensor[k][i+x][j+y]:
                               mp = tensor[k][i+x][j+y]
                               mp_coord = [i+x, j+y]
                    mp_coords.append(mp_coord)
                    out[k][i][j] = mp
        return out, mp_coords

    def reverse_max_pooling(tensor_in, tensor_out, max_coords):
        (d, h, w) = np.shape(tensor_in)        
        for k in range(d):
            for i in range(h):
                for j in range(w):
                    [x, y] = max_coords[k*d + i*h + j]
                    tensor_out[k][x][y] = tensor_in[k][i][j]

    def relu(tensor) -> np.array:
        (d,h,w) = np.shape(tensor)
        out = [[[tensor[k][i][j] if tensor[k][i][j] > 0 else 0 for j in range(w)] for i in range(h)] for k in range(d)]
        return np.array(out)

    def deriv_relu(tensor) -> np.array:
        (d,h,w) = np.shape(tensor)
        out = [[[1 if tensor[k][i][j] > 0 else 0 for j in range(w)] for i in range(h)] for k in range(d)]
        return np.array(out)

class Layer2D:
    def __init__(self) -> None:
        self.prev_layer  = None
        self.output      = None
        self.input       = None
        self.dinput      = None
        self.doutput     = None
        self.bias        = None
        self.num_channels = 0

    def forward(self):
        pass

    def backward(self, learning_rate):
        pass

    def flatten(self) -> np.array:
        cl = []
        (d,h,w) = np.shape(self.output)
        for k in range(d):
            cl.append(np.reshape(self.output[k], (h*w,)))
        fcl = []
        for k in range(d):
            fcl.extend(cl[k])
        return np.array(fcl)

    def flat_size(self):
        (d,h,w) = np.shape(self.output)
        return d*h*w

    def shape(self):
        return np.shape(self.output)

    def print(self):
        print('@layer2D')
        print(self.output)
        print('@layer2D')

class Input2D(Layer2D):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
        
    def load(self, arr, img_size):
        self.output = Tensor.from_list(arr, self.num_channels, img_size, img_size)

class Conv2D(Layer2D):        
    def __init__(self, num_kernels, kernel_size, prev_layer: Layer2D):
        super().__init__()
        self.kernels = Tensor.make_kernels(prev_layer.num_channels, kernel_size, kernel_size, num_kernels)
        self.prev_layer = prev_layer
        self.num_channels = num_kernels

    def forward(self):
        self.input = self.prev_layer.output
        self.output = Tensor.correlation(self.input, self.kernels, 'valid')
        if type(self.bias) == type(None):
            (d, h, w) = np.shape(self.output)
            self.bias = Tensor.from_random(d, h, w)
        self.output = self.output + self.bias
        self.output = Tensor.relu(self.output)

    def backward(self, learning_rate):
        self.output = Tensor.deriv_relu(self.output)
        self.doutput = self.output * self.doutput
        dfilter = Tensor.correlation(self.input, self.doutput, 'valid', 'tensor-tensor')
        self.dinput = Tensor.convolution(self.doutput, self.kernels, 'same', 'kernel-tensor')
        self.kernels = self.kernels - (learning_rate * dfilter)
        self.bias = self.bias - (learning_rate * self.doutput)
        self.prev_layer.doutput = self.dinput
        

class Maxp2D(Layer2D): 
    def __init__(self, kernel_size, prev_layer: Layer2D) -> None:
        super().__init__()
        self.mp_coords = []
        self.num_channels = prev_layer.num_channels
        self.prev_layer = prev_layer   
        self.kernel_size = kernel_size

    def forward(self):
        self.mp_coords = []
        self.input = self.prev_layer.output
        self.output, self.mp_coords = Tensor.max_pooling(self.input, self.kernel_size)

    def backward(self, learning_rate):
        self.dinput = np.zeros(np.shape(self.prev_layer.output))
        Tensor.reverse_max_pooling(self.doutput, self.dinput, self.mp_coords)
        self.prev_layer.doutput = self.dinput

class Layer1D:
    def __init__(self, prev_layer, num_neurons, actv_func, deactv_func) -> None:
        self.prev_layer = prev_layer
        self.num_neurons = num_neurons
        if prev_layer.num_neurons > 0:
           t = prev_layer.num_neurons * num_neurons
           self.weights = np.reshape(self.randn(t), (num_neurons,prev_layer.num_neurons))
           self.dweights = copy.deepcopy(self.weights)
        else:
            self.weights = None
            self.dweights = None

        self.biases = self.randn(num_neurons)
        self.outputs = self.randn(num_neurons)
        self.doutputs = self.randn(num_neurons)
        self.dinputs = copy.deepcopy(prev_layer.doutputs)
        self.inputs = None

        self.actv_func = actv_func
        self.deactv_func = deactv_func

    def randn(self, n):
        return np.random.randint(0, 10000, n)/10000

    def activation(self):
        self.outputs = self.actv_func(self.outputs)

    def deactivation(self):
        self.outputs = self.deactv_func(self.outputs, self.doutputs)

    def forward(self):
        self.inputs = np.array(self.prev_layer.outputs)
        self.outputs = np.matmul(self.weights,self.inputs) + self.biases
        self.activation()

    def backward(self, learning_rate):
        self.dinputs = self.prev_layer.doutputs
        self.deactivation()
        d = self.outputs
        for i in range(self.num_neurons):
            self.dweights[i] = d[i] * self.inputs
        self.dinputs = np.matmul(d, self.weights)
        self.biases -= learning_rate * self.doutputs
        self.weights -= learning_rate * self.dweights
        #self.prev_layer.doutputs = self.dinputs

    def print(self):
        pass

class Bridge1D(Layer1D):
    def __init__(self, prev_layer: Layer2D) -> None:
        #super().__init__(None, None)
        self.prev_layer = prev_layer
        self.num_neurons = 0
        self.doutputs = None
        self.dinputs = None

    def forward(self):
        if type(self.doutputs) == type(None):
            self.num_neurons = self.prev_layer.flat_size()
            self.doutputs = self.randn(self.num_neurons)
        self.outputs = self.prev_layer.flatten()

    def backward(self, learning_rate):
        s = self.prev_layer.shape()
        self.dinputs = np.reshape(self.doutputs, s)
        self.prev_layer.doutput = self.dinputs

class Input1D(Layer1D):
    def __init__(self, num_neurons) -> None:
        self.num_neurons = num_neurons
        self.outputs = []
        self.doutputs = []

    def load(self, data):
        self.outputs = np.array(data)

    def forward(self):
        pass
        
    def backward(self, learning_rate):
        pass

class Dense1D(Layer1D):
    def __init__(self, num_neurons: int, prev_layer: Layer1D, actv_func, deactv_func) -> None:
        super().__init__(prev_layer, num_neurons, actv_func, deactv_func)

    def forward(self):
        if type(self.weights) == type(None):
           t = self.prev_layer.num_neurons * self.num_neurons
           self.weights = np.reshape(self.randn(t), (self.num_neurons,self.prev_layer.num_neurons))
           self.dweights = copy.deepcopy(self.weights)
        super().forward()
        
    def backward(self, learning_rate):
        super().backward(learning_rate)

class Error1D(Layer1D):
    def __init__(self, num_neurons: int, prev_layer: Layer1D, actv_func, deactv_func) -> None:
        super().__init__(prev_layer, num_neurons, actv_func, deactv_func)       
        self.expected_outputs = []
        self.total_cost = 0

    def load(self, data):
        self.expected_outputs = np.array(data)

    def forward(self):
        super().forward()        

    def backward(self, learning_rate):
        self.doutputs = self.outputs - self.expected_outputs
        print(f"Expected = {self.expected_outputs}")
        print(f"Calculated = {self.outputs}")
        self.cost()
        print('\n')
        super().backward(learning_rate)

    def cost(self):
        t = (self.doutputs * self.doutputs) / 2  
        self.total_cost = (self.total_cost + np.sum(t)) / self.num_neurons
        print(t)

def relu(x):
    return np.array([max(0, e) for e in x])

def deriv_relu(y, dout):
    return np.multiply(np.array([1 if(e > 0) else 0 for e in y]), dout)

def sigmoid(x):
    #print(f'Inside Softmax x = {x}')
    x = 1 / (1 + np.exp(-x))
    return x

def deriv_sigmoid(y, dout):
    y = y * (1 - y)
    return np.multiply(y, dout)

def softmax(x):
    e = np.exp(x - np.max(x ,keepdims = True ))
    p = e / np.sum(e , keepdims = True )
    return p

def deriv_softmax(x, dout):
    y = np.diagflat(x)
    z = np.array(x).reshape(-1, 1)#np.dot(np.array(x).T, x)
    x = np.array(x).reshape(1, -1)
    z = np.matmul(z, x)
    jacobian = y - z
    return np.dot(jacobian, dout)

class NeuNet:
    def __init__(self, layers: List[int]) -> None:
        self.layers = []
        curr_lay = Input1D(layers[0])
        self.layers.append(curr_lay)
        hidden_lays = layers[1:-1]
        for hidden in hidden_lays:
            curr_lay = Dense1D(hidden, curr_lay, relu, deriv_relu)
            self.layers.append(curr_lay)
        curr_lay = Error1D(layers[-1], curr_lay, softmax, deriv_softmax)
        self.layers.append(curr_lay)

    def train(self, data, label, learning_rate):
        (n,_) = np.shape(data)
        in_lay = self.layers[0]
        out_lay = self.layers[-1]
        for i in range(40000):
            for j in range(n):
                in_lay.load(data[j])
                out_lay.load(label[j])
                for x in np.arange(len(self.layers)):
                    self.layers[x].forward()
                for x in np.arange(len(self.layers)-1,-1,-1):
                    self.layers[x].backward(learning_rate)

    def test(self, data):
        label = []
        for d in data:
            self.layers[0].load(d)
            for x in np.arange(len(self.layers)):
                self.layers[x].forward()
            label.append(self.layers[-1].outputs)
        return label



class CNN:
    def __init__(self, layers: List[Layer2D]) -> None:
        self.layers = layers
        self.one_hot = lambda n: [1 if(n==i) else 0 for i in range(10)]

    def train(self, data, label, learning_rate):
        (n,_) = np.shape(data)
        in_lay = self.layers[0]
        out_lay = self.layers[-1]
        for i in range(1):
            for j in range(n):
                in_lay.load(data[j], 32)
                out_lay.load(self.one_hot(label[j]))
                for x in np.arange(len(self.layers)):
                    self.layers[x].forward()
                for x in np.arange(len(self.layers)-1,-1,-1):
                    self.layers[x].backward(learning_rate)

    def test(self, data):
        label = []
        for d in data:
            self.layers[0].load(d)
            for x in np.arange(len(self.layers)):
                self.layers[x].forward()
            label.append(self.layers[-1].outputs)
        return label

    