# Imports
import json
from json import JSONEncoder
import numpy as np
from enum import Enum
import re

# Encoder
class Encoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Layer):
            return obj.value[0]
        return JSONEncoder.default(self, obj)


# Layer definitions
def conv_layer(input, f):
    """
    Evaluate the output of a convolutional layer using the filter f.
    input.shape = (h, w, c)
    f.shape = (c_out, hf, wf, c_in)
    """
    h, w, c = input.shape
    c_out, hf, wf, c_in = f.shape

    assert c == c_in, "Input channels must match!"
    assert hf%2 == 1, "Height of the filter (f.shape[1]) must be an uneven number!"
    assert wf%2 == 1, "Width of the filter (f.shape[2]) must be an uneven number!"

    # unilateral width and heght
    dh = hf//2
    dw = wf//2

    # after convolution dw and dh get substracted from all sides of the image, c_out is number of convolutions which dictates # of channels
    # initialize matrix with 0s
    output = np.zeros(shape=(h-2*dh, w-2*dw, c_out))

# run convolution
# go over image height - kernel padding (2*dh)
    for i in range(dh, h-dh):
        # go over image width - kernel padding (2*dw)
        for j in range(dw, w-dw):
            # kernel slice
            a = input[i-dh:i+dh+1, j-dw:j+dw+1]
            for k in range(c_out):
                # filter channel 1..c_out
                b = f[k,:,:,:]
                # apply filter
                output[i-dh, j-dw, k] = (a*b).sum() # a.size multiplication
    
    n_params = f.size
    n_multiplications = a.size * c_out * (w-2*dw) * (h-2*dh)
    name = f"conv {'x'.join([str(e) for e in f.shape])}"
    
    return output, n_params, n_multiplications, name


def relu_layer(input):    
    output = input.copy()
    output[output<0] = 0
    
    n_params = 0
    n_multiplications = input.size
    
    return output, n_params, n_multiplications, "relu"


def max_pooling_layer(input, s):
    """
    Apply max pooling layer using a sxs patch.
    """
    h, w, c = input.shape

    assert h%s == 0, "Height must be divisible by s!"
    assert w%s == 0, "Width must be dibisible by s!"

    output = np.zeros(shape=(h//s, w//s, c))

    for i in range(0, h, s):
        for j in range(0, w, s):
            for k in range(c):
                a = input[i:i+s, j:j+s, k]
                output[i//s, j//s, k] = a.max()
    
    n_params = 0
    n_multiplications = input.size
    return output, n_params, n_multiplications, "max-pool"


def flatten_layer(input):
    output = input.flatten()
    n_params = 0
    n_multiplications = 0
    return output, n_params, n_multiplications, "flatten"


def fully_connected_layer(input, weights, biases):
    """
    Evaluate the output of a fully connected layer.
    input.shape = (output_dim)
    weights.shape = (output_dim, input_dim)
    f.shape = (output_dim)
    """
    assert input.ndim == 1, "Input must be a flattend array!"
    assert weights.shape[1] == input.shape[0], "Input shapes must match!"
    assert weights.shape[0] == biases.shape[0], "Output shapes must match!"

    output = np.dot(weights, input) + biases
    
    n_params = weights.size + biases.size
    n_multiplications = weights.size
    name = f"full {'x'.join([str(e) for e in weights.shape])}"
    
    return output, n_params, n_multiplications, name


def normalize(input):
    output = input / np.linalg.norm(input)
    n_params = 0
    n_multiplications = 1 + input.size
    return output, n_params, n_multiplications, "normalize"

############
#      Model    #
############

class Layer(Enum):
    Convolution = 'convolution',
    MaxPool = 'max_pool',
    Relu = 'relu',
    Flatten = 'flatten',
    FullyConnected = 'fully_connected',
    Normalize = 'normalize',


np.random.seed(12345)

p = "{:>20} | {:>15} | {:>15} | {:>15} "
print(p.format("layer", "output shape", "#parameters", "#ops"))
print(p.format("-"*20, "-"*15, "-"*15, "-"*15))


shape = (120,80,3)

# input
x = np.random.randint(low=-5, high=5, size=shape)

initial = x.flatten().astype(np.float32, copy=False)

data = {
    "v": 1,
    "dim": shape,
    "data": initial
    }

# conv layer
shape = (32,5,5,3)
f = np.random.randint(low=-10, high=+10, size=shape) 

conv1 = f.flatten().astype(np.float32, copy=False)

data = {
    "v": 1,
    "dim": shape,
    "data": conv1
    }

conv = {
    "layer_type": Layer.Convolution,
    "input_shape": x.shape,
    "kernel": data,
}

model = [conv]

x, n_params, n_multiplications, name = conv_layer(x, f)
print(p.format(name, str(x.shape), n_params, n_multiplications))

# max pooling
maxpool = {
    "layer_type": Layer.MaxPool,
    "input_shape": x.shape,
    "window": 2,
}

model.append(maxpool)

x, n_params, n_multiplications, name =  max_pooling_layer(x, 2)
print(p.format(name, str(x.shape), n_params, n_multiplications))

# relu layer
relu = {
    "layer_type": Layer.Relu,
    "input_shape": x.shape
}

model.append(relu)

x, n_params, n_multiplications, name = relu_layer(x)
print(p.format(name, str(x.shape), n_params, n_multiplications))

# conv layer
shape = (32,5,5,32)

f = np.random.randint(low=-10, high=+10, size=shape) 

conv2 = f.flatten().astype(np.float32, copy=False)

data = {
    "v": 1,
    "dim": shape,
    "data": conv2
    }

conv = {
    "layer_type": Layer.Convolution,
    "input_shape": x.shape,
    "kernel":data
}

model.append(conv)

x, n_params, n_multiplications, name = conv_layer(x, f)
print(p.format(name, str(x.shape), n_params, n_multiplications))

# max pooling
maxpool = {
    "layer_type": Layer.MaxPool,
    "input_shape": x.shape,
    "window": 2
}

model.append(maxpool)

x, n_params, n_multiplications, name =  max_pooling_layer(x, 2)
print(p.format(name, str(x.shape), n_params, n_multiplications))

# relu layer
relu = {
    "layer_type": Layer.Relu,
    "input_shape": x.shape,
}

model.append(relu)

x, n_params, n_multiplications, name = relu_layer(x)
print(p.format(name, str(x.shape), n_params, n_multiplications))

# flatten
flatten = {
    "layer_type": Layer.Flatten,
    "input_shape": x.shape,
}

model.append(flatten)

x, n_params, n_multiplications, name =  flatten_layer(x)
print(p.format(name, str(x.shape), n_params, n_multiplications))

# fully connected
shape = (1000, x.shape[0])
weights = np.random.randint(low=-10, high=+10, size=shape)

# weights json
data = {
    "v": 1,
    "dim": shape,
    "data": weights.flatten().astype(np.float32, copy=False)
}

# biases json
shape = [1000]
biases = np.random.randint(low=-10, high=+10, size=(1000))

data2 = {
    "v": 1,
    # ndarray can't take a single value, needs to be in json array
    "dim": shape,
    "data": biases.flatten().astype(np.float32, copy=False)
}

fully_connected = {
    "layer_type": Layer.FullyConnected,
    "input_shape": x.shape,
    "weights":data,
    "biases": data2
}

model.append(fully_connected)

x, n_params, n_multiplications, name = fully_connected_layer(x, weights, biases)
print(p.format(name, str(x.shape), n_params, n_multiplications))

# relu layer
relu = {
    "layer_type": Layer.Relu,
    "input_shape": x.shape,
}

model.append(relu)

x, n_params, n_multiplications, name = relu_layer(x)
print(p.format(name, str(x.shape), n_params, n_multiplications))

# fully connected
shape = (5, x.shape[0])
weights = np.random.randint(low=-10, high=+10, size=shape)

data = {
    "v": 1,
    "dim": shape,
    "data": weights.flatten().astype(np.float32, copy=False)
}

shape = [5]
biases = np.random.randint(low=-10, high=+10, size=shape)

data2 = {
    "v": 1,
    # ndarray can't take a single value, needs to be in json array
    "dim": [5],
    "data": biases.flatten().astype(np.float32, copy=False)
}

fully_connected = {
    "layer_type": Layer.FullyConnected,
    "input_shape": x.shape,
    "weights":data,
    "biases": data2
}

model.append(fully_connected)

x, n_params, n_multiplications, name = fully_connected_layer(x, weights, biases)
print(p.format(name, str(x.shape), n_params, n_multiplications))

assert(np.isclose(x, [ -9404869, -11033050, -34374361, -20396580,  70483360.]).all())

# normalization
norm = {
    "layer_type": Layer.Normalize,
    "input_shape": x.shape,
}

model.append(norm)

model = {
    "layers": model
}

x, n_params, n_multiplications, name = normalize(x)
print(p.format(name, str(x.shape), n_params, n_multiplications))


print("\nfinal output:", x)


model_data = json.dumps(model, cls=Encoder)
with open('../src/json/model.json', "w") as f:
    print('\ncreated model.json in the proto-neural-zkp/src/json folder')
    f.write(model_data)