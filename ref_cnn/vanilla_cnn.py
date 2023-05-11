import numpy as np

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

    dh = hf//2
    dw = wf//2

    output = np.zeros(shape=(h-2*dh, w-2*dw, c_out))

    for i in range(dh, h-dh):
        for j in range(dw, w-dw):
            a = input[i-dh:i+dh+1, j-dw:j+dw+1]
            for k in range(c_out):
                b = f[k,:,:,:]
                output[i-dh, j-dw, k] = (a*b).sum() # a.size multiplication
    
    n_params = f.size
    n_multiplications = a.size * c_out * (w-2*dw) * (h-2*dh)
    name = f"conv {'x'.join([str(e) for e in f.shape])}"
    
    return output, n_params, n_multiplications, name


def relu_layer(input):    
    output = input.copy()
    output[output<0] = 0
    
    n_params = 0
    n_multiplications = 0
    
    return output, n_params, n_multiplications, "relu"


def max_pooling_layer(input, s):
    """
    Apply max pooling layer using a sxs patch.
    """
    h, w, c = input.shape

    assert h%s == 0, "Height must be devisable by s!"
    assert w%s == 0, "Width must be devisable by s!"

    output = np.zeros(shape=(h//s, w//s, c))

    for i in range(0, h, s):
        for j in range(0, w, s):
            for k in range(c):
                a = input[i:i+s, j:j+s, k]
                output[i//s, j//s, k] = a.max()
    
    n_params = 0
    n_multiplications = 0
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
    name = f"conv {'x'.join([str(e) for e in weights.shape])}"
    
    return output, n_params, n_multiplications, name

def normalize(input):
    output = input / np.linalg.norm(input)
    n_params = 0
    n_multiplications = 1 + input.size
    return output, n_params, n_multiplications, "normalize"


if __name__ == "__main__":
    
    np.random.seed(12345)

    p = "{:>20} | {:>15} | {:>15} | {:>15} "
    print(p.format("layer", "output shape", "#parameters", "#ops"))
    print(p.format("-"*20, "-"*15, "-"*15, "-"*15))

    # input
    x = np.random.randint(low=-5, high=5, size=(120,80,3))

    # conv layer
    f = np.random.randint(low=-10, high=+10, size=(32,5,5,3)) 
    x, n_params, n_multiplications, name = conv_layer(x, f)
    print(p.format(name, str(x.shape), n_params, n_multiplications))

    # max pooling
    x, n_params, n_multiplications, name =  max_pooling_layer(x, 2)
    print(p.format(name, str(x.shape), n_params, n_multiplications))

    # relu layer
    x, n_params, n_multiplications, name = relu_layer(x)
    print(p.format(name, str(x.shape), n_params, n_multiplications))

    # conv layer
    f = np.random.randint(low=-10, high=+10, size=(32,5,5,32)) 
    x, n_params, n_multiplications, name = conv_layer(x, f)
    print(p.format(name, str(x.shape), n_params, n_multiplications))

    # max pooling
    x, n_params, n_multiplications, name =  max_pooling_layer(x, 2)
    print(p.format(name, str(x.shape), n_params, n_multiplications))

    # relu layer
    x, n_params, n_multiplications, name = relu_layer(x)
    print(p.format(name, str(x.shape), n_params, n_multiplications))

    # flatten
    x, n_params, n_multiplications, name =  flatten_layer(x)
    print(p.format(name, str(x.shape), n_params, n_multiplications))

    # fully connected
    weights = np.random.randint(low=-10, high=+10, size=(1000, x.shape[0])) 
    biases = np.random.randint(low=-10, high=+10, size=(1000)) 
    x, n_params, n_multiplications, name = fully_connected_layer(x, weights, biases)
    print(p.format(name, str(x.shape), n_params, n_multiplications))

    # relu layer
    x, n_params, n_multiplications, name = relu_layer(x)
    print(p.format(name, str(x.shape), n_params, n_multiplications))

    # fully connected
    weights = np.random.randint(low=-10, high=+10, size=(5, x.shape[0])) 
    biases = np.random.randint(low=-10, high=+10, size=(5)) 
    x, n_params, n_multiplications, name = fully_connected_layer(x, weights, biases)
    print(p.format(name, str(x.shape), n_params, n_multiplications))

    assert(np.isclose(x, [ -9404869, -11033050, -34374361, -20396580,  70483360.]).all())

    # normalization
    x, n_params, n_multiplications, name = normalize(x)
    print(p.format(name, str(x.shape), n_params, n_multiplications))

    print("\nfinal output:", x)