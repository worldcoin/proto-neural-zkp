### Will take a long time! Took 13.2min on my machine (M1 Max MBP - DC)
import timeit
import numpy as np
from vanilla_cnn import *

if __name__ == "__main__":
    
    # instantiate matrices outside of time it block
    np.random.seed(12345)

    x = np.random.randint(low=-5, high=5, size=(120, 80, 3))

    f = np.random.randint(low=-10, high=+10, size=(32, 5, 5, 3)) 

    k = np.random.randint(low=-10, high=+10, size=(32,5,5,32)) 

    weights1 = np.random.randint(low=-10, high=+10, size=(1000, 14688)) 

    biases1 = np.random.randint(low=-10, high=+10, size=(1000)) 

    weights2 = np.random.randint(low=-10, high=+10, size=(5, 1000)) 

    biases2 = np.random.randint(low=-10, high=+10, size=(5)) 

    times = []

    runs = 100

    for _ in range(runs):
        starttime = timeit.default_timer()
        # conv layer
        x, n_params, n_multiplications, name = conv_layer(x, f)

        # max pooling
        x, n_params, n_multiplications, name =  max_pooling_layer(x, 2)

        # relu layer
        x, n_params, n_multiplications, name = relu_layer(x)

        # conv layer
        x, n_params, n_multiplications, name = conv_layer(x, k)

        # max pooling
        x, n_params, n_multiplications, name =  max_pooling_layer(x, 2)

        # relu layer
        x, n_params, n_multiplications, name = relu_layer(x)

        # flatten
        x, n_params, n_multiplications, name =  flatten_layer(x)

        # fully connected
        x, n_params, n_multiplications, name = fully_connected_layer(x, weights1, biases1)

        # relu layer
        x, n_params, n_multiplications, name = relu_layer(x)

        # fully connected
        x, n_params, n_multiplications, name = fully_connected_layer(x, weights2, biases2)

        # normalization
        x, n_params, n_multiplications, name = normalize(x)

        times.append(timeit.default_timer() - starttime)

        np.random.seed(12345)

        x = np.random.randint(low=-5, high=5, size=(120, 80, 3))

    average = sum(times) / len(times)
    print(f'The average time is {average} seconds for {runs} runs')
    # Result = 0.8297840171150046 for 1000 runs