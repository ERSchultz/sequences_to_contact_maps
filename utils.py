from numba import jit
import numpy as np
import os

def make_dataset(dir):
    data_file_arr = []
    for file in os.listdir(dir):
        data_file = dir + '/' + file
        data_file_arr.append(data_file)
        # TODO zero padded??
    return data_file_arr

@jit
def x2xx(x):
    # input x is nxk, output is 2kxnxn
    # this is slow
    n, k = x.shape
    xx = np.zeros((k*2, n, n))
    for i in range(n):
        for j in range(i+1):
            xx[:, i, j] = np.append(x[i], x[j])
            xx[:, j, i] = np.append(x[j], x[i])
    return xx
