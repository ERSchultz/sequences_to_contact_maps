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

@jit
def diagonal_normalize(y, d = 1):
    prob = generateProbDist(y, d)
    for i in range(len(y)):
        for j in range(i + 1):
            distance = y[i,j] - y[j,i]
            pos = int(distance / d) # need to guarantee int
            y[i,j] /= prob[pos]
            y[j,i] = y[i,j]

    return y

@jit
def generateProbDist(y, d):
    n = len(y)
    distances = range(0, n + d, d)
    observed = np.zeros(len(distances))
    possible = np.zeros(len(distances))
    for pos, distance in enumerate(distances):
        possible[pos] = (n - distance + d) * d
    for i in range(n):
        for j in range(i + 1):
            distance = y[i,j] - y[j,i]
            pos = int(distance / d) # need to guarantee int
            observed[pos] += y[i,j] # = y[j,i]

    prob = np.divide(observed, possible)
    return prob

if __name__ == '__main__':
    main()
