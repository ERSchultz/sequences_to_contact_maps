import torch
from torch.utils.data import DataLoader
from numba import jit
import numpy as np
import os
import math
import matplotlib.pyplot as plt

def make_dataset(dir):
    data_file_arr = []
    for file in os.listdir(dir):
        data_file = dir + '/' + file
        data_file_arr.append(data_file)
        # TODO zero padded??
    return data_file_arr

def getDataLoaders(dataset, batchSize = 64, num_workers = 0, split = [0.7, 0.2, 0.1]):
    N = len(dataset)
    assert sum(split) - 1 < 1e-5, sum(split)
    trainN = math.floor(N * split[0])
    valN = math.floor(N * split[1])
    testN = N - trainN - valN
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                        [trainN, valN, testN],
                                        generator = torch.Generator().manual_seed(42))
    # TODO may need to shuffle before split
    train_dataloader = DataLoader(train_dataset, batch_size = batchSize,
                                    shuffle = True, num_workers = num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size = batchSize,
                                    shuffle = True, num_workers = num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size = batchSize,
                                    shuffle = True, num_workers = num_workers)

    return train_dataloader, val_dataloader, test_dataloader

@jit
def x2xx(x, append = False):
    # input x is nxk, output is 2kxnxn
    # this is slow
    n, k = x.shape
    if append:
        xx = np.zeros((k*2, n, n))
    else:
        xx = np.zeros((k, n, n))
    for i in range(n):
        for j in range(i+1):
            if append:
                xx[:, i, j] = np.append(x[i], x[j])
                xx[:, j, i] = np.append(x[j], x[i])
            else:
                xx[:, i, j] = np.add(x[i], x[j])
                xx[:, j, i] = np.add(x[j], x[i])
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

def plotModelFromDir(dir, model, ofile):
    saveDict = torch.load(dir, map_location=torch.device('cpu'))
    model.load_state_dict(saveDict['model_state_dict'])
    epochs = saveDict['epoch']
    print(epochs)
    train_loss_arr = saveDict['train_loss']
    plt.plot(np.arange(0, epochs), train_loss_arr, label = 'train loss')
    plt.legend()
    plt.savefig('images/' + ofile)
    plt.close()

if __name__ == '__main__':
    main()
