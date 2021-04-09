import torch
from torch.utils.data import DataLoader
from numba import jit
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import matplotlib.colors
import argparse
import seaborn as sns

def make_dataset(dir):
    data_file_arr = []
    for file in os.listdir(dir):
        data_file = dir + '/' + file
        data_file_arr.append(data_file)
        # TODO zero padded??
    return data_file_arr

def getDataLoaders(dataset, batch_size = 64, num_workers = 0, split = [0.7, 0.2, 0.1], seed = 42):
    N = len(dataset)
    assert sum(split) - 1 < 1e-5, sum(split)
    trainN = math.floor(N * split[0])
    valN = math.floor(N * split[1])
    testN = N - trainN - valN
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                        [trainN, valN, testN],
                                        generator = torch.Generator().manual_seed(seed))
    # TODO may need to shuffle before split
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size,
                                    shuffle = True, num_workers = num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size = batch_size,
                                    shuffle = True, num_workers = num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size,
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
                # TODO mean
    return xx

@jit
def diagonal_normalize(y):
    exp = generateExpectedDist(y)
    for i in range(len(y)):
        for j in range(i + 1):
            distance = i - j
            exp_d = exp[distance]
            if exp_d == 0:
                pass
            else:
                y[i,j] /= exp_d
                y[j,i] = y[i,j]

    return y

def generateExpectedDist(y):
    n = len(y)
    distances = range(0, n, 1)
    denom = [n-distance for distance in distances]
    num = np.zeros(len(distances))
    for i in range(n):
        for j in range(i + 1):
            distance = i - j
            num[distance] += y[i,j] # = y[j,i]

    exp = np.divide(num, denom)
    return exp

# Plotting functions
def plotExpectedDist(y, ofile, title = None, y_scale = None):
    exp = generateExpectedDist(y)
    print(exp)
    for i in exp:
        print(i)
    fig, ax = plt.subplots()
    ax.plot(exp)
    if y_scale == 'log':
        plt.yscale('log')
    plt.ylabel('expected interaction count')
    plt.xlabel('distance along chain')
    if title is not None:
        plt.title(title)
    plt.savefig(os.path.join('images', ofile))
    plt.close()

def plotProbDist(y, x, ofile, title = None):
    fig, ax = plt.subplots()
    ax.plot(exp)
    if y_scale == 'log':
        plt.yscale('log')
    plt.ylabel('expected interaction count')
    plt.xlabel('distance along chain')
    if title is not None:
        plt.title(title)
    plt.savefig(os.path.join('images', ofile))
    plt.close()

def plotModelFromDir(dir, model, ofile):
    saveDict = torch.load(dir, map_location=torch.device('cpu'))
    model.load_state_dict(saveDict['model_state_dict'])
    train_loss_arr = saveDict['train_loss']
    plt.plot(np.arange(0, epochs), train_loss_arr, label = 'train loss')
    plt.legend()
    plt.savefig(os.path.join('images', ofile))
    plt.close()

def plotModelFromArrays(train_loss_arr, ofile, val_loss = None):
    plt.plot(np.arange(0, len(train_loss_arr)), train_loss_arr, label = 'train loss')
    if val_loss is not None:
        plt.axhline(y = val_loss, color = 'black', linestyle = '--', label = 'final val loss')
    plt.legend()
    plt.savefig(os.path.join('images', ofile))
    plt.close()

def plotContactMap(y, ofile, title = None):
    mycmap = matplotlib.colors.LinearSegmentedColormap.from_list('custom',
                                             [(0,    'white'),
                                              (1,    'red')], N=126)
    y = y / np.mean(y)
    plt.figure(figsize=(10, 10))
    ax = sns.heatmap(ynpy, linewidth=0, vmin = 0, vmax = 1, cmap = mycmap)
    if title is not None:
        plt.title(title)
    plt.savefig(os.path.join('images', ofile))
    plt.close()

def getBaseParser():
    parser = argparse.ArgumentParser(description='Base parser')

    # train args
    parser.add_argument('--start_epoch', type=int, default=1, help='Starting epoch')
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs to train for')
    parser.add_argument('--save_mod', type=int, default=5, help='How often to save')
    parser.add_argument('--print_mod', type=int, default=2, help='How often to print')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning eate. Default=0.001')
    parser.add_argument('--gpus', type=int, default=1, help='Number of gpus')
    parser.add_argument('--milestones', type=str, help='Milestones for lr decay format: <milestone1-milestone2>')
    parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for lr decay')

    # model args
    parser.add_argument('--data_folder', type=str, default='dataset_04_06_21', help='Location of data')
    parser.add_argument('--pretrained', type=bool, default=False)
    parser.add_argument('--resume_training', type=bool, default=False)
    parser.add_argument('--ifile_folder', type=str, default='models/', help='Location of input file for pretrained model')
    parser.add_argument('--ifile', type=str, help='Name of input file for pretrained model')
    parser.add_argument('--ofile_folder', type=str, default='models/', help='Location to save checkpoint models')
    parser.add_argument('--ofile', type=str, default='model', help='Name of save file')
    parser.add_argument('--k', type=int, default=2, help='Number of epigenetic marks')
    parser.add_argument('--seed', type=int, default=42, help='random seed to use. Default: 42')

    return parser
