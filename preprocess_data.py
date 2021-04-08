import os
import numpy as np
from utils import *

def main(dir):
    paths = sorted(make_dataset(dir))
    for path in paths:
        print(path)
        x1_path = path + '/seq1.txt'
        x1 = np.loadtxt(x1_path)
        x2_path = path + '/seq2.txt'
        x2 = np.loadtxt(x2_path)
        x = np.vstack((x1, x2)).T.astype(np.int8)
        np.save(path + '/x.npy', x, x.astype(np.int8))

        xx = x2xx(x)
        np.save(path + '/xx.npy', xx.astype(np.int8))

        y_path = path + '/data_out/contacts.txt'
        y = np.loadtxt(y_path)[:1024, :1024] # TODO delete this later
        # y /= np.max(y)
        np.save(path + '/y.npy', y.astype(np.int8))

if __name__ == '__main__':
    clusterdir = '../../../project2/depablo/erschultz/dataset_04_06_21'
    mydir = 'dataset_04_06_21'
    main(clusterdir)
