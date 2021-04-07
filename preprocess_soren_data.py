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
        np.savetxt(path + '/x.txt', x, fmt = '%i')

        xx = x2xx(x)
        np.save('/xx.npy', xx.astype(int))

if __name__ == '__main__':
    clusterdir = '../../../project2/depablo/skyhl/dataset_04_06_21'
    mydir = 'dataset_04_06_21'
    main(mydir)
