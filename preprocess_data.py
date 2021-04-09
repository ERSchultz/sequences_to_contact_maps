import numpy as np
from neural_net_utils.utils import *
import sys

def process_data(dirname):
    paths = sorted(make_dataset(dirname))
    for path in paths:
        print(path)
        x1_path = path + '/seq1.txt'
        x1 = np.loadtxt(x1_path)
        x2_path = path + '/seq2.txt'
        x2 = np.loadtxt(x2_path)
        x = np.vstack((x1, x2)).T
        np.save(path + '/x.npy', x.astype(np.int8))

        xx = x2xx(x)
        np.save(path + '/xx.npy', xx.astype(np.int8))

        y_path = path + '/data_out/contacts.txt'
        y = np.loadtxt(y_path)[:1024, :1024] # TODO delete this later
        np.save(path + '/y.npy', y.astype(np.int16))

        y_diag = diagonal_normalize(y)
        np.save(path + '/y_diag_norm.npy', y_diag)

def test_process_data(dirname):
    paths = sorted(make_dataset(dirname))
    path = paths[0]
    print(path)
    x1_path = path + '/seq1.txt'
    x1 = np.loadtxt(x1_path)
    x2_path = path + '/seq2.txt'
    x2 = np.loadtxt(x2_path)
    x = np.vstack((x1, x2)).T
    np.save(path + '/x.npy', x.astype(np.int8))
    xload = np.load(path + '/x.npy')
    assert np.array_equal(x, xload)

    xx = x2xx(x)
    np.save(path + '/xx.npy', xx.astype(np.int8))
    xxload = np.load(path + '/xx.npy')
    assert np.array_equal(xx, xxload)

    y_path = path + '/data_out/contacts.txt'
    y = np.loadtxt(y_path)[:1024, :1024] # TODO delete this later
    np.save(path + '/y.npy', y.astype(np.int16))
    yload = np.load(path + '/y.npy')
    assert np.array_equal(y, yload)

    y_diag = diagonal_normalize(y.astype(np.float64))
    np.save(path + '/y_diag_norm.npy', y_diag)
    y_diagload = np.load(path + '/y_diag_norm.npy')
    assert np.array_equal(y_diag, y_diagload)

def main():
    if len(sys.argv) == 1:
        dirname = 'dataset_04_06_21'
    else:
        dirname = sys.argv[1]
    print(dirname)
    process_data(dirname)
    # test_process_data(dirname)


if __name__ == '__main__':
    main()
