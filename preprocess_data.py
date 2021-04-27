import multiprocessing
import numpy as np
import torch
from neural_net_utils.utils import *
from neural_net_utils.dataset_classes import Names
import time
import argparse
import os

def setupParser():
    parser = argparse.ArgumentParser(description='Base parser')
    parser.add_argument('--input_folder', type=str, default='dataset_04_18_21', help='Location of input data')
    parser.add_argument('--output_folder', type=str, default='test', help='Location to write data to')
    parser.add_argument('--num_workers', type=int, default=5, help='Number of processes to use')
    parser.add_argument('--k', type=int, default=2, help='Number of epigenetic marks')
    parser.add_argument('--n', type=int, default=1024, help='Number of particles')
    parser.add_argument('--sample_size', type=int, default=5, help='size of sample for diagonal normalization')
    parser.add_argument('--seed', type=int, default=42, help='random seed to use. Default: 42')

    return parser

def process_data(opt):
    in_paths = sorted(make_dataset(opt.input_folder))

    # ensure output files exist
    if not os.path.exists(opt.output_folder):
        os.mkdir(opt.cd , mode = 0o755)
    samples_path = os.path.join(opt.output_folder, 'samples')
    if not os.path.exists(samples_path):
        os.mkdir(samples_path, mode = 0o755)
    for in_path in in_paths:
        sample = os.path.split(in_path)[-1]
        out_path = os.path.join(opt.output_folder, 'samples', sample)
        if not os.path.exists(out_path):
            os.mkdir(out_path, mode = 0o755)

    out_paths = sorted(make_dataset(opt.output_folder))

    # set up for multiprocessing
    mapping = []
    for in_path, out_path in zip(in_paths, out_paths):
        mapping.append((in_path, out_path, opt.k, opt.n))

    with multiprocessing.Pool(opt.num_workers) as p:
        p.starmap(process_sample_save, mapping)

    # determine mean_dist for diagonal normaliztion
    meanDist_path = os.path.join(opt.input_folder, 'meanDist.npy')
    if not os.path.exists(meanDist_path):
        train_dataloader, _, _ = getDataLoaders(Names(opt.output_folder), 1,
                                                opt.num_workers, opt.seed, [0.8, 0.1, 0.1])
        meanDist = np.zeros(opt.n)
        for i, path in enumerate(train_dataloader):
            if i < opt.sample_size: # dataloader shuffles so this is a random sample
                path = path[0]
                y = np.load(os.path.join(path, 'y.npy'))
                meanDist += generateDistStats(y)
        meanDist = meanDist / opt.sample_size
        print(meanDist)
        np.save(meanDist_path, meanDist)
    else:
        meanDist = np.load(meanDist_path)

    # set up for multiprocessing
    mapping = []
    for in_path, out_path in zip(in_paths, out_paths):
        mapping.append((in_path, out_path, meanDist.copy()))

    with multiprocessing.Pool(opt.num_workers) as p:
        p.starmap(process_sample_diag_norm, mapping)

    # determine prcnt_dist for percentile normalization
    prcntDist_path = os.path.join(opt.input_folder, 'prcntDist.npy')
    if not os.path.exists(prcntDist_path):
        train_dataloader, _, _ = getDataLoaders(Names(opt.output_folder), 1,
                                                opt.num_workers, opt.seed, [0.8, 0.1, 0.1])
        y_arr = np.zeros((opt.sample_size, opt.n, opt.n))
        for i, path in enumerate(train_dataloader):
            if i < opt.sample_size: # dataloader shuffles so this is a random sample
                path = path[0]
                y_arr[i,:,:] = np.load(os.path.join(path, 'y_diag_norm.npy'))
                # This should be ok from a RAM standpoint

        prcntDist = getPercentiles(y_arr, [20, 40, 50, 60, 70, 80, 85, 90, 95, 100]) # flattens array to do computation
        print(prcntDist)
        np.save(prcntDist_path, prcntDist)
    else:
        prcntDist = np.load(prcntDist_path)

    # set up for multiprocessing
    mapping = []
    for in_path, out_path in zip(in_paths, out_paths):
        mapping.append((in_path, out_path, prcntDist.copy()))

    with multiprocessing.Pool(opt.num_workers) as p:
        p.starmap(process_sample_percentile_norm, mapping)

    # copy over chi
    chi = np.loadtxt(os.path.join(opt.input_folder, 'chis.txt'))
    np.save(os.path.join(opt.output_folder, 'chis.npy'), chi)

def process_sample_save(in_path, out_path, k, n):
    # check if sample needs to be processed
    if os.path.exists(os.path.join(out_path, 'x.npy')):
        return

    x = np.zeros((n, k))
    for i in range(1, k + 1):
        xi_path = os.path.join(in_path, 'seq{}.txt'.format(i))
        xi = np.loadtxt(xi_path)
        x[:, i-1] = xi
    np.save(os.path.join(out_path, 'x.npy'), x.astype(np.int8))

    xx = x2xx(x)
    np.save(os.path.join(out_path, 'xx.npy'), xx.astype(np.int8))

    y_path = os.path.join(in_path, 'data_out/contacts.txt')
    y = np.loadtxt(y_path)[:n, :n] # TODO delete this later
    np.save(os.path.join(out_path, 'y.npy'), y.astype(np.int16))

def process_sample_diag_norm(in_path, out_path, meanDist):
    # check if sample needs to be processed
    if os.path.exists(os.path.join(out_path, 'y_diag_norm.npy')):
        return

    y = np.load(os.path.join(in_path, 'y.npy')).astype(np.float64)
    y_diag = diagonal_normalize(y, meanDist)
    np.save(os.path.join(out_path, 'y_diag_norm.npy'), y_diag)

def process_sample_percentile_norm(in_path, out_path, prcntDist):
    # check if sample needs to be processed
    # if os.path.exists(os.path.join(out_path, 'y_prcnt_norm.npy')):
    #     return

    y_diag = np.load(os.path.join(in_path, 'y_diag_norm.npy')).astype(np.float64)
    y_prcnt = percentile_normalize(y_diag, prcntDist)
    np.save(os.path.join(out_path, 'y_prcnt_norm.npy'), y_prcnt)


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
    t0 = time.time()
    parser = setupParser()
    opt = parser.parse_args()
    print(opt.input_folder)
    process_data(opt)
    print('Total time: {}'.format(time.time() - t0))


if __name__ == '__main__':
    main()
