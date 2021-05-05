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

    # dataloader args
    parser.add_argument('--split', type=str2list, default=[0.8, 0.1, 0.1], help='Train, val, test split for dataset')
    parser.add_argument('--shuffle', type=str2bool, default=True, help='Whether or not to shuffle dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--num_workers', type=int, default=5, help='Number of processes to use')

    # model args
    parser.add_argument('--k', type=int, default=2, help='Number of epigenetic marks')
    parser.add_argument('--n', type=int, default=1024, help='Number of particles')

    # preprocessing args
    parser.add_argument('--sample_size', type=int, default=5, help='Size of sample for preprocessing statistics')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to use. Default: 42')
    parser.add_argument('--overwrite', type=str2bool, default=False, help='Wheter or not to overwrite existing preprocessing files')
    parser.add_argument('--percentiles', type=str2list, default=[20, 40, 50, 60, 70, 80, 85, 90, 95, 100], help='Percentiles to use for percentile preprocessing')

    return parser

def process_data(opt):
    in_paths = sorted(make_dataset(opt.input_folder))

    # ensure output files exist
    if not os.path.exists(opt.output_folder):
        os.mkdir(opt.output_folder, mode = 0o755)
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
        mapping.append((in_path, out_path, opt.k, opt.n, opt.overwrite))

    with multiprocessing.Pool(opt.num_workers) as p:
        p.starmap(process_sample_save, mapping)

    diag_processing(opt, out_paths)

    percentile_processing(opt, out_paths)

    # find min,max of y_diag and y_prcnt
    y_diag_min_max = np.array([float('inf'), -float('inf')])
    y_prcnt_min_max = np.array([float('inf'), -float('inf')])
    train_dataloader, _, _ = getDataLoaders(Names(opt.output_folder), opt)
    for i, path in enumerate(train_dataloader):
        if i < opt.sample_size: # dataloader shuffles so this is a random sample
            path = path[0]
            y_diag = np.load(os.path.join(path, 'y_diag.npy'))
            min_val = np.min(y_diag)
            if min_val < y_diag_min_max[0]:
                y_diag_min_max[0] = min_val
            max_val = np.max(y_diag)
            if max_val > y_diag_min_max[1]:
                y_diag_min_max[1] = max_val

            y_prcnt = np.load(os.path.join(path, 'y_prcnt.npy'))
            min_val = np.min(y_prcnt)
            if min_val < y_prcnt_min_max[0]:
                y_prcnt_min_max[0] = min_val
            max_val = np.max(y_prcnt)
            if max_val > y_prcnt_min_max[1]:
                y_prcnt_min_max[1] = max_val
    print(y_diag_min_max)
    print(y_prcnt_min_max)
    np.save(os.path.join(opt.output_folder, 'y_diag_min_max.npy'), y_diag_min_max.astype(np.float64))
    np.save(os.path.join(opt.output_folder, 'y_prcnt_min_max.npy'), y_prcnt_min_max.astype(np.float64))


    # copy over chi
    chi = np.loadtxt(os.path.join(opt.input_folder, 'chis.txt'))
    np.save(os.path.join(opt.output_folder, 'chis.npy'), chi)

def process_sample_save(in_path, out_path, k, n, overwrite):
    # check if sample needs to be processed
    if not os.path.exists(os.path.join(out_path, 'x.npy')) or overwrite:
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

def diag_processing(opt, out_paths):
    # determine mean_dist for diagonal preprocessing
    meanDist_path = os.path.join(opt.output_folder, 'meanDist.npy')
    if not os.path.exists(meanDist_path):
        train_dataloader, _, _ = getDataLoaders(Names(opt.output_folder), 1,
                                                opt.num_workers, opt.seed, opt.split)
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
    for out_path in out_paths:
        mapping.append((out_path, meanDist.copy(), opt.overwrite))

    with multiprocessing.Pool(opt.num_workers) as p:
        p.starmap(process_sample_diag, mapping)

def process_sample_diag(path, meanDist, overwrite):
    # check if sample needs to be processed
    if not os.path.exists(os.path.join(path, 'y_diag.npy')) or overwrite:
        y = np.load(os.path.join(path, 'y.npy')).astype(np.float64)
        y_diag = diagonal_preprocessing(y, meanDist)
        np.save(os.path.join(path, 'y_diag.npy'), y_diag)

def percentile_processing(opt, out_paths):
    # determine prcnt_dist for percentile preprocessing
    prcntDist_path = os.path.join(opt.output_folder, 'prcntDist.npy')
    if not os.path.exists(prcntDist_path):
        train_dataloader, _, _ = getDataLoaders(Names(opt.output_folder), 1,
                                                opt.num_workers, opt.seed, opt.split)
        y_arr = np.zeros((opt.sample_size, opt.n, opt.n))
        for i, path in enumerate(train_dataloader):
            if i < opt.sample_size: # dataloader shuffles so this is a random sample
                path = path[0]
                y_arr[i,:,:] = np.load(os.path.join(path, 'y_diag.npy'))
                # This should be ok from a RAM standpoint

        prcntDist = getPercentiles(y_arr, opt.percentiles) # flattens array to do computation
        print(prcntDist)
        np.save(prcntDist_path, prcntDist)
    else:
        prcntDist = np.load(prcntDist_path)

    # set up for multiprocessing
    mapping = []
    for out_path in out_paths:
        mapping.append((out_path, prcntDist.copy(), opt.overwrite))

    with multiprocessing.Pool(opt.num_workers) as p:
        p.starmap(process_sample_percentile, mapping)


def process_sample_percentile(path, prcntDist, overwrite):
    if not os.path.exists(os.path.join(path, 'y_prcnt.npy')) or overwrite:
        y_diag = np.load(os.path.join(path, 'y_diag.npy')).astype(np.float64)
        y_prcnt = percentile_preprocessing(y_diag, prcntDist)
        np.save(os.path.join(path, 'y_prcnt.npy'), y_prcnt.astype(np.int16))

def main():
    t0 = time.time()
    parser = setupParser()
    opt = parser.parse_args()
    print(opt.input_folder)
    process_data(opt)
    print('Total time: {}'.format(time.time() - t0))


if __name__ == '__main__':
    main()
