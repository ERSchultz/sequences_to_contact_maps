import argparse
import multiprocessing
import os
import os.path as osp
import time

import matplotlib.pyplot as plt
import numpy as np

from .argparse_utils import str2bool, str2list
from .dataset_classes import Names, make_dataset
from .neural_net_utils import get_data_loaders
from .utils import (diagonal_preprocessing, genomic_distance_statistics,
                    get_percentiles, percentile_preprocessing, x2xx)


def getArgs():
    parser = argparse.ArgumentParser(description='Base parser')
    parser.add_argument('--input_folder', type=str, default='/home/eric/dataset_test',
                        help='Location of input data')
    parser.add_argument('--output_folder', type=str, default='/home/eric/dataset_test2',
                        help='Location to write data to')
    parser.add_argument('--min_sample', type=int, default=0,
                        help='minimum sample id')
    parser.add_argument('--verbose', type=str2bool, default=True,
                        help='True to print')

    # dataloader args
    parser.add_argument('--split_percents', type=str2list,
                        help='Train, val, test split for dataset (percents)')
    parser.add_argument('--split_sizes', type=str2list, default=[-1, 200, 0],
                        help='Train, val, test split for dataset (counts), -1 for remainder')
    parser.add_argument('--random_split', type=str2bool, default=False,
                        help='True to use random train, val, test split')
    parser.add_argument('--shuffle', type=str2bool, default=True,
                        help='Whether or not to shuffle dataset')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of threads for data loader to use')


    # model args
    parser.add_argument('--m', type=int, default=1024, help='Number of particles')

    # preprocessing args
    parser.add_argument('--sample_size', type=int, default=200,
                        help='Size of sample for preprocessing statistics')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed to use. Default: 42')
    parser.add_argument('--overwrite', type=str2bool, default=False,
                        help='Whether or not to overwrite existing preprocessing files')
    parser.add_argument('--percentiles', type=str2list,
                        help='Percentiles to use for percentile preprocessing (None to skip)')
    parser.add_argument('--diag_batch', type=str2bool, default=False,
                        help='True to use batch for diag norm (in addition to instance;)')
    parser.add_argument('--use_x2xx', type=str2bool, default=False,
                        help='True to calculate x2xx')

    args = parser.parse_args()

    # used in get_data_loaders
    args.GNN_mode = False
    args.batch_size = 1

    return args

def make_paths(args, in_paths):
    '''Helper function to ensure that necessary paths exist.'''
    if not osp.exists(args.output_folder):
        os.mkdir(args.output_folder, mode = 0o755)
    samples_path = osp.join(args.output_folder, 'samples')
    if not osp.exists(samples_path):
        os.mkdir(samples_path, mode = 0o755)

    for in_path in in_paths:
        sample = osp.split(in_path)[-1]
        out_path = osp.join(args.output_folder, 'samples', sample)
        if not osp.exists(out_path):
            os.mkdir(out_path, mode = 0o755)

def get_min_max(sample_size, dataloader, ifile):
    '''
    Helper function to get min and max values of all contact maps.

    Inputs:
        sample_size: number of samples to consider when calculating statistics
        data_loader: data loader to use
        ifile: type of file to load
    '''
    min_max = np.array([float('inf'), -float('inf')])
    for i, path in enumerate(dataloader):
        if i < sample_size: # dataloader shuffles so this is a random sample
            path = path[0]
            y = np.load(osp.join(path, ifile))
            min_val = np.min(y)
            if min_val < min_max[0]:
                min_max[0] = min_val
            max_val = np.max(y)
            if max_val > min_max[1]:
                min_max[1] = max_val
    return min_max

def process_sample_save(in_path, out_path, n, overwrite, use_x2xx):
    '''Saves relevant files as .npy files'''
    x_npy_file = osp.join(out_path, 'x.npy')
    if not osp.exists(x_npy_file) or overwrite:

        # determine k
        k = 0
        while osp.exists(osp.join(in_path, f'seq{k}.txt')):
            k += 1

        x = np.zeros((n, k))
        for i, seq_i in enumerate(range(k)):
            xi_path = osp.join(in_path, 'seq{}.txt'.format(seq_i))
            if osp.exists(xi_path):
                xi = np.loadtxt(xi_path)
                x[:, i] = xi
        np.save(x_npy_file, x.astype(np.int8))

    if use_x2xx:
        # check if sample needs to be processed
        xx_file = osp.join(out_path, 'xx.npy')
        if not osp.exists(xx_file) or overwrite:
            xx = x2xx(x)
            np.save(osp.join(out_path, 'xx.npy'), xx.astype(np.int8))

    s_source_file = osp.join(in_path, 's_matrix.txt')
    if osp.exists(s_source_file):
        s_output_file = osp.join(out_path, 's.npy')
        if not osp.exists(s_output_file) or overwrite:
            s = np.loadtxt(s_source_file)
            np.save(s_output_file, s)

    y_npy_file = osp.join(out_path, 'y.npy')
    if not osp.exists(y_npy_file) or overwrite:
        y_path = osp.join(in_path, 'data_out/contacts.txt')
        if osp.exists(y_path):
            y = np.loadtxt(y_path)[:n, :n]
        else:
            y_path = osp.join(in_path, 'y.npy')
            if osp.exists(y_path):
                y = np.load(y_path)
        np.save(y_npy_file, y.astype(np.int16))

    chi_source_file = osp.join(in_path, 'chis.txt')
    if osp.exists(chi_source_file):
        chi_output_file = osp.join(out_path, 'chis.npy')
        if not osp.exists(chi_output_file) or overwrite:
            chi = np.loadtxt(chi_source_file)
            np.save(chi_output_file, chi)

def process_diag(args, out_paths):
    # determine mean_dist for batch diagonal preprocessing
    if args.diag_batch:
        meanDist_path = osp.join(args.output_folder, 'meanDist.npy')
        if not osp.exists(meanDist_path) or args.overwrite:
            train_dataloader, _, _ = get_data_loaders(Names(args.input_folder, args.min_sample), args)
            assert args.sample_size <= args.trainN, "Sample size too large - max {}".format(args.trainN)
            meanDist = np.zeros(args.m)
            for i, path in enumerate(train_dataloader):
                if i < args.sample_size: # dataloader shuffles so this is a random sample
                    path = path[0]
                    y = np.load(osp.join(path, 'y.npy'))
                    meanDist += genomic_distance_statistics(y)
            meanDist = meanDist / args.sample_size

            plt.plot(meanDist[10:])
            plt.yscale('log')
            plt.xlabel('Genomic Distance')
            plt.ylabel('Mean Contact Frequency')
            plt.savefig(osp.join(args.output_folder, 'meanDist.png'))
            plt.close()

            np.save(meanDist_path, meanDist)
        else:
            meanDist = np.load(meanDist_path)
    else:
        meanDist = np.array(None) # not needed

    # set up for multiprocessing
    mapping = []
    for out_path in out_paths:
        mapping.append((out_path, meanDist.copy(), args.overwrite, args.diag_batch))

    with multiprocessing.Pool(args.num_workers) as p:
        p.starmap(process_sample_diag, mapping)

def process_sample_diag(path, meanDist, overwrite, use_batch):
    '''Inner function for process_diag.'''
    if use_batch:
        y_diag_batch_path = osp.join(path, 'y_diag_batch.npy')
        # check if sample needs to be processed
        if not osp.exists(y_diag_batch_path) or overwrite:
            y = np.load(osp.join(path, 'y.npy')).astype(np.float64)
            y_diag_batch = diagonal_preprocessing(y, meanDist)
            np.save(y_diag_batch_path, y_diag_batch)

    y_diag_instance_path = osp.join(path, 'y_diag.npy')
    if not osp.exists(y_diag_instance_path) or overwrite:
        y = np.load(osp.join(path, 'y.npy')).astype(np.float64)
        meanDist = genomic_distance_statistics(y)
        y_diag_instance = diagonal_preprocessing(y, meanDist)
        np.save(y_diag_instance_path, y_diag_instance)

def process_percentile(args, out_paths):
    # determine prcnt_dist for percentile preprocessing
    prcntDist_path = osp.join(args.output_folder, 'prcntDist.npy')
    if not osp.exists(prcntDist_path) or args.overwrite:
        train_dataloader, _, _ = getDataLoaders(Names(args.output_folder), args)
        assert args.sample_size <= args.trainN, "Sample size too large - max {}".format(args.trainN)
        y_arr = np.zeros((args.sample_size, args.m, args.m))
        for i, path in enumerate(train_dataloader):
            if i < args.sample_size: # dataloader shuffles so this is a random sample
                path = path[0]
                y_diag = np.load(osp.join(path, 'y_diag.npy'))
                y_arr[i,:,:] = y_diag
                # This should be ok from a RAM standpoint

        prcntDist = get_percentiles(y_arr, args.percentiles, plot = False) # flattens array to do computation
        print('prcntDist: ', prcntDist)
        np.save(prcntDist_path, prcntDist)
    else:
        prcntDist = np.load(prcntDist_path)

    # set up for multiprocessing
    mapping = []
    for out_path in out_paths:
        mapping.append((out_path, prcntDist.copy(), args.overwrite))

    with multiprocessing.Pool(args.num_workers) as p:
        p.starmap(process_sample_percentile, mapping)

def process_sample_percentile(path, prcntDist, overwrite):
    '''Inner function for process_percentile.'''
    if not osp.exists(osp.join(path, 'y_prcnt.npy')) or overwrite:
        y_diag = np.load(osp.join(path, 'y_diag.npy')).astype(np.float64)
        y_prcnt = percentile_preprocessing(y_diag, prcntDist)
        np.save(osp.join(path, 'y_prcnt.npy'), y_prcnt.astype(np.int16))

def main():
    t0 = time.time()
    args = getArgs()
    print(args)

    print('Input folder: ', args.input_folder)
    print('Output folder: ', args.output_folder)
    in_paths = sorted(make_dataset(args.input_folder, args.min_sample))

    # ensure output files exist
    make_paths(args, in_paths)

    out_paths = sorted(make_dataset(args.output_folder, args.min_sample))

    # set up for multiprocessing
    mapping = []
    for in_path, out_path in zip(in_paths, out_paths):
        mapping.append((in_path, out_path, args.m, args.overwrite, args.use_x2xx))

    with multiprocessing.Pool(args.num_workers) as p:
        p.starmap(process_sample_save, mapping)

    train_dataloader, _, _ = get_data_loaders(Names(args.output_folder, args.min_sample), args)

    # diag
    print('Diagonal Preprocessing')
    process_diag(args, out_paths)
    y_diag_min_max = get_min_max(args.sample_size, train_dataloader, 'y_diag.npy')
    np.save(osp.join(args.output_folder, 'y_diag_min_max.npy'), y_diag_min_max.astype(np.float64))
    print('y_diag_min_max: ', y_diag_min_max)

    if args.diag_batch:
        y_diag_batch_min_max = get_min_max(args.sample_size, train_dataloader, 'y_diag_batch.npy')
        np.save(osp.join(args.output_folder, 'y_diag__batch_min_max.npy'), y_diag_batch_min_max.astype(np.float64))
        print('y_diag_batch_min_max: ', y_diag_batch_min_max)

    # percentile
    if args.percentiles is not None:
        print('Percentile Preprocessing')
        process_percentile(args, out_paths)
        y_prcnt_min_max = get_min_max(args.sample_size, train_dataloader, 'y_prcnt.npy')
        np.save(osp.join(args.output_folder, 'y_prcnt_min_max.npy'), y_prcnt_min_max.astype(np.float64))
        print('y_prcnt_min_max: ', y_prcnt_min_max)

    # copy over chi
    chis_path = osp.join(args.input_folder, 'chis.txt')
    if osp.exists(chis_path):
        chi = np.loadtxt(chis_path)
        np.save(osp.join(args.output_folder, 'chis.npy'), chi)
        np.savetxt(osp.join(args.output_folder, 'chis.txt'), chi, fmt='%0.5f')

    print('Total time: {}'.format(time.time() - t0))

if __name__ == '__main__':
    main()
