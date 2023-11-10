import math
import multiprocessing
import os
import os.path as osp
import re
import shutil
import sys
import tarfile
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch
from numba import jit, njit
from pylib.utils.utils import nan_pearsonr
from scipy.stats import pearsonr, spearmanr
from skimage.measure import block_reduce
from sklearn.decomposition import PCA

LETTERS='ABCDEFGHIJKLMNOPQRSTUVWXYZ'

## data processing functions ##
def x2xx(x, mode = 'mean'):
    # TODO better explanation here
    """
    Function for converting x to an image xx.

    For add:
        xx[k, i, j] = x[i, :] + x[j, :] (element-wise)
    For mean:
        xx[k, i, j] = (x[i, :] + x[j, :]) / 2 (element-wise)
    For append:
        xx[2k, i, j] = [x[i, :] ... x[j, :]]


    Inputs:
        x: (n x k) array
        mode: method for combining xi and xj

    Outputs:
        xx: (2k x n x n) or (k x n x n) array
    """
    n, k = x.shape
    if mode == 'append':
        xx = np.zeros((k*2, n, n))
    else:
        xx = np.zeros((k, n, n))
    for i in range(n):
        for j in range(i+1):
            if mode == 'append':
                xx[:, i, j] = np.append(x[i], x[j])
                xx[:, j, i] = np.append(x[j], x[i])
            elif mode =='add':
                xx[:, i, j] = np.add(x[i], x[j])
                xx[:, j, i] = np.add(x[j], x[i])
            elif mode == 'mean':
                xx[:, i, j] = np.add(x[i], x[j]) / 2
                xx[:, j, i] = np.add(x[j], x[i]) / 2
    return xx

def percentile_preprocessing(y, percentiles):
    """
    Performs percentile preprocessing on contact map y.

    The maximum value in y must at most percentiles[-1].

    Inputs:
        y: contact map numpy array (2 dimensional)
        percentiles: list of percentiles (frequencies)

    Outputs:
        result: new contact map
    """

    if len(y.shape) == 3:
        N, H, W = y.shape
        assert N == 1
        y = y.reshape(H,W)
    elif len(y.shape) == 4:
        N, C, H, W = y.shape
        assert N == 1 and C == 1
        y = y.reshape(H,W)

    result = np.zeros_like(y)
    for i in range(len(y)):
        for j in range(i + 1):
            val = y[i,j]
            p = 0
            while p < len(percentiles)-1 and percentiles[p] < val:
                p += 1
            result[i,j] = p
            result[j,i] = p

    return result

def crop(input, size):
    '''
    Crop input np array to have ncols and nrows (size < 0 returns original input).
    '''
    if size > 0:
        return input[:size, :size]
    else:
        return input

def rescale_matrix(inp, factor, triu=True):
    '''
    Rescales input matrix by factor.
    if inp is 1024x1024 and factor=2, out is 512x512
    '''
    if inp is None:
        return None
    assert len(inp.shape) == 2, f'must be 2d array not {inp.shape}'
    m, _ = inp.shape
    assert m % factor == 0, f'factor must evenly divide m {m}%{factor}={m%factor}'

    if triu:
        inp = np.triu(inp) # need triu to not double count entries
    processed = block_reduce(inp, (factor, factor), np.sum) # sum-pool operation

    if triu:
        # need to make symmetric again
        processed = np.triu(processed)
        out = processed + np.triu(processed, 1).T
        return out
    else:
        return processed

def diagonal_rescale(inp, factor):
    '''
    Rescales input matrix by factor.
    if inp is 1024x1024 and factor=2, out is 512x512
    '''
    assert len(inp.shape) == 2, f'must be 2d array not {inp.shape}'
    m, _ = inp.shape
    assert m % factor == 0, f'factor must evenly divide m: {m}%{factor}={m%factor}'

    # diagonal-pool operation
    out = block_reduce(inp, (factor, factor), lambda x, axis: np.sum(np.multiply(x, np.eye(factor)), axis = axis))

    return out

## plotting helper functions ##
def un_normalize(y, minmax):
    ymin = minmax[0].item()
    ymax = minmax[1].item()
    return y.copy() * (ymax - ymin) + ymin

def get_percentiles(arr, prcnt_arr, plot = True):
    """Helper function to get multiple percentiles at once."""
    result = np.zeros_like(prcnt_arr).astype(np.float64)
    if plot:
        plt.hist(arr.flatten())
        plt.show()
    for i, p in enumerate(prcnt_arr):
        result[i] = np.percentile(arr, p)
    return result

def calc_dist_strat_corr(y, yhat, mode = 'pearson', return_arr = False):
    """
    Helper function to calculate correlation stratified by distance.

    Inputs:
        y: target
        yhat: prediction
        mode: pearson or spearman (str)

    Outpus:
        avg: average distance stratified correlation
        corr_arr: array of distance stratified correlations
    """
    if mode.lower() == 'pearson':
        stat = pearsonr
    elif mode.lower() == 'nan_pearson':
        stat = nan_pearsonr
    elif mode.lower() == 'spearman':
        stat = spearmanr

    assert len(y.shape) == 2
    n, _ = y.shape
    triu_ind = np.triu_indices(n)

    corr_arr = np.zeros(n-2)
    corr_arr[0] = np.NaN
    for d in range(1, n-2):
        # n-1, n, and 0 are NaN always, so skip
        y_diag = np.diagonal(y, offset = d)
        yhat_diag = np.diagonal(yhat, offset = d)
        corr, _ = stat(y_diag, yhat_diag)
        corr_arr[d] = corr

    avg = np.nanmean(corr_arr)
    if return_arr:
        return avg, corr_arr
    else:
        return avg

def calc_per_class_acc(val_dataloader, model, opt):
    '''Calculate per class accuracy.'''
    print('Class Accuracy Results:', file = opt.log_file)
    assert opt.y_preprocessing in {'diag', 'prcnt'}, "invalid preprocessing: {}".format(opt.y_preprocessing)
    if opt.y_preprocessing != 'prcnt':
        prcntDist_path = osp.join(opt.data_folder, 'prcntDist.npy')
        prcntDist = np.load(prcntDist_path)
        print('prcntDist', prcntDist, file = opt.log_file)

    loss_arr = np.zeros(opt.valN)
    acc_arr = np.zeros(opt.valN)
    acc_c_arr = np.zeros((opt.valN, opt.classes))
    freq_c_arr = np.zeros((opt.valN, opt.classes))

    for i, data in enumerate(val_dataloader):
        if opt.GNN_mode:
            data = data.to(opt.device)
            assert opt.autoencoder_mode
            y = data.contact_map
            yhat = model(data)
            yhat = torch.reshape(yhat, (opt.m, opt.m))
            path = data.path[0]
            minmax = data.minmax
        else:
            x, y, path, minmax = data
            x = x.to(opt.device)
            y = y.to(opt.device)
            path = path[0]
            yhat = model(x)
        loss = opt.criterion(yhat, y).item()
        loss_arr[i] = loss
        y = y.cpu().numpy().reshape((opt.m, opt.m))
        y = un_normalize(y, minmax)
        if opt.loss == 'BCE':
            # using BCE with logits loss, which combines sigmoid into loss
            # so need to do sigmoid here
            yhat = torch.sigmoid(yhat)
        yhat = yhat.cpu().detach().numpy()

        if opt.y_preprocessing == 'prcnt':
            ytrue = y
            yhat = np.argmax(yhat, axis = 1)
        if opt.y_preprocessing == 'diag':
            ytrue = np.load(osp.join(path, 'y_prcnt.npy'))
            yhat = un_normalize(yhat, minmax)
            yhat = percentile_preprocessing(yhat, prcntDist)
        yhat = yhat.reshape((opt.m,opt.m))
        acc = np.sum(yhat == ytrue) / yhat.size
        acc_arr[i] = acc

        for c in range(opt.classes):
            denom = np.sum(ytrue == c)
            freq_c_arr[i, c] = denom / ytrue.size
            num = np.sum(np.logical_and((ytrue == c), (yhat == ytrue)))
            acc = num / denom
            acc_c_arr[i, c] = acc

    acc_result = 'Accuracy: {}% +- {}'.format(round(np.mean(acc_arr) * 100, 3) , round(np.std(acc_arr) * 100, 3) )
    print(acc_result, file = opt.log_file)
    print('Loss: {} +- {}'.format(np.round(np.mean(loss_arr), 3), np.round( np.std(loss_arr), 3)), file = opt.log_file)
    return acc_c_arr, freq_c_arr, acc_result

def compare_PCA(val_dataloader, imagePath, model, opt, count = 5):
    """Computes statistics of 1st PC of contact map"""
    acc_arr = np.zeros(opt.valN)
    rho_arr = np.zeros(opt.valN)
    p_arr = np.zeros(opt.valN)
    pca = PCA()
    model.eval()
    for i, data in enumerate(val_dataloader):
        if opt.GNN_mode:
            data = data.to(opt.device)
            assert opt.autoencoder_mode
            y = data.contact_map
            yhat = model(data)
            path = data.path[0]
            minmax = data.minmax
        else:
            x, y, path, minmax = data
            x = x.to(opt.device)
            y = y.to(opt.device)
            path = path[0]
            yhat = model(x)
        y = y.cpu().numpy().reshape((opt.m, opt.m))
        y = un_normalize(y, minmax)
        if opt.loss == 'BCE':
            # using BCE with logits loss, which combines sigmoid into loss
            # so need to do sigmoid here
            yhat = torch.sigmoid(yhat)
        yhat = yhat.cpu().detach().numpy()

        if opt.y_preprocessing == 'prcnt' and opt.loss == 'cross_entropy':
            yhat = np.argmax(yhat, axis = 1)
        else:
            yhat = un_normalize(yhat, minmax)
        yhat = yhat.reshape((opt.m, opt.m))

        # y
        pca.fit(y/np.std(y, axis = 0))
        comp1_y = pca.components_[0]
        sign1_y = np.sign(comp1_y)

        # yhat
        pca.fit(yhat/np.std(yhat, axis = 0))
        comp1_yhat = pca.components_[0]
        sign1_yhat = np.sign(comp1_yhat)

        # results
        acc = np.sum((sign1_yhat == sign1_y)) / sign1_y.size
        acc_arr[i] = max(acc, 1 - acc)

        corr, pval = spearmanr(comp1_yhat, comp1_y)
        rho_arr[i] = abs(corr)

        corr, pval = pearsonr(comp1_yhat, comp1_y)
        p_arr[i] = abs(corr)

        # for plotting
        if i < count:
            sample = osp.split(path)[-1]
            subpath = osp.join(opt.ofile_folder, sample)
            if not osp.exists(subpath):
                os.mkdir(subpath, mode = 0o755)

            plt.plot(comp1_yhat, label = 'yhat')
            plt.plot(comp1_y, label = 'y')
            plt.legend()
            plt.title('PC 1\nPearson R: corr')
            plt.savefig(osp.join(subpath, 'pc1.png'))
            plt.close()

    results = 'PCA Results:\n' +\
            'Accuracy: {} +- {}\n'.format(np.round(np.mean(acc_arr), 3), np.round(np.std(acc_arr), 3)) +\
            'Spearman R: {} +- {}\n'.format(np.round(np.mean(rho_arr), 3), np.round(np.std(rho_arr), 3))+\
            'Pearson R: {} +- {}\n'.format(np.round(np.mean(p_arr), 3), np.round(np.std(p_arr), 3))
    print(results, file = opt.log_file)
    with open(osp.join(imagePath, 'PCA_results.txt'), 'w') as f:
        f.write(results)

def round_up_by_10(val):
    """Rounds value up to the nearst multiple of 10."""
    assert val > 0, "val too small"
    assert val < 10**10, "val too big"
    mult = 1
    while val > mult:
        mult *= 10
    return mult

def print_time(t0, tf, name = '', file = sys.stdout):
    print(f'{name} time: {np.round(tf - t0, 3)} s', file = file)

def print_size(arr, name = '', file = sys.stdout):
    if arr is None:
        return
    if isinstance(arr, np.ndarray):
        size_b = arr.nbytes
    elif isinstance(arr, sp._arrays.csr_array):
        size_b = arr.data.nbytes
    elif isinstance(arr, sp._csr.csr_matrix):
        size_b = arr.data.nbytes
    else:
        print(f'Unrecognized type {type(arr)}', file = file)
        return
    size_kb = size_b / 1000
    size_mb = size_kb / 1000
    size_gb = size_mb / 1000

    size_names = ['Gigabytes', 'Megabytes', 'kilobytes', 'bytes']
    sizes = [size_gb, size_mb, size_kb, size_b]
    for size, size_name in zip(sizes, size_names):
        if size > 1:
            print(f'{name} size: {np.round(size, 1)} {size_name}', file = file)
            return

def diagonalpool_HiC(HiC, factor):
    HiC_new = np.zeros([int(len(HiC)/factor), int(len(HiC)/factor)])
    for i in range(len(HiC_new)):
        for j in range(len(HiC_new)):
            HiC_new[i,j] = HiC[factor*i, factor*j]+HiC[factor*i+1,factor*j+1]
    return HiC_new

def load_time_dir(dir):
    assert osp.exists(dir), dir
    def load_time_file(file):
        if not osp.exists(file):
            print(f'Warning: {file} does not exist')
            return 0
        t = None
        with open(file) as f:
            for line in f:
                if line.startswith('elapsed'):
                    line_split = line.split()
                    t = int(line_split[1][:-3])
        return t

    eq_log_file = osp.join(dir, 'equilibration/log.log')
    if osp.exists(osp.join(dir, 'equilibration.tar.gz')):
        t_file = tarfile.open(osp.join(dir, 'equilibration.tar.gz'))
        try:
            log = t_file.extract('equilibration/log.log', dir)
        except KeyError:
            print(dir)
            raise

    t_eq = load_time_file(eq_log_file)

    if osp.exists(osp.join(dir, 'production_out/log.log')):
        t_prod = load_time_file(osp.join(dir, 'production_out/log.log'))
    elif osp.exists(osp.join(dir, 'production_out.tar.gz')):
        t_file = tarfile.open(osp.join(dir, 'production_out.tar.gz'))
        log = t_file.extract('production_out/log.log', dir)
        t_prod = load_time_file(osp.join(dir, 'production_out/log.log'))
    else:
        assert osp.exists(osp.join(dir, 'core0')), dir
        t_prod = []
        for f in os.listdir(dir):
            if f.startswith('core'):
                t_prod.append(load_time_file(osp.join(dir, f, 'log.log')))
        t_prod = np.sum(t_prod)


    # cleanup
    for t_file in ['equilibration.tar.gz', 'production_out.tar.gz']:
        if osp.exists(osp.join(dir, t_file)):
            file = t_file.split('.')[0]
            shutil.rmtree(osp.join(dir, file))


    return t_eq + t_prod


def test():
    # test rescale_contact_map
    y = np.array([[0, 1, 2, 3],[4, 5, 6, 7],[8, 9, 10, 11],[12, 13, 14, 15]]).astype(np.float)
    # y = y + y.T
    print(y)
    y_new = diagonal_rescale(y, 2)
    # y /= np.mean(np.diagonal(y))
    print(y_new)
    y = diagonalpool_HiC(y, 2)
    print(y)
    print('---')
    y = np.array([[0, 1, 2],[3, 4, 5],[6, 7, 8]]).astype(np.float)
    print(y)
    y_new = diagonal_rescale(y, 3)
    # y /= np.mean(np.diagonal(y))
    print(y_new)
    y = diagonalpool_HiC(y, 3)
    print(y)

def test2():
    y = np.array([[0, 1, 2, 3],[4, 5, 6, 7],[8, 9, 10, 11],[12, 13, 14, 15]]).astype(np.float)
    y = y + y.T
    print(y)
    y_new = rescale_matrix(y, 2)
    print(y_new)

    y_new = rescale_matrix(y, 2, False)
    print(y_new)

def test3():
    sample_dir = '/home/erschultz/dataset_04_05_23/samples/sample1001'
    dir = osp.join(sample_dir, 'optimize_grid_b_140_phi_0.03-max_ent10/iteration0')
    t = load_time_dir(dir)
    print(t)


if __name__ == '__main__':
    test3()
