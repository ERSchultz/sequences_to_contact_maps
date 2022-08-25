import math
import multiprocessing
import os
import os.path as osp
import sys
import time

import hicrep
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch
from numba import jit, njit
from scipy.ndimage import uniform_filter
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sympy import solve, symbols

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

class DiagonalPreprocessing():
    '''
    Functions for removing diagonal effect from contact map, y.
    '''
    def get_expected(mean_per_diagonal):
        '''
        Generate matrix of expected contact frequencies.

        Inputs:
            mean_per_diagonal: output of genomic_distance_statistics
        '''
        m = len(mean_per_diagonal)
        expected = np.zeros((m, m))
        for i in range(m):
            for j in range(i, m):
                distance = j - i
                val = mean_per_diagonal[distance]
                expected[i,j] = val
                expected[j,i] = val

        return expected

    def genomic_distance_statistics(y, mode = 'freq', stat = 'mean',
                        zero_diag = False, zero_offset = 1,
                        plot = False, ofile = None, normalize = None):
        '''
        Calculate statistics of contact frequency/probability as a function of genomic distance
        (i.e. along a give diagonal)

        Inputs:
            mode: freq for frequencies, prob for probabilities
            stat: mean to calculate mean, var for variance
            zero_diag: zero digonals up to zero_offset of contact map
            zero_offset: all diagonals up to zero_offset will be zero-d
            plot: True to plot stat_per_diagonal
            ofile: file path to plot to (None for plt.show())
            normalize: divide each value in stat_per_diagonal by value in normalize

        Outputs:
            stat_per_diagonal: numpy array where result[d] is the contact frequency/probability stat at distance d
        '''
        y = y.copy().astype(np.float64)
        if zero_diag:
            y = np.triu(y, zero_offset + 1)
        else:
            y = np.triu(y)

        if mode == 'prob':
            y /= np.max(y)

        if stat == 'mean':
            np_stat = np.mean
        elif stat == 'var':
            np_stat = np.var
        m = len(y)
        distances = range(0, m, 1)
        stat_per_diagonal = np.zeros_like(distances).astype(float)
        for d in distances:
            stat_per_diagonal[d] = np_stat(np.diagonal(y, offset = d))


        if isinstance(normalize, np.ndarray) or isinstance(normalize, float):
            stat_per_diagonal = np.divide(stat_per_diagonal, normalize)

        if plot:
            plt.plot(stat_per_diagonal)
            if ofile is not None:
                plt.savefig(ofile)
            else:
                plt.show()
            plt.close()

        return stat_per_diagonal

    def process(y, mean_per_diagonal, triu = False):
        """
        Inputs:
            y: contact map numpy array or path to .npy file
            mean_per_diagonal: mean contact frequency distribution where
                mean_per_diagonal[distance] is the mean at a given distance
            triu: True if y is 1d array of upper triangle instead of full 2d contact map
                (relatively slow version)

        Outputs:
            result: new contact map
        """
        if isinstance(y, str):
            assert osp.exists(y)
            y = np.load(y)
        y = y.astype('float64')
        if triu:
            y = triu_to_full(y)

        for d in range(len(mean_per_diagonal)):
            expected = mean_per_diagonal[d]
            if expected == 0:
                # this is unlikely to happen
                print(f'WARNING: 0 contacts expected at distance {d}')

        m = len(mean_per_diagonal)
        result = np.zeros((m, m))
        for i in range(m):
            for j in range(i + 1):
                distance = i - j
                expected = mean_per_diagonal[distance]
                if expected > 0:
                    result[i,j] = y[i,j] / expected
                    result[j,i] = result[i,j]

        if triu:
            result = result[np.triu_indices(m)]

        return result

    def process_chunk(dir, mean_per_diagonal, odir = None, chunk_size = 100,
                    jobs = 1, sparse_format = False):
        '''
        Faster version of process when using many contact maps
        but RAM is limited.

        Assumes contact maps are in triu format (i.e. an N x m x m array of N m by m
        contact maps has been reshaped to size N x m*(m+1)/2 )
        '''
        zeros = np.argwhere(mean_per_diagonal == 0)
        if len(zeros) > 0:
            print(f'WARNING: 0 contacts expected at distance {zeros}')
            # replace with minimum observed mean
            mean_per_diagonal[zeros] = np.min(mean_per_diagonal[mean_per_diagonal > 0])

        expected = DiagonalPreprocessing.get_expected(mean_per_diagonal)

        # infer m
        m = len(mean_per_diagonal)

        N = len([f for f in os.listdir(dir) if f.endswith('.npy')])
        files = [f'y_sc_{i}.npy' for i in range(N)]
        if odir is None:
            sc_contacts_diag = np.zeros((N, int(m*(m+1)/2)))
        for i in range(0, N, chunk_size):
            # load sc contacts
            sc_contacts = np.zeros((len(files[i:i + chunk_size]), int(m*(m+1)/2)))
            for j, file in enumerate(files[i:i + chunk_size]):
                sc_contacts[j] = np.load(osp.join(dir, file))

            # process
            result = DiagonalPreprocessing.process_bulk(sc_contacts,
                                                    expected = expected,
                                                    triu = True)
            if odir is None:
                sc_contacts_diag[i:i+chunk_size] = result
            else:
                if jobs > 1:
                    mapping = []
                    for y, f in zip(result, files[i:i + chunk_size]):
                        mapping.append((osp.join(odir, f), y))
                    with multiprocessing.Pool(jobs) as p:
                        p.starmap(np.save, mapping)
                else:
                    for y, f in zip(result, files[i:i + chunk_size]):
                        np.save(osp.join(odir, f), y)


        if odir is None:
            if sparse_format:
                sc_contacts_diag = sp.csr_array(sc_contacts_diag)

            return sc_contacts_diag


    def process_bulk(y_arr, mean_per_diagonal = None, expected = None,
                    triu = False):
        '''Faster version of process when using many contact maps.'''
        y_arr = y_arr.astype('float64', copy = False)

        if expected is None:
            assert mean_per_diagonal is not None
            zeros = np.argwhere(mean_per_diagonal == 0)
            if len(zeros) > 0:
                print(f'WARNING: 0 contacts expected at distance {zeros}')
                # replace with minimum observed mean
                mean_per_diagonal[zeros] = np.min(mean_per_diagonal[mean_per_diagonal > 0])

            expected = DiagonalPreprocessing.get_expected(mean_per_diagonal)

        if triu:
            m, _ = expected.shape
            expected = expected[np.triu_indices(m)]
        if sp.issparse(y_arr):
            result = y_arr.copy()
            result.data /= np.take(expected, result.indices)
        else:
            result = y_arr / expected

        return result

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

def triu_to_full(arr, m = None):
    '''Convert array of upper triangle to symmetric matrix.'''
    # infer m given length of upper triangle
    if m is None:
        l, = arr.shape
        x, y = symbols('x y')
        y=x*(x+1)/2-l
        result=solve(y)
        m = int(np.max(result))

    # need to reconstruct from upper traingle
    y = np.zeros((m, m))
    y[np.triu_indices(m)] = arr
    y += np.triu(y, 1).T

    return y

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

def pearson_round(x, y, stat = 'pearson', round = 2):
    "Wrapper function that combines np.round and pearsonr."
    if stat == 'pearson':
        fn = pearsonr
    elif stat == 'spearman':
        fn = spearmanr
    stat, _ = fn(x, y)
    return np.round(stat, round)

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

class SCC():
    '''
    Class for calculation of SCC as defined by https://pubmed.ncbi.nlm.nih.gov/28855260/
    '''
    def __init__(self):
        self.r_2k_dict = {} # memoized solution for var_stabilized r_2k

    def r_2k(self, x_k, y_k, var_stabilized):
        '''
        Compute r_2k (numerator of pearson correlation)

        Inputs:
            x: contact map
            y: contact map of same shape as x
            var_stabilized: True to use var_stabilized version
        '''
        # x and y are stratums
        if var_stabilized:
            N_k = len(x_k)
            if N_k in self.r_2k_dict:
                result = self.r_2k_dict[N_k]
            else:
                result = np.var(np.arange(1, N_k+1)/N_k)
                self.r_2k_dict[N_k] = result
            return result
        else:
            return math.sqrt(np.var(x_k) * np.var(y_k))

    def mean_filter(x, size):
        return uniform_filter(x, size, mode = 'constant') / (size)**2

    def scc(self, x, y, h = 3, K = None, var_stabilized = False, verbose = False,
            distance = False):
        '''
        Compute scc between contact map x and y.

        Inputs:
            x: contact map
            y: contact map of same shape as x
            h: span of mean filter (width = (1+2h)) (None to skip)
            K: maximum stratum (diagonal) to consider (None for all) (5 Mb recommended)
            var_stabilized: True to use var_stabilized r_2k
            verbose: True to print when nan found
            distance: True to return 1 - scc
        '''
        if len(x.shape) == 1:
            x = triu_to_full(x)
        if len(y.shape) == 1:
            y = triu_to_full(y)

        if h is not None:
            x = SCC.mean_filter(x.astype(np.float64), 1+2*h)
            y = SCC.mean_filter(y.astype(np.float64), 1+2*h)

        if K is None:
            K = len(y) - 2

        num = 0
        denom = 0
        nan_list = []
        for k in range(K):
            # get stratum (diagonal) of contact map
            x_k = np.diagonal(x, k)
            y_k = np.diagonal(y, k)

            N_k = len(x_k)
            r_2k = self.r_2k(x_k, y_k, var_stabilized)
            p_k, _ = pearsonr(x_k, y_k)

            if np.isnan(p_k):
                # nan is hopefully rare so just set to 0
                nan_list.append(k)
                p_k = 0
                if verbose:
                    print(f'k={k}')
                    print(x_k)
                    print(y_k)

            num_i = N_k * r_2k * p_k
            if num_i > 0:
                # negative values should be zero
                num += num_i
            denom += N_k * r_2k

        if len(nan_list) > 0:
            print(f'{len(nan_list)} nans: k = {nan_list}')

        scc = num / denom
        if distance:
            return 1 - scc
        else:
            return scc

def hicrep_scc(fmcool1, fmcool2, h, K, binSize=-1, distance = False):
    '''
    Compute scc between contact map x and y.

    Inputs:
        x: contact map
        y: contact map of same shape as x
        h: span of mean filter (width = (1+2h)) (None to skip)
        K: maximum stratum (diagonal) to consider (None for all) (5 Mb recommended)
        distance: True to return 1 - scc
    '''
    cool1, binSize1 = readMcool(fmcool1, binSize)
    cool2, binSize2 = readMcool(fmcool2, binSize)
    if binSize == -1:
        assert binSize1 == binSize2, f"bin size mismatch: {binSize1} vs {binSize2}"
        binSize = binSize1


    scc = hicrep.hicrepSCC()
    if distance:
        return 1 - scc
    else:
        return scc

def test():
    scc = SCC()
    dir = '/home/erschultz/sequences_to_contact_maps/dataset_05_18_22/samples/sample1'
    x = np.load(osp.join(dir, 'y.npy'))
    dir = osp.join(dir, 'GNN-150-E-diagOn/knone/replicate1/y.npy')
    y = np.load(dir)

    for _ in range(2):
        t0 = time.time()
        print(scc.scc(x,y, var_stabilized = False, verbose = True))
        tf = time.time()
        print_time(t0, tf, 'scc')


if __name__ == '__main__':
    test()
