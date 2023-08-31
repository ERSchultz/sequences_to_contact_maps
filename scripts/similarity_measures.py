import math
import multiprocessing
import os
import os.path as osp

import hicrep
import numpy as np
import scipy
import sklearn.metrics
from scipy.stats import pearsonr, zscore


# similarity measures
class SCC():
    """
    Calculate Stratified Correlation Coefficient (SCC)
    as defined by https://pubmed.ncbi.nlm.nih.gov/28855260/.
    """
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
            # var_stabilized computes variance of ranks
            N_k = len(x_k)
            if N_k in self.r_2k_dict:
                result = self.r_2k_dict[N_k]
            else:
                # variance is permutation invariant, so just use np.arange instead of computing ranks
                # futher var(rank(x_k)) = var(rank(y_k)), so no need to compute for each
                # this allows us to memoize the solution via self.r_2k_dict (solution depends only on N_k)
                # memoization offers a marginal speedup when computing lots of scc's
                result = np.var(np.arange(1, N_k+1)/N_k, ddof = 1)
                self.r_2k_dict[N_k] = result
            return result
        else:
            return math.sqrt(np.var(x_k) * np.var(y_k))

    def scc_file(self, xfile, yfile, h=1, K=None, var_stabilized=True,
                verbose=False, distance=False):
        '''
        Wrapper for scc that takes file path as input. Must be .npy file.

        Inputs:
            xfile: file path to contact map
            yfile: file path to contact map
            ...: see self.scc()
            chr: chromosome if mcool files
            resolution: resoultion if mcool files
        '''
        if xfile == yfile:
            # no need to compute
            result = 1 - distance
            if verbose:
                return result, None, None
            else:
                return result

        x = np.load(xfile)
        y = np.load(yfile)
        return self.scc(x, y, h, K, var_stabilized, verbose, distance)

    def mean_filter(self, x, size):
        return scipy.ndimage.uniform_filter(x, size, mode="constant") / (size) ** 2

    def scc(self, x, y, h=1, K=None, var_stabilized=False, verbose=False,
            debug=False, distance=False):
        '''
        Compute scc between contact map x and y.

        Inputs:
            x: contact map
            y: contact map of same shape as x
            h: span of mean filter (width = (1+2h)) (None or 0 to skip)
            K: maximum stratum (diagonal) to consider (5 Mb recommended)
            var_stabilized: True to use var_stabilized r_2k (default = True)
            verbose: True to print when nan found
            debug: True to return p_arr and w_arr
            distance: True to return 1 - scc
        '''
        if len(x.shape) == 1:
            x = triu_to_full(x)
        if len(y.shape) == 1:
            y = triu_to_full(y)

        if K is None:
            K = len(y) - 2

        x = self.mean_filter(x.astype(np.float64), 1+2*h)
        y = self.mean_filter(y.astype(np.float64), 1+2*h)

        nan_list = []
        p_arr = []
        w_arr = []
        for k in range(K):
            # get stratum (diagonal) of contact map
            x_k = np.diagonal(x, k)
            y_k = np.diagonal(y, k)

            # filter to subset of diagonals where at least 1 is nonzero
            # i.e if x_k[i] == y_k[i] == 0, ignore element i
            # use 1e-12 for numerical stability
            x_zeros = np.argwhere(abs(x_k)<1e-12)
            y_zeros = np.argwhere(abs(y_k)<1e-12)
            both_zeros = np.intersect1d(x_zeros, y_zeros)
            mask = np.ones(len(x_k), bool)
            mask[both_zeros] = 0
            x_k = x_k[mask]
            y_k = y_k[mask]

            N_k = len(x_k)

            if N_k > 1:
                p_k, _ = pearsonr(x_k, y_k)

                if np.isnan(p_k):
                    # ignore nan
                    nan_list.append(k)
                else:
                    p_arr.append(p_k)
                    r_2k = self.r_2k(x_k, y_k, var_stabilized)
                    w_k = N_k * r_2k
                    w_arr.append(w_k)

        w_arr = np.array(w_arr)
        p_arr = np.array(p_arr)

        scc = np.sum(w_arr * p_arr / np.sum(w_arr))

        if verbose and len(nan_list) > 0:
            print(f'{len(nan_list)} nans: k = {nan_list}')

        if distance:
            scc =  1 - scc

        if debug:
            return scc, p_arr, w_arr
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

class InnerProduct():
    '''Based off of InnerProduct from https://github.com/liu-bioinfo-lab/scHiCTools'''
    def __init__(self, dir = None, files = None, K = 10, jobs = 10,
                resolution = None, chr = None):
        '''
        Inputs:
            dir: directory containing input data (required if files is None)
            files: list of input file paths (required if dir is None)
            K: maximum stratum to consider
            jobs: number of jobs for multiprocessing
            resolution: resolution of input data (required if .mcool files)
            chr: chromosome of input data (required if .mcool files)
        '''
        self.K = K
        self.resolution = resolution
        self.chr = chr
        if dir is not None:
            self.files = [osp.join(dir, f) for f in os.listdir(dir) if f.endswith('.npy')]
        else:
            assert files is not None
            self.files = files


        with multiprocessing.Pool(jobs) as p:
            self.zscores = p.map(self.get_zscore_feature, self.files)
        self.zscores = np.array(self.zscores)

    def get_distance_matrix(self):
        inner = self.zscores.dot(self.zscores.T) / self.zscores.shape[1]
        inner[inner > 1] = 1
        inner[inner < -1] = -1
        distance_mat = np.sqrt(2 - 2 * inner)

        return distance_mat

    def get_zscore_feature(self, file):
        y = load_contact_map(file, self.chr, self.resolution)
        if len(y) == 1:
            y = triu_to_full(y)
        zscores = []
        for k in range(self.K):
            y_k = np.diagonal(y, k)
            z = zscore(y_k)
            z[np.isnan(z)] = 0
            zscores.append(z)
        return np.concatenate(zscores)
