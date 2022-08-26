import time

import numpy as np
from scipy.ndimage import uniform_filter
from scipy.stats import pearsonr


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
            # var_stabilized computes variance of ranks
            N_k = len(x_k)
            if N_k in self.r_2k_dict:
                result = self.r_2k_dict[N_k]
            else:
                # variance is permutation invariant, so just use np.arange instead of computing ranks
                # var(rank(x_k)) = var(rank(y_k)), so no need to compute for each
                # this allows for memoizing the solution via self.r_2k_dict (solution depends only on N_k)
                # memoization offers a marginal speedup when computing lots of scc's
                result = np.var(np.arange(1, N_k+1)/N_k, ddof = 1)
                self.r_2k_dict[N_k] = result
            return result
        else:
            return math.sqrt(np.var(x_k) * np.var(y_k))

    def scc_file(self, xfile, yfile, h = 1, K = None, var_stabilized = False, verbose = False,
                distance = False):
        '''Wrapper for scc that takes file path as input. Must be .npy file.'''
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

    def scc(self, x, y, h = 1, K = 10, var_stabilized = True, verbose = False,
            distance = False):
        '''
        Compute scc between contact map x and y.

        Inputs:
            x: contact map
            y: contact map of same shape as x
            h: span of mean filter (width = (1+2h)) (None or 0 to skip)
            K: maximum stratum (diagonal) to consider (5 Mb recommended)
            var_stabilized: True to use var_stabilized r_2k (default = True)
            verbose: True to return diagonal correlations and weights
            distance: True to return 1 - scc
        '''
        if h is not None and h > 0:
            x = uniform_filter(x.astype(np.float64), 1+2*h, mode = 'constant')
            y = uniform_filter(y.astype(np.float64), 1+2*h, mode = 'constant')

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

        # if verbose and len(nan_list) > 0:
        #     print(f'{len(nan_list)} nans: k = {nan_list}')

        if distance:
            scc =  1 - scc

        if verbose:
            return scc, p_arr, w_arr
        else:
            return scc

def test():
    scc = SCC()
    x = np.random.rand(1000, 1000)
    y = np.random.rand(1000, 1000)

    t0 = time.time()
    for _ in range(100):
        val, p, w = scc.scc(x, y, verbose = True)
    tf = time.time()
    print(val)
    print(p)
    print(w)

    print('time: ', tf - t0)

if __name__ == '__main__':
    test()
