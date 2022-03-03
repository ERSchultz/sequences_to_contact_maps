import json
import os.path as osp
import time

import numpy as np
from scipy.sparse.csgraph import laplacian
from sklearn.metrics import pairwise_distances, pairwise_distances_chunked
from sympy import solve, symbols
from utils.load_utils import load_sc_contacts
from utils.plotting_utils import (plot_sc_contact_maps_inner, plot_top_PCs,
                                  plotContactMap)
from utils.utils import (calc_dist_strat_corr, pearson_round, print_size,
                         print_time)
from utils.xyz_utils import xyz_load, xyz_to_contact_grid


def sort_laplacian(A_tilde, sc_contacts):
    w, v = np.linalg.eig(A_tilde)
    v = v[:, np.argsort(w)]
    w = np.sort(w)
    print('w', w[:10])

    # get first nonzero eigenvector of A_tilde
    lmbda = w[0]; i = 0
    while lmbda <= 1e-10:
        i += 1
        lmbda = w[i]
    where = np.argsort(v[:, i])

    # sort sc_contacts
    sc_contacts = sc_contacts[where, :]

    # merge adjacent contacts
    # modify in place to preserve RAM
    # I did test this - it works
    N, _ = sc_contacts.shape
    for i in range(N):
        if i == 0:
            new = np.mean(sc_contacts[0:2, :], axis = 0)
        elif i == N:
            new = np.mean(sc_contacts[-2:N, :], axis = 0)
        else:
            new = np.mean(sc_contacts[i-1:i+2, :], axis = 0)
        if i > 0:
            sc_contacts[i-1, :] = new_prev
        new_prev = new.copy()
    sc_contacts[N-1] = new_prev

    # undo sort
    order = np.argsort(where)
    sc_contacts = sc_contacts[order]

    return sc_contacts, where

def main():
    dir = '/home/eric/dataset_test/samples/sample91'
    odir = osp.join(dir, 'sc_contact_test')
    sc_contacts = load_sc_contacts(dir, N_min = 0, N_max = None, triu = True,
                                    gaussian = True, zero_diag = True, jobs = 8,
                                    down_sampling = 1, diagonal_preprocessing = False)
    print_size(sc_contacts, 'sc_contacts')
    N, _ = sc_contacts.shape
    for i in range(2):
        t0 = time.time()
        A = pairwise_distances(sc_contacts, sc_contacts, metric = 'correlation')
        tf = time.time()
        print_time(t0, tf, 'distances')
        A = np.exp(-1 * A**2 / 100)
        plotContactMap(A, ofile = osp.join(odir, f'A{i}.png'),
                        vmin = 'min', vmax = 'max')
        np.savetxt(osp.join(odir, f'A_{i}.txt'), A, fmt='%.2f')

        t0 = time.time()
        A_tilde = laplacian(A, normed = True)
        plotContactMap(A_tilde, ofile = osp.join(odir, f'A_tilde_{i}.png'),
                        vmin = np.min(A_tilde), vmax = np.max(A_tilde), cmap = 'blue-red')
        np.savetxt(osp.join(odir, f'A_tilde_{i}.txt'), A_tilde, fmt='%.2f')
        sc_contacts, order = sort_laplacian(A_tilde, sc_contacts)
        np.savetxt(osp.join(odir, f'order_{i}.txt'), order, fmt='%i')
        print('order corr: ', pearson_round(order, np.arange(0, N, 1), stat = 'spearman'))
        tf = time.time()
        print_time(t0, tf, 'laplacian')

        t0 = time.time()
        plot_sc_contact_maps_inner(sc_contacts, osp.join(odir, f'iteration_{i}'),
                                    count = 10, jobs = 10)
        tf = time.time()
        print_time(t0, tf, 'plot')
        print('\n')


if __name__ == '__main__':
    main()
