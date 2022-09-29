import math

import numpy as np


def calculate_E_S(x, chi):
    if x is None or chi is None:
        return None, None
    s = calculate_S(x, chi)
    e = s_to_E(s)
    return e, s

def calculate_E(x, chi):
    s = calculate_S(x, chi)
    e = s_to_E(s)
    return e

def s_to_E(s):
    if s is None:
        return None

    return s + s.T - np.diag(np.diagonal(s).copy())

def calculate_S(x, chi):
    # zero lower triangle (double check)
    chi = np.triu(chi)

    try:
        s = x @ chi @ x.T
    except ValueError as e:
        print('x', x, x.shape)
        print('chi', chi, chi.shape)
        raise
    return s

def calculate_diag_chi_step(config, diag_chi = None):
    m = config['nbeads']
    if diag_chi is None:
        diag_chi = config['diag_chis']
    diag_bins = len(diag_chi)

    if 'diag_start' in config.keys():
        diag_start = config['diag_start']
    else:
        diag_start = 0

    if 'diag_cutoff' in config.keys():
        diag_cutoff = config['diag_cutoff']
    else:
        diag_cutoff = m

    if 'dense_diagonal_on' in config.keys():
        dense = config['dense_diagonal_on']
    else:
        dense = False

    if dense:
        n_small_bins = config['n_small_bins']
        small_binsize = config['small_binsize']
        big_binsize = config['big_binsize']

    diag_chi_step = np.zeros(m)
    for d in range(diag_cutoff):
        if d < diag_start:
            continue
        d_eff = d - diag_start
        if dense:
            dividing_line = n_small_bins * small_binsize

            if d_eff > dividing_line:
                bin = n_small_bins + math.floor( (d_eff - dividing_line) / big_binsize)
            else:
                bin =  math.floor( d_eff / small_binsize)
        else:
            binsize = m / diag_bins
            bin = int(d_eff / binsize)
        diag_chi_step[d] = diag_chi[bin]

    return diag_chi_step

def calculate_D(diag_chi_continuous):
    m = len(diag_chi_continuous)
    D = np.zeros((m, m))
    for d in range(m):
        rng = np.arange(m-d)
        D[rng, rng+d] = diag_chi_continuous[d]

    return D
