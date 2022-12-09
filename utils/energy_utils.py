import math

import numpy as np
import scipy


def calculate_all_energy(config, x, chi):
    diag_chis_continuous = calculate_diag_chi_step(config)
    D = calculate_D(diag_chis_continuous)
    E, S = calculate_E_S(x, chi)
    ED = calculate_net_energy(S, D)
    return S, D, E, ED

def calculate_SD(S, D):
    S_sym = (S + S.T)/2 # double check that S is symmetric
    SD = S_sym + D + np.diag(np.diagonal(D.copy()))
    return SD

def calculate_SD_ED(S, D):
    SD = calculate_SD(S, D)
    ED = s_to_E(SD)
    return SD, ED

def calculate_net_energy(S, D):
    SD = calculate_SD(S, D)
    ED = s_to_E(SD)
    return ED

def calculate_E_S(x, chi):
    if x is None or chi is None:
        return None, None
    S = calculate_S(x, chi)
    E = s_to_E(S)
    return E, S

def calculate_E(x, chi):
    S = calculate_S(x, chi)
    E = s_to_E(S)
    return E

def s_to_E(S):
    if S is None:
        return None

    return S + S.T - np.diag(np.diagonal(S).copy())

def calculate_S(x, chi):
    assert len(chi.shape) == 2, f"chi has shape {chi.shape}"
    m, k = x.shape
    assert m > k, f'x has shape {x.shape}, try x.T'
    # zero lower triangle (double check)
    chi = np.triu(chi)

    try:
        S = x @ chi @ x.T
    except ValueError as e:
        print('x', x, x.shape)
        print('chi', chi, chi.shape)
        raise
    return S

def calculate_diag_chi_step(config, diag_chi = None):
    m = config['nbeads']
    if diag_chi is None:
        diag_chi = config['diag_chis']
    diag_bins = len(diag_chi)

    if diag_bins == m:
        return diag_chi

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
        if 'n_small_bins' in config.keys():
            n_small_bins = config['n_small_bins']
            small_binsize = config['small_binsize']
            big_binsize = config['big_binsize']
        else:
            # soren compatibility
            n_small_bins = int(config['dense_diagonal_loading'] * diag_bins)
            n_big_bins = diag_bins - n_small_bins
            m_eff = diag_cutoff - diag_start # number of beads with nonzero interaction
            dividing_line = m_eff * config['dense_diagonal_cutoff']
            small_binsize = int(dividing_line / (n_small_bins))
            big_binsize = int((m_eff - dividing_line) / n_big_bins)

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
    return scipy.linalg.toeplitz(diag_chi_continuous)

def main():
    d = np.array([1,2,3])
    print(calculate_D(d))

if __name__ == '__main__':
    main()
