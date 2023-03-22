import math

import numpy as np
import scipy


def calculate_all_energy(config, psi, chi, diag_chis = None):
    diag_chis_continuous = calculate_diag_chi_step(config, diag_chis)
    D = calculate_D(diag_chis_continuous)
    L = calculate_L(psi, chi)
    S = calculate_S(L, D)
    return L, D, S

def calculate_S(L, D):
    # S is symmetric net energy
    Lp = L + L.T - np.diag(np.diagonal(L).copy())
    if D is None:
        return Lp
    else:
        return Lp + 2*D

def convert_L_to_Lp(L):
    # Lp only requires upper triangle
    return L + L.T - np.diag(np.diagonal(L).copy())

def calculate_Lp(x, chi):
    L = calculate_L(x, chi)
    Lp = convert_L_to_Lp(L)
    return Lp

def calculate_L(psi, chi):
    assert len(chi.shape) == 2, f"chi has shape {chi.shape}"
    m, k = psi.shape
    assert m > k, f'x has shape {psi.shape}, try psi.T'
    # zero lower triangle (double check)
    chi = np.triu(chi)

    try:
        L = psi @ chi @ psi.T
    except ValueError as e:
        print('psi', psi, psi.shape)
        print('chi', chi, chi.shape)
        raise
    L = (L+L.T)/2 # make symmetric

    return L

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

def test():
    m = 100
    I = np.random.choice([0, 1], size=(m*m,)).reshape(m, m)
    I = np.triu(I) + np.triu(I, 1).T
    print(I)

    psi = np.random.rand(m, 5)
    print('psi', psi)

    chi = np.random.rand(5,5)*5
    chi = np.triu(chi)
    print('chi\n', chi)

    Lp = calculate_L_prime(psi, chi)
    L = calculate_L(psi, chi)

    d = np.linspace(0, 2, m)
    D = calculate_D(d)
    print('D\n', D)

    def psi_chi_energy():
        print('psi, chi')
        sum = 0
        for i in range(m):
            for j in range(m):
                sum += I[i,j]* (psi[i] @ chi @ psi[j])
        print(sum)

    def Lp_energy():
        print('Lp')
        sum = 0
        for i in range(m):
            for j in range(m):
                sum += I[i,j]*Lp[i,j]

        print(sum)

    def L_energy():
        print('L')
        sum = 0
        for i in range(m):
            for j in range(i+1):
                sum += I[i,j]*L[i,j]

        print(sum)

    psi_chi_energy()
    Lp_energy()
    L_energy()


if __name__ == '__main__':
    test()
