import json
import multiprocessing
import os
import os.path as osp
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from .energy_utils import calculate_E_S, calculate_S, s_to_E
from .utils import (LETTERS, diagonal_preprocessing,
                    diagonal_preprocessing_bulk, genomic_distance_statistics,
                    print_time, triu_to_full)
from .xyz_utils import xyz_load, xyz_to_contact_grid


## load data functions ##
def load_X_psi(sample_folder, throw_exception = True):
    x_file = osp.join(sample_folder, 'x.npy')
    psi_file = osp.join(sample_folder, 'psi.npy')
    if osp.exists(x_file):
        x = np.load(x_file)
        print(f'x loaded with shape {x.shape}')
    elif throw_exception:
        raise Exception(f'x not found for {sample_folder}')
    else:
        x = None

    if osp.exists(psi_file):
        psi = np.load(psi_file)
        print(f'psi loaded with shape {psi.shape}')
    else:
        psi = x
        print(f'Warning: assuming x == psi for {sample_folder}')

    return x, psi

def load_Y(sample_folder, throw_exception = True):
    y_file = osp.join(osp.join(sample_folder, 'y.npy'))
    if osp.exists(y_file):
        y = np.load(y_file)
    elif throw_exception:
        raise Exception(f'y not found for {sample_folder}')
    else:
        y = None

    ydiag_file = osp.join(sample_folder, 'y_diag.npy')
    if osp.exists(ydiag_file):
        ydiag = np.load(ydiag_file)
    elif throw_exception:
        raise Exception(f'ydiag not found for {sample_folder}')
    else:
        ydiag = None

    return y, ydiag

def load_E_S(sample_folder, psi = None, chi = None, save = False, throw_exception = True):
    '''
    Load E and S.

    Inputs:
        sample_folder: path to sample
        psi: psi np array (None to load if needed)
        chi: chi np array (None to load if needed)
        save: True to save s.npy
        throw_exception: True to throw exception if E and S missing
    '''
    calc = False # TRUE if need to calculate e or s matrix

    load_fns = [np.load, np.loadtxt]
    s_files = [osp.join(sample_folder, i) for i in ['s.npy', 's_matrix.txt']]
    for s_file, load_fn in zip(s_files, load_fns):
        if osp.exists(s_file):
            s = load_fn(s_file)
            break
    else:
        s = None
        calc = True

    e_files = [osp.join(sample_folder, i) for i in ['e.npy', 'e_matrix.txt']]
    for e_file, load_fn in zip(e_files, load_fns):
        if osp.exists(e_file):
            e = load_fn(e_file)
            break
    else:
        if s is not None:
            e = s_to_E(s)
        else:
            calc = True

    if calc:
        if psi is None:
            _, psi = load_X_psi(sample_folder, throw_exception=throw_exception)
        if chi is None:
            chi_path = osp.join(sample_folder, 'chis.npy')
            if osp.exists(chi_path):
                chi = np.load(chi_path)
            else:
                chi = None
        e, s = calculate_E_S(psi, chi)

        if save and s is not None:
            np.save(osp.join(sample_folder, 's.npy'), s)

    return e, s

def load_all(sample_folder, plot = False, data_folder = None, log_file = None,
                save = False, experimental = False, throw_exception = True):
    '''Loads x, psi, chi, e, s, y, ydiag.'''
    y, ydiag = load_Y(sample_folder, throw_exception = throw_exception)

    if experimental:
        # everything else is None
        return None, None, None, None, None, y, ydiag

    x, psi = load_X_psi(sample_folder, throw_exception = throw_exception)
    # x = x.astype(float)

    if plot and x is not None:
        m, k = x.shape
        for i in range(k):
            plt.plot(x[:, i])
            plt.title(r'$X$[:, {}]'.format(i))
            plt.savefig(osp.join(sample_folder, 'x_{}'.format(i)))
            plt.close()

    chi = None
    if data_folder is not None:
        chi_path1 = osp.join(data_folder, 'chis.npy')
    chi_path2 = osp.join(sample_folder, 'chis.npy')
    if data_folder is not None and osp.exists(chi_path1):
        chi = np.load(chi_path1)
    elif osp.exists(chi_path2):
        chi = np.load(chi_path2)
    elif throw_exception:
        raise Exception('chi not found at {} or {}'.format(chi_path1, chi_path2))
    if chi is not None:
        chi = chi.astype(np.float64)
        if log_file is not None:
            print('Chi:\n', chi, file = log_file)

    e, s = load_E_S(sample_folder, psi, save = save)

    return x, psi, chi, e, s, y, ydiag

def load_final_max_ent_chi(k, replicate_folder = None, max_it_folder = None,
                throw_exception = True):
    if max_it_folder is None:
        # find final it
        max_it = -1
        for file in os.listdir(replicate_folder):
            if osp.isdir(osp.join(replicate_folder, file)) and file.startswith('iteration'):
                it = int(file[9:])
                if it > max_it:
                    max_it = it

        if max_it < 0:
            raise Exception(f'max it not found for {replicate_folder}')

        max_it_folder = osp.join(replicate_folder, f'iteration{max_it}')

    config_file = osp.join(max_it_folder, 'config.json')
    if osp.exists(config_file):
        with open(config_file, 'rb') as f:
            config = json.load(f)
    else:
        return None

    chi = np.zeros((k,k))
    for i, bead_i in enumerate(LETTERS[:k]):
        for j in range(i,k):
            bead_j = LETTERS[j]
            try:
                chi[i,j] = config[f'chi{bead_i}{bead_j}']
            except KeyError:
                if throw_exception:
                    print(f'config_file: {config_file}')
                    print(config)
                    raise
                else:
                    return None

    return chi

def load_final_max_ent_S(k, replicate_path, max_it_path = None):
    s_path = osp.join(replicate_path, 'resources', 's.npy')
    if osp.exists(s_path):
        s = np.load(s_path)
    else:
        # load x
        x_file = osp.join(replicate_path, 'resources', 'x.npy')
        if osp.exists(x_file):
            x = np.load(x_file)

        # load chi
        chi = load_final_max_ent_chi(k, replicate_path, max_it_path)

        if chi is None:
            raise Exception(f'chi not found: {replicate_path}, {max_it_path}')

        # calculate s
        s = calculate_S(x, chi)

    return s

def load_sc_contacts(sample_folder, N_min = None, N_max = None, triu = False,
                    gaussian = False, zero_diag = False, jobs = 1, down_sampling = 1,
                    sparsify = False, correct_diag = False, return_xyz = False,
                    xyz = None):
    '''
    Load single cell contacts from sample_folder/data_out/output.xyz.

    Inputs:
        sample_folder: sample to load data for
        N_min: minimum sc_contact to load (because early sweeps may not be equillibrated)
        N_max: maximum sc_contact to load (for computational efficiency)
        triu: True to flatten and upper triangularize sc_contact maps
        gaussian: True to apply gaussian filter
        zero_diag: True to zero diagonal
        down_sampling (int): down sample by given value
        sparsify (bool): True to match experimental sparseness
        diagonal_preprocessing: process diagonal based on overall contact map
        return_xyz: True to return xyz as well
        xyz: None to load xyz

    Outputs:
        sc_contacts: (N, (m+1)*m/2) if triu, else (N, m, m)
    '''
    config_file = osp.join(sample_folder, 'config.json')
    if osp.exists(config_file):
        with open(config_file, 'rb') as f:
            config = json.load(f)
            grid_size = float(config['grid_size'])
    else:
        grid_size = 28.7

    # load xyz
    if xyz is None:
        xyz = xyz_load(osp.join(sample_folder, 'data_out/output.xyz'),
                        multiple_timesteps=True, save = True, N_min = N_min,
                        N_max = N_max, down_sampling = down_sampling)

    # set up N, m, and xyz
    N, _, _ = xyz.shape

    # process sc_contacts
    t0 = time.time()
    if jobs > 1:
        mapping = []
        for i in range(N):
            mapping.append((xyz[i], grid_size, triu, gaussian, zero_diag, sparsify))
        with multiprocessing.Pool(jobs) as p:
            sc_contacts = p.starmap(load_sc_contacts_xyz, mapping)
    else:
        sc_contacts = list(map(load_sc_contacts_xyz, xyz, [grid_size]*N, [triu]*N, [gaussian]*N, [zero_diag]*N, [sparsify]*N))
    sc_contacts = np.array(sc_contacts)

    if correct_diag:
        overall = np.load(osp.join(sample_folder, 'y.npy'))
        mean_per_diag = genomic_distance_statistics(overall, mode = 'prob')
        sc_contacts = diagonal_preprocessing_bulk(sc_contacts, mean_per_diag, triu)

    tf = time.time()
    print_time(t0, tf, 'load')

    print(f'Loaded {len(sc_contacts)} sc contacts')
    if return_xyz:
        return sc_contacts, xyz
    return sc_contacts

def load_sc_contacts_xyz(xyz, grid_size, triu, gaussian, zero_diag, sparsify):
    m, _ = xyz.shape

    contact_map = xyz_to_contact_grid(xyz, grid_size)
    if zero_diag:
        np.fill_diagonal(contact_map, 0)
    if sparsify:
        temp_contact_map = np.zeros_like(contact_map)
        # wasn't sure how to do this in place and ensure symmetry
        for i in range(m):
            where = np.argwhere(contact_map[i])
            if len(where) > 0:
                where2 = np.random.choice(where.reshape(-1))
                temp_contact_map[i, where2] = 1
                temp_contact_map[where2, i] = 1
        contact_map = temp_contact_map
    if gaussian:
        contact_map = gaussian_filter(contact_map, sigma = 4)
    if triu:
        contact_map = contact_map[np.triu_indices(m)]

    return contact_map
