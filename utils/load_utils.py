import json
import multiprocessing
import os
import os.path as osp
import sys
import time
from shutil import rmtree

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.ndimage import gaussian_filter

from .energy_utils import calculate_E_S, calculate_S, s_to_E
from .utils import (LETTERS, DiagonalPreprocessing, print_size, print_time,
                    triu_to_full)
from .xyz_utils import xyz_load, xyz_to_contact_grid


## load data functions ##
def load_X_psi(sample_folder, throw_exception = True, verbose = False):
    x_file = osp.join(sample_folder, 'x.npy')
    psi_file = osp.join(sample_folder, 'psi.npy')
    if osp.exists(x_file):
        x = np.load(x_file)
        if verbose:
            print(f'x loaded with shape {x.shape}')
    elif throw_exception:
        raise Exception(f'x not found for {sample_folder}')
    else:
        x = None

    if osp.exists(psi_file):
        psi = np.load(psi_file)
        if verbose:
            print(f'psi loaded with shape {psi.shape}')
    else:
        psi = x
        if x is not None:
            print(f'Warning: assuming x == psi for {sample_folder}')

    return x, psi

def load_Y(sample_folder, throw_exception = True):
    y_file = osp.join(sample_folder, 'y.npy')
    y_file2 = osp.join(sample_folder, 'data_out/contacts.txt')
    if osp.exists(y_file):
        y = np.load(y_file)
    elif osp.exists(y_file2):
        y = np.loadtxt(y_file2)
        np.save(y_file, y) # save in proper place
    elif throw_exception:
        raise Exception(f'y not found for {sample_folder}')
    else:
        y = None

    ydiag_file = osp.join(sample_folder, 'y_diag.npy')
    try:
        if osp.exists(ydiag_file):
            ydiag = np.load(ydiag_file)
        elif y is not None:
            meanDist = DiagonalPreprocessing.genomic_distance_statistics(y)
            ydiag = DiagonalPreprocessing.process(y, meanDist)
            np.save(ydiag_file, ydiag) # save in proper place
        else:
            ydiag = None
    except Exception:
        print(f'Exception when loading y_diag for {sample_folder}')
        raise

    return y, ydiag

def load_Y_diag(sample_folder, throw_exception = False):
    ydiag_file = osp.join(sample_folder, 'y_diag.npy')
    if osp.exists(ydiag_file):
        ydiag = np.load(ydiag_file)
    else:
        _, ydiag = load_Y(sample_folder)

    return ydiag


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
            e = None
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
        if psi is not None and chi is not None:
            e, s = calculate_E_S(psi, chi)

        if save and s is not None:
            np.save(osp.join(sample_folder, 's.npy'), s)

    return e, s

def load_all(sample_folder, plot = False, data_folder = None, log_file = None,
                save = False, experimental = False, throw_exception = True):
    '''Loads x, psi, chi, chi_diag, e, s, y, ydiag.'''
    y, ydiag = load_Y(sample_folder, throw_exception = throw_exception)

    if experimental:
        # everything else is None
        return None, None, None, None, None, None, y, ydiag

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

    chi_diag = None
    config_file = osp.join(sample_folder, 'config.json')
    if osp.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
            if "diag_chis" in config:
                chi_diag = np.array(config["diag_chis"])

    e, s = load_E_S(sample_folder, psi, save = save, throw_exception = throw_exception)

    return x, psi, chi, chi_diag, e, s, y, ydiag

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
            if throw_exception:
                raise Exception(f'max it not found for {replicate_folder}')
            else:
                return None

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

def load_final_max_ent_S(replicate_path, max_it_path = None):
    s_path = osp.join(replicate_path, 'resources', 's.npy')
    if osp.exists(s_path):
        s = np.load(s_path)
    else:
        # load x
        x_file = osp.join(replicate_path, 'resources', 'x.npy')
        if osp.exists(x_file):
            x = np.load(x_file)
        else:
            return None

        _, k = x.shape

        # load chi
        chi = load_final_max_ent_chi(k, replicate_path, max_it_path)

        if chi is None:
            raise Exception(f'chi not found: {replicate_path}, {max_it_path}')

        # calculate s
        s = calculate_S(x, chi)

    return s

def save_sc_contacts(xyz, odir, jobs = 5, triu = True, zero_diag = True,
                     sparsify = False, overwrite = False, fmt = 'npy',
                     log_file = sys.stdout):
    '''
    Saves single cell contacts from sample_folder/data_out/output.xyz.

    Inputs:
        xyz (np.array or None): None to load xyz
        odir (str): dir to save to
        jobs (int): Number of jobs
        triu (bool): True to flatten and upper triangularize sc_contact maps
        zero_diag (bool): True to zero diagonal
        sparsify (bool): True to match experimental sparseness
        overwrite (bool): True to overwrite existing data, False to load
        fmt (str): format to save data in
        log_file (file object): location to write to
    '''
    grid_size = 28.7
    if xyz is None:
        dir = osp.split(odir)[0]
        xyz_file = osp.join(dir, 'data_out/output.xyz')
        lammps_file = osp.join(dir, 'traj.dump.lammpstrj')
        if osp.exists(xyz_file):
            xyz = xyz_load(xyz_file, multiple_timesteps = True)
        elif osp.exists(lammps_file):
            xyz = lammps_load(lammps_file)
        else:
            raise Exception(f'xyz not found: {dir}')

    N, _, _ = xyz.shape

    if overwrite and osp.exists(odir):
        rmtree(odir)
    if not osp.exists(odir):
        os.mkdir(odir, mode = 0o755)

    # process sc_contacts
    t0 = time.time()
    print(f'found {multiprocessing.cpu_count()} total CPUs, {len(os.sched_getaffinity(0))} available CPUs, using {jobs}')
    mapping = []
    for i in range(N):
        ofile = osp.join(odir, f'y_sc_{i}.{fmt}')
        if not osp.exists(ofile):
            mapping.append((xyz[i], ofile, grid_size, triu, zero_diag, sparsify, fmt))
    if len(mapping) > 0:
        with multiprocessing.Pool(jobs) as p:
            p.starmap(save_sc_contacts_xyz, mapping)

    tf = time.time()
    print_time(t0, tf, 'sc save', file = log_file)

def save_sc_contacts_xyz(xyz, ofile, grid_size, triu, zero_diag, sparsify, fmt):
    m, _ = xyz.shape

    contact_map = xyz_to_contact_grid(xyz, grid_size, dtype = np.int8)
    if zero_diag:
        np.fill_diagonal(contact_map, 0)
    if sparsify:
        contact_map = sparsify_contact_map(contact_map)
    if triu:
        contact_map = contact_map[np.triu_indices(m)]

    if fmt == 'npy':
        np.save(ofile, contact_map)
    elif fmt == 'txt':
        np.savetxt(ofile, contact_map, fmt = '%d')
    else:
        raise Exception(f'Unrecognized format: {fmt}')

def load_sc_contacts(sample_folder, N_min = None, N_max = None, triu = False,
                    gaussian = False, zero_diag = False, jobs = 1, down_sampling = 1,
                    sparsify = False, correct_diag = False, return_xyz = False,
                    xyz = None, sparse_format = False):
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
        sparse_format: True to use sparse format

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
        print(f'found {multiprocessing.cpu_count()} total CPUs, {len(os.sched_getaffinity(0))} available CPUs, using {jobs}')
        mapping = []
        for i in range(N):
            mapping.append((xyz[i], grid_size, triu, gaussian, zero_diag, sparsify, sparse_format))
        with multiprocessing.Pool(jobs) as p:
            sc_contacts = p.starmap(load_sc_contacts_xyz, mapping)
    else:
        sc_contacts = []
        for i in range(N):
            sc_contacts.append(load_sc_contacts_xyz(xyz[i], grid_size, triu, gaussian, zero_diag, sparsify, sparse_format))
    if sparse_format:
        sc_contacts = sp.vstack(sc_contacts, format = 'csr')
        sc_contacts = sp.csr_array(sc_contacts)
    else:
        sc_contacts = np.array(sc_contacts)
    print(sc_contacts.shape)
    print_size(sc_contacts, 'sc_contacts')


    if correct_diag:
        overall = np.load(osp.join(sample_folder, 'y.npy'))
        mean_per_diag = DiagonalPreprocessing.genomic_distance_statistics(overall, mode = 'prob')
        sc_contacts = DiagonalPreprocessing.process_bulk(sc_contacts, mean_per_diag, triu)

    tf = time.time()
    print(f'Loaded {sc_contacts.shape[0]} sc contacts')
    print_time(t0, tf, 'sc load')
    if return_xyz:
        return sc_contacts, xyz
    return sc_contacts

def load_sc_contacts_xyz(xyz, grid_size, triu, gaussian, zero_diag, sparsify, sparse_format):
    m, _ = xyz.shape

    contact_map = xyz_to_contact_grid(xyz, grid_size, dtype = np.int8)
    if zero_diag:
        np.fill_diagonal(contact_map, 0)
    if sparsify:
        contact_map = sparsify_contact_map(contact_map)
    if gaussian:
        contact_map = gaussian_filter(contact_map, sigma = 4)
    if triu:
        contact_map = contact_map[np.triu_indices(m)]
    if sparse_format:
        contact_map = sp.csr_array(contact_map)

    return contact_map

def sparsify_contact_map_old(contact_map):
    # deprecated, doesn't guarantee row sums are 1
    rng = np.random.default_rng(135)
    temp = np.zeros_like(contact_map)
    # wasn't sure how to do this in place and ensure symmetry
    for i in range(len(contact_map)):
        where = np.argwhere(contact_map[i])
        if len(where) > 0:
            where2 = rng.choice(where.reshape(-1))
            temp[i, where2] = 1
            temp[where2, i] = 1
    return temp

def sparsify_contact_map(contact_map):
    rng = np.random.default_rng(135)
    contact_map = np.triu(contact_map)
    for i in range(len(contact_map)):
        where = np.argwhere(contact_map[i])
        contact_map[i] = 0
        if len(where) > 0:
            where2 = rng.choice(where.reshape(-1))
            contact_map[i, where2] = 1
    return contact_map + contact_map.T
