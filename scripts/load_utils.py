import json
import multiprocessing
import os
import os.path as osp
import sys
import time
from shutil import rmtree

import hicrep
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from pylib.utils.DiagonalPreprocessing import DiagonalPreprocessing
from pylib.utils.energy_utils import (calculate_D, calculate_diag_chi_step,
                                      calculate_L, calculate_S)
from pylib.utils.utils import triu_to_full
from pylib.utils.xyz import xyz_load, xyz_to_contact_grid
from scipy.ndimage import gaussian_filter

from .utils import LETTERS, print_size, print_time


def load_import_log(dir, obj=None):
    import_file = osp.join(dir, 'import.log')
    if not osp.exists(import_file):
        print(f'{import_file} does not exist')
        return

    results = {}
    url = None
    genome = None
    with open(import_file) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('=')
            if line[0].startswith('https') or line[0].endswith('.hic'):
                url = line[0]
            elif line[0] == 'chrom':
                chrom = line[1]
            elif line[0] == 'start':
                start = int(line[1])
                start_mb = start / 1000000
            elif line[0] == 'end':
                end = int(line[1])
                end_mb = end / 1000000
            elif line[0] == 'resolution':
                resolution = int(line[1])
                resolution_mb = resolution / 1000000
            elif line[0] == 'norm':
                norm = line[1]
            elif line[0] == 'genome':
                genome = line[1]

    results['url'] = url
    results['start'] = start
    results['end'] = end
    results['start_mb'] = start_mb
    results['end_mb'] = end_mb
    results['resolution'] = resolution
    results['resolution_mb'] = resolution_mb
    results['norm'] = norm
    results['genome'] = genome
    results['chrom'] = chrom

    if obj is not None:
        obj.url = url
        obj.start = start
        obj.end = end
        obj.start_mb = start_mb
        obj.end_mb = end_mb
        obj.resolution = resolution
        obj.resolution_mb = resolution_mb
        obj.norm = norm
        obj.genome = genome
        obj.chrom = chrom

    return results



## load data functions ##
def load_psi(sample_folder, throw_exception = True, verbose = False):
    x_files = ['x.npy', 'resources/x.npy', 'iteration0/x.npy']
    for f in x_files:
        f = osp.join(sample_folder, f)
        if osp.exists(f):
            x = np.load(f)
            if verbose:
                print(f'x loaded with shape {x.shape}')
            break
    else:
        if throw_exception:
            raise Exception(f'x not found for {sample_folder}')
        else:
            x = None

    assert not osp.exists(osp.join(sample_folder, 'psi.npy')), 'deprecated'

    if x is not None and x.shape[1] > x.shape[0]:
        x = x.T

    return x

def load_Y(sample_folder, throw_exception = True):
    y_file = osp.join(sample_folder, 'y.npy')
    y_file2 = osp.join(sample_folder, 'data_out/contacts.txt')
    y_file3 = osp.join(sample_folder, 'production_out/contacts.txt')
    y_file4 = osp.join(sample_folder, 'y.cool')
    y = None
    if osp.exists(y_file):
        y = np.load(y_file)
    elif osp.exists(y_file2):
        y = np.loadtxt(y_file2)
        np.save(y_file, y) # save in proper place
    elif osp.exists(y_file3):
        y = np.loadtxt(y_file3)
        np.save(y_file, y) # save in proper place
    elif osp.exists(y_file4):
        clr, binsize = hicrep.utils.readMcool(y_file4, -1)
        y = clr.matrix(balance=False).fetch('10')
        np.save(y_file, y) # save in proper place
    else:
        files = os.listdir(osp.join(sample_folder, 'production_out'))
        try:
            max_sweeps = -1
            for f in files:
                if f.startswith('contacts') and f.endswith('.txt'):
                    sweeps = int(f[8:-4])
                    if sweeps > max_sweeps:
                        max_sweeps = sweeps
            y = np.loadtxt(osp.join(sample_folder, 'production_out', f'contacts{max_sweeps}.txt'))
            np.save(y_file, y) # save in proper place
        except Exception as e:
            if throw_exception:
                raise e
            else:
                print(e)

    if y is None and throw_exception:
        raise Exception(f'y not found for {sample_folder}')
    else:
        y = y.astype(float)

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

def load_L(sample_folder, psi = None, chi = None, save = False,
                throw_exception = True):
    '''
    Load L

    Inputs:
        sample_folder: path to sample
        psi: psi np array (None to load if needed)
        chi: chi np array (None to load if needed)
        save: True to save L.npy
        throw_exception: True to throw exception if L missing
    '''
    calc = False # TRUE if need to calculate L

    load_fns = [np.load, np.loadtxt]
    files = [osp.join(sample_folder, i) for i in ['L.npy', 'L_matrix.txt']]
    for f, load_fn in zip(files, load_fns):
        if osp.exists(f):
            L = load_fn(f)
            break
    else:
        L = None
        calc = True

    if calc:
        if psi is None:
            psi = load_psi(sample_folder, throw_exception=throw_exception)
        if chi is None:
            chi = load_chi(sample_folder, throw_exception=throw_exception)
        if psi is not None and chi is not None:
            L = calculate_L(psi, chi)

        if save and L is not None:
            np.save(osp.join(sample_folder, 'L.npy'), L)

    return L

def load_D(sample_folder, throw_exception = True):
    config_file = osp.join(sample_folder, 'config.json')
    with open(config_file, 'r') as f:
        config = json.load(f)
        diag_chi_step = calculate_diag_chi_step(config)

    D = calculate_D(diag_chi_step)
    return D

def load_S(sample_folder, throw_exception = True):
    L = load_L(sample_folder, throw_exception=throw_exception)
    D = load_D(sample_folder, throw_exception=throw_exception)
    S = calculate_S(L, D)

    return S


def load_all(sample_folder, plot = False, data_folder = None, log_file = None,
                save = False, experimental = False, throw_exception = True):
    '''Loads x, psi, chi, chi_diag, L, y, ydiag.'''
    y, ydiag = load_Y(sample_folder, throw_exception = throw_exception)

    if experimental:
        # everything else is None
        return None, None, None, None, None, None, y, ydiag

    x = load_psi(sample_folder, throw_exception = throw_exception)
    # x = x.astype(float)

    if plot and x is not None:
        m, k = x.shape
        for i in range(k):
            plt.plot(x[:, i])
            plt.title(r'$X$[:, {}]'.format(i))
            plt.savefig(osp.join(sample_folder, 'x_{}'.format(i)))
            plt.close()

    chi = load_chi(sample_folder, throw_exception)
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

    L = load_L(sample_folder, x, chi, save = save,
                    throw_exception = throw_exception)

    return x, chi, chi_diag, L, y, ydiag

def load_chi(dir, throw_exception=True):
    if osp.exists(osp.join(dir, 'chis.txt')):
        chi = np.loadtxt(osp.join(dir, 'chis.txt'))
        chi = np.atleast_2d(chi)[-1]
        return triu_to_full(chi)
    elif osp.exists(osp.join(dir, 'chis.npy')):
        chi = np.load(osp.join(dir, 'chis.npy'))
        return chi
    elif osp.exists(osp.join(dir, 'config.json')):
        with open(osp.join(dir, 'config.json'), 'rb') as f:
            config = json.load(f)

        if 'chis' in config:
            chi = np.array(config['chis'])
            return chi


    if throw_exception:
        raise Exception(f'chi not found for {dir}')

    return None

def load_max_ent_chi(k, path, throw_exception = True):

    config_file = osp.join(path, 'config.json')
    if osp.exists(config_file):
        with open(config_file, 'rb') as f:
            config = json.load(f)
    else:
        return None

    try:
        chi = config['chis']
        chi = np.array(chi)
    except:
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

def get_final_max_ent_folder(replicate_folder, throw_exception = True, return_it = False):
    '''Find final max ent iteration within replicate folder.'''
    max_it = -1
    for file in os.listdir(replicate_folder):
        if osp.isdir(osp.join(replicate_folder, file)) and 'iteration' in file:
            start = file.find('iteration')+9
            it = int(file[start:])
            if it > max_it:
                max_it = it

    if max_it < 0:
        if throw_exception:
            raise Exception(f'max it not found for {replicate_folder}')
        else:
            return None

    final_folder = osp.join(replicate_folder, f'iteration{max_it}')
    if return_it:
        return final_folder, max_it
    return final_folder

def get_converged_max_ent_folder(replicate_folder, conv_defn, throw_exception=True):
    if conv_defn == 'strict':
        eps = 1e-3
    elif conv_defn == 'normal':
        eps = 1e-2
    else:
        raise Exception(f'Unrecognized conv_defn: {conv_defn}')

    conv_file = osp.join(replicate_folder, 'convergence.txt')
    assert osp.exists(conv_file), f'conv_file does not exists: {conv_file}'
    conv = np.atleast_1d(np.loadtxt(conv_file))
    converged_it = None
    for j in range(1, len(conv)):
        diff = conv[j] - conv[j-1]
        if np.abs(diff) < eps and conv[j] < conv[0]:
            converged_it = j
            break

    if converged_it is not None:
        final = osp.join(replicate_folder, f'iteration{j}')
        return final
    elif throw_exception:
        raise Exception(f'{replicate_folder} did not converge')
    else:
        return None

def load_max_ent_D(path):
    if path is None:
        return None

    final = get_final_max_ent_folder(path)

    config_file = osp.join(final, 'config.json')
    if osp.exists(config_file):
        with open(config_file, 'rb') as f:
            config = json.load(f)
    else:
        return None

    diag_chis_step = calculate_diag_chi_step(config)

    D = calculate_D(diag_chis_step)
    return D

def load_max_ent_L(path, throw_exception=False):
    if path is None:
        return None
    # load x
    x = load_psi(path, throw_exception=throw_exception)
    if x is None:
        return

    _, k = x.shape
    # load chi
    chi = load_chi(path, throw_exception)

    if chi is None and throw_exception:
        raise Exception(f'chi not found: {path}')

    # calculate s
    L = calculate_L(x, chi)

    if L is None and throw_exception:
        raise Exception(f'L is None: {path}')

    return L

def load_max_ent_S(path, throw_exception=False):
    L = load_max_ent_L(path, throw_exception = throw_exception)
    D = load_max_ent_D(path)
    return calculate_S(L, D)


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
    print(f'found {multiprocessing.cpu_count()} total CPUs, ',
            f'{len(os.sched_getaffinity(0))} available CPUs, using {jobs}')
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
        N_min: minimum sc_contact to load (early sweeps may not be equilibrated)
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
        print(f'found {multiprocessing.cpu_count()} total CPUs, ',
                f'{len(os.sched_getaffinity(0))} available CPUs, using {jobs}')
        mapping = []
        for i in range(N):
            mapping.append((xyz[i], grid_size, triu, gaussian, zero_diag,
                            sparsify, sparse_format))
        with multiprocessing.Pool(jobs) as p:
            sc_contacts = p.starmap(load_sc_contacts_xyz, mapping)
    else:
        sc_contacts = []
        for i in range(N):
            sc_contacts.append(load_sc_contacts_xyz(xyz[i], grid_size, triu,
                                                    gaussian, zero_diag,
                                                    sparsify, sparse_format))
    if sparse_format:
        sc_contacts = sp.vstack(sc_contacts, format = 'csr')
        sc_contacts = sp.csr_array(sc_contacts)
    else:
        sc_contacts = np.array(sc_contacts)
    print(sc_contacts.shape)
    print_size(sc_contacts, 'sc_contacts')


    if correct_diag:
        overall = np.load(osp.join(sample_folder, 'y.npy'))
        DP = DiagonalPreprocessing()
        mean_per_diag = DP.genomic_distance_statistics(overall, mode = 'prob')
        sc_contacts = DP.process_bulk(sc_contacts, mean_per_diag, triu)

    tf = time.time()
    print(f'Loaded {sc_contacts.shape[0]} sc contacts')
    print_time(t0, tf, 'sc load')
    if return_xyz:
        return sc_contacts, xyz
    return sc_contacts

def load_sc_contacts_xyz(xyz, grid_size, triu, gaussian, zero_diag,
                            sparsify, sparse_format):
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

def load_contact_map(file, chrom = None, resolution = None):
    file_type = file.split('.')[1]
    if file_type == 'cool':
        clr, binsize = hicrep.utils.readMcool(file, -1)
        if resolution is not None:
            assert resolution == binsize, f"{resolution} != {binsize}"
        if chrom is None:
            y = []
            for chrom in clr.chromnames:
                y.append(clr.matrix(balance=False).fetch(f'{chrom}'))
        else:
            y = clr.matrix(balance=False).fetch(f'{chrom}')
    elif file_type == 'mcool':
        assert resolution is not None
        clr, _ = hicrep.utils.readMcool(file, resolution)
        if chrom is None:
            y = []
            for chrom in clr.chromnames:
                y.append(clr.matrix(balance=False).fetch(f'{chrom}'))
        else:
            y = clr.matrix(balance=False).fetch(f'{chrom}')
    elif file_type == 'npy':
        y = np.load(file)
    else:
        raise Exception(f'Unaccepted file type: {file_type}')

    return y
