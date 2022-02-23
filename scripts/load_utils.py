import json
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np

from .utils import LETTERS, calculate_E_S, calculate_S, s_to_E


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

def load_all(sample_folder, plot = False, data_folder = None, log_file = None, save = False, experimental = False, throw_exception = True):
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

def load_final_max_ent_chi(k, replicate_folder = None, max_it_folder = None, throw_exception = True):
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
