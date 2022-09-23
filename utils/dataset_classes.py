import json
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import uniform_filter
from torch.utils.data import Dataset

from .argparse_utils import ArgparserConverter
from .utils import DiagonalPreprocessing, get_diag_chi_step


def make_dataset(dir, minSample = 0, maxSample = float('inf'), verbose = False,
                samples = None, prefix = 'sample'):
    """
    Make list data file paths.

    Inputs:
        dir: data source directory
        minSample: ignore samples < minSample
        maxSample: ignore samples > maxSample
        verbose: True for verbose mode
        samples: list/set of samples to include, None for all samples
        prefix: only folders starting with prefix will be considered

    Outputs:
        data_file_arr: list of data file paths
    """
    data_file_arr = []
    samples_dir = osp.join(dir, 'samples')
    l = len(prefix)
    sample_ids = [int(file[l:]) for file in os.listdir(samples_dir) if prefix in file and file[l:].isnumeric()]
    for sample_id in sorted(sample_ids):
        if sample_id < minSample:
            continue
        if sample_id > maxSample:
            continue
        if samples is None or sample_id in samples:
            data_file = osp.join(samples_dir, f'{prefix}{sample_id}')
            data_file_arr.append(data_file)

    return data_file_arr

class Names(Dataset):
    # TODO make this directly iterable
    "Dataset that only returns names of paths"
    def __init__(self, dirname, min_sample = 0):
        super(Names, self).__init__()
        self.paths = sorted(make_dataset(dirname, minSample = min_sample))

    def __getitem__(self, index):
        return self.paths[index]

    def __len__(self):
        return len(self.paths)

class SequencesContacts(Dataset):
    def __init__(self, dirname, toxx, toxx_mode, y_preprocessing, y_norm, x_reshape, ydtype,
                y_reshape, crop, min_subtraction, names = False, minmax = False, min_sample = 0):
        super(SequencesContacts, self).__init__()
        self.toxx = toxx
        self.toxx_mode = toxx_mode
        self.y_norm = y_norm
        self.min_subtraction = min_subtraction
        if self.y_norm == 'batch':
            assert y_preprocessing is not None, "use instance normalization instead"
            min_max = np.load(osp.join(dirname, "y_{}_min_max.npy".format(y_preprocessing)))
            print("min, max: ", min_max)
            self.ymin = min_max[0]
            self.ymax = min_max[1]
        else:
            self.ymin = 0
            self.ymax = 1
        self.y_preprocessing = y_preprocessing
        self.x_reshape = x_reshape
        self.ydtype = ydtype
        self.y_reshape = y_reshape
        self.crop = crop
        self.names = names
        self.append_minmax = minmax
        self.paths = sorted(make_dataset(dirname, minSample = min_sample))

    def __getitem__(self, index):
        if self.y_preprocessing is None:
            y_path = osp.join(self.paths[index], 'y.npy')
        elif self.y_preprocessing == 'diag':
            y_path = osp.join(self.paths[index], 'y_diag.npy')
        elif self.y_preprocessing == 'prcnt':
            y_path = osp.join(self.paths[index], 'y_prcnt.npy')
        elif self.y_preprocessing == 'diag_instance':
            y_path = osp.join(self.paths[index], 'y_diag_instance.npy')
        else:
            raise Exception("Warning: Unknown preprocessing: {}".format(self.y_preprocessing))
        y = np.load(y_path)

        if self.crop is not None:
            y = y[self.crop[0]:self.crop[1], self.crop[0]:self.crop[1]]

        if self.y_reshape:
            y = np.expand_dims(y, 0)

        if self.y_norm == 'instance':
            self.ymax = np.max(y)
            self.ymin = np.min(y)

        # if y_norm is batch this uses batch parameters from init, if y_norm is None, this does nothing
        if self.min_subtraction:
            y = (y - self.ymin) / (self.ymax - self.ymin)
        else:
            y = y / self.ymax

        if self.toxx:
            if self.toxx_mode == 'concat':
                # not implemented yet
                # TODO
                return
            else:
                x_path = osp.join(self.paths[index], 'xx.npy')
                x = np.load(x_path)
                if self.toxx_mode == 'add':
                    x = x * 2 # default is mean, so undo it
            if self.crop is not None:
                x = x[:, self.crop[0]:self.crop[1], self.crop[0]:self.crop[1]]
        else:
            x_path = osp.join(self.paths[index], 'x.npy')
            x = np.load(x_path)
            if self.crop is not None:
                x = x[self.crop[0]:self.crop[1], :]
            if self.x_reshape:
                # treat x as 1d image with k channels
                x = x.T
        x = torch.tensor(x, dtype = torch.float32)
        y = torch.tensor(y, dtype = self.ydtype)
        result = [x, y]
        if self.names:
            result.append(self.paths[index])
        if self.append_minmax:
            result.append([self.ymin, self.ymax])

        return result

    def __len__(self):
        return len(self.paths)

class Sequences(Dataset):
    def __init__(self, dirname, crop, x_reshape, names = False, min_sample = 0):
        super(Sequences, self).__init__()
        self.crop = crop
        self.x_reshape = x_reshape
        self.names = names
        self.paths = sorted(make_dataset(dirname, minSample = min_sample))

    def __getitem__(self, index):
        x_path = osp.join(self.paths[index], 'x.npy')
        x = np.load(x_path)
        if self.crop is not None:
            x = x[self.crop[0]:self.crop[1], :]
        if self.x_reshape:
            # treat x as 1d image with k channels
            x = x.T

        x = torch.tensor(x, dtype = torch.float32)
        result = [x]
        if self.names:
            result.append(self.paths[index])

        return result

    def __len__(self):
        return len(self.paths)

class DiagFunctions(Dataset):
    '''
    Dataset that contains mean contact probability along each diagonal (x)
    and corresponding diagonal chi functions (y).
    '''
    def __init__(self, dirname, crop, norm, log, zero_diag_count,
                    output_mode,
                    min_sample = 0, max_sample = float('inf'), names = False,
                    samples = None):
        '''
        Inputs:
            dirname: dataset directory
            crop (tuples): pair of values to crop meanDist to
            norm (str): type of normalization (only instance currently accepted)
            log (str): typue of log preprocessing (None to skip)
            zero_diag_count (int): number of diagonals to zero
            output_mode (str): type of data to output
        '''
        super(DiagFunctions, self).__init__()
        self.crop = crop
        assert norm is None or norm in {'max', 'mean'}, f'Unaccepted norm: {norm}'
        self.norm = norm
        self.log = log
        self.output_mode = output_mode
        self.zero_diag_count = zero_diag_count
        self.names = names
        self.paths = sorted(make_dataset(dirname, minSample = min_sample,
                                        maxSample = max_sample, samples = samples))

    def __getitem__(self, index):
        if len(self.paths[index]) == 0:
            return

        # get meanDist
        meanDist_file = osp.join(self.paths[index], 'meanDist.npy')
        if osp.exists(meanDist_file):
            meanDist = np.load(meanDist_file)
        else:
            y = np.load(osp.join(self.paths[index], 'y.npy')).astype(np.float64)
            if self.crop is not None:
                y = y[self.crop[0]:self.crop[1], self.crop[0]:self.crop[1]]
            if self.zero_diag_count > 0:
                y = np.triu(y, self.zero_diag_count)
            if self.norm == 'max':
                y /= np.max(y)
            elif self.norm == 'mean':
                y /= np.mean(np.diagonal(y))

            meanDist = DiagonalPreprocessing.genomic_distance_statistics(y)

            if self.log is not None:
                if self.log == 'ln':
                    meanDist = np.log(meanDist+1e-8)
                elif self.log.isdigit():
                    val = int(self.log)
                    if val == 2:
                        meanDist = np.log2(meanDist+1e-8)
                    elif val == 10:
                        meanDist = np.log10(meanDist+1e-8)
                    else:
                        raise Exception(f'Unaccepted log base: {val}')
                else:
                    raise Exception(f'Unrecognized log transform: {self.log}')

            if self.output_mode is not None:
                # only save if not in predict_mode
                np.save(meanDist_file, meanDist)
            else:
                # smoothing filter
                meanDist[50:] = uniform_filter(meanDist[50:], 3, mode = 'constant')

        meanDist = torch.tensor(meanDist, dtype = torch.float32)
        # print('meanDist', meanDist, meanDist.shape)
        result = [meanDist]

        # get Y
        if self.output_mode is not None:
            # load config
            config_file = osp.join(self.paths[index], 'config.json')
            if osp.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
            else:
                raise Exception(f'config_file does not exist for {self.paths[index]}')

            # get diag chi
            if self.output_mode.startswith('diag_chi_step'):
                chi_diag = get_diag_chi_step(config)
            elif self.output_mode.startswith('diag_chi_continuous'):
                chi_diag = np.load(osp.join(self.paths[index], 'diag_chis_continuous.npy'))
            elif self.output_mode.startswith('diag_chi'):
                chi_diag = np.array(config["diag_chis"])
            elif self.output_mode.startswith('diag_param'):
                param_file = osp.join(self.paths[index], 'params.npy')
                if osp.exists(param_file):
                    chi_diag = np.load(param_file)
                else:
                    params_log_file = osp.join(self.paths[index], 'params.log')
                    AC = ArgparserConverter()
                    with open(params_log_file, 'r') as f:
                        line = f.readline()
                        while line != '':
                            line = f.readline()
                            if line.startswith('Diag chi args:'):
                                line = f.readline().split(', ')
                                for arg in line:
                                    if arg.startswith('diag_chi_method'):
                                        method = arg.split('=')[1].strip("'")
                                    elif arg.startswith('diag_chi_slope'):
                                        slope = float(arg.split('=')[1])/1000
                                    elif arg.startswith('diag_chi_scale'):
                                        scale = AC.str2float(arg.split('=')[1])
                                    elif arg.startswith('diag_chi_constant'):
                                        constant = float(arg.split('=')[1])
                                    elif arg.startswith('diag_chi_midpoint'):
                                        midpoint = float(arg.split('=')[1])
                                    elif arg.startswith('min_diag_chi'):
                                        min_val = float(arg.split('=')[1])
                                    elif arg.startswith('max_diag_chi'):
                                        max_val = float(arg.split('=')[1])
                    if method == 'logistic':
                        chi_diag = np.array([min_val, max_val, slope, midpoint])
                    elif method == 'log':
                        chi_diag = np.array([scale, slope, constant])
                    np.save(param_file, chi_diag)

            # get bond_length
            if 'bond_length' in self.output_mode:
                bond_length = config['bond_length']
                chi_diag = np.append(chi_diag, bond_length)


            Y = torch.tensor(chi_diag, dtype = torch.float32)
            result.append(Y)

        if self.names:
            result.append(self.paths[index])

        return result

    def __len__(self):
        return len(self.paths)
