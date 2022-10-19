import json
import math
import os
import os.path as osp
import sys
import time
from shutil import rmtree

import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
import torch
import torch_geometric.data
import torch_geometric.transforms
import torch_geometric.utils
from torch_scatter import scatter_max, scatter_mean, scatter_min, scatter_std

from .argparse_utils import finalize_opt, get_base_parser
from .dataset_classes import DiagFunctions, make_dataset
from .energy_utils import calculate_D
from .networks import get_model


class ContactsGraph(torch_geometric.data.Dataset):
    # How to backprop through model after converting to GNN:
    # https://github.com/rusty1s/pytorch_geometric/issues/1511
    def __init__(self, dirname, root_name = None, m = 1024, y_preprocessing = 'diag',
                y_log_transform = None, y_norm = 'mean', min_subtraction = True,
                use_node_features = True, mlp_model_id = None,
                sparsify_threshold = None, sparsify_threshold_upper = None,
                split_neg_pos_edges = False, max_diagonal = None,
                transform = None, pre_transform = None, output = 'contact',
                crop = None, ofile = sys.stdout, verbose = True,
                max_sample = float('inf'), samples = None):
        '''
        Inputs:
            dirname: directory path to raw data
            root_name: directory for loaded data
            m: number of particles/beads
            y_preprocessing: type of contact map preprocessing ('diag', None, etc)
            y_log_transform: type of log transform (int k for log_k, 'ln' for ln, None to skip)
            y_norm: type of normalization ('mean', 'max')
            min_subtraction: True to subtract min during normalization
            use_node_features: True to use bead labels as node features
            mlp_model_id: id for mlp diagonal parameters (can be used as edge attr)
            sparsify_threshold: lower threshold for sparsifying contact map (None to skip)
            sparsify_threshold_upper: upper threshold for sparsifying contact map (None to skip)
            split_neg_pos_edges: True to split negative and positive edges for training
            max_diagonal: maximum diagonal of adjacency matrix to consider
            transform: list of transforms
            pre_transform: list of transforms
            output: output mode ('contact', 'energy', 'energy_sym')
            crop: tuple of crop sizes
            ofile: where to print to if verbose == True
            verbose: True to print
            max_sample: max sample id to save
            samples: set of samples to include (None for all)
        '''
        t0 = time.time()
        self.m = m
        self.dirname = dirname
        self.y_preprocessing = y_preprocessing
        self.y_log_transform = y_log_transform
        self.y_norm = y_norm
        self.min_subtraction = min_subtraction
        self.use_node_features = use_node_features
        self.mlp_model_id = mlp_model_id
        self.sparsify_threshold = sparsify_threshold
        self.sparsify_threshold_upper = sparsify_threshold_upper
        self.split_neg_pos = split_neg_pos_edges
        self.max_diagonal = max_diagonal
        self.output = output
        self.crop = crop
        self.samples = None
        self.num_edges_list = [] # list of number of edges per graph
        self.degree_list = [] # created in self.process()
        self.verbose = verbose
        self.file_paths = make_dataset(self.dirname, maxSample = max_sample, samples = samples)

        if root_name is None:
            # find any currently existing graph data folders
            # make new folder for this dataset
            max_val = -1
            for file in os.listdir(dirname):
                file_path = osp.join(dirname, file)
                if file.startswith('graphs') and osp.isdir(file_path):
                    # format is graphs<i> where i is integer
                    val = int(file[6:])
                    if val > max_val:
                        max_val = val
            self.root = osp.join(dirname, 'graphs{}'.format(max_val+1))
        else:
            # use exsting graph data folder
            self.root = osp.join(dirname, root_name)
        super(ContactsGraph, self).__init__(self.root, transform, pre_transform)

        if verbose:
            print('Dataset construction time: '
                    f'{np.round((time.time() - t0) / 60, 3)} minutes', file = ofile)
            print('Average num edges per graph: ',
                    f'{np.mean(self.num_edges_list)}', file = ofile)
            print('Average num edges per graph: ',
                    f'{np.mean(self.num_edges_list)}')

            if self.degree_list:
                # self.degree_list will be None if loading already processed dataset
                self.degree_list = np.array(self.degree_list)
                mean_deg = np.round(np.mean(self.degree_list, axis = 1), 2)
                std_deg = np.round(np.std(self.degree_list, axis = 1), 2)
                print('Mean degree: {} +- {}\n'.format(mean_deg, std_deg), file = ofile)

    @property
    def raw_file_names(self):
        return self.file_paths

    @property
    def processed_file_names(self):
        return ['graph_{}.pt'.format(i) for i in range(self.len())]

    def process(self):
        for i, raw_folder in enumerate(self.raw_file_names):
            sample = int(osp.split(raw_folder)[1][6:])
            x, psi = self.process_x_psi(raw_folder)
            self.contact_map = self.process_y(raw_folder)
            self.diag_chis_continuous, self.diag_chis_continuous_mlp = self.process_diag_params(raw_folder)
            edge_index, pos_edge_index, neg_edge_index = self.generate_edge_index()

            if self.use_node_features:
                graph = torch_geometric.data.Data(x = x, edge_index = edge_index)
            else:
                graph = torch_geometric.data.Data(x = None, edge_index = edge_index)

            graph.path = raw_folder
            graph.num_nodes = self.m
            graph.pos_edge_index = pos_edge_index
            graph.neg_edge_index = neg_edge_index
            graph.mlp_model_id = self.mlp_model_id

            # copy these temporarily
            graph.weighted_degree = self.weighted_degree
            graph.contact_map = self.contact_map
            graph.diag_chis_continuous = self.diag_chis_continuous
            graph.diag_chis_continuous_mlp = self.diag_chis_continuous_mlp

            if self.pre_transform is not None:
                graph = self.pre_transform(graph)
            del graph.weighted_degree # no longer needed

            if self.output != 'contact':
                del graph.contact_map

            if self.output is not None and self.output.startswith('energy'):
                # first look for s
                s_path1 = osp.join(raw_folder, 's.npy')
                s_path2 = osp.join(raw_folder, 's_matrix.txt')
                if osp.exists(s_path1) or osp.exists(s_path2):
                    if osp.exists(s_path1):
                        s = np.load(s_path1)
                    else:
                        s = np.loadtxt(s_path2)
                    if self.crop is not None:
                        s = s[self.crop[0]:self.crop[1], self.crop[0]:self.crop[1]]
                    graph.energy = torch.tensor(s, dtype = torch.float32)
                else:
                    # look for chi
                    chi_path1 = osp.join(raw_folder, 'chis.npy')
                    chi_path2 = osp.join(osp.split(osp.split(raw_folder)[0])[0], 'chis.npy')
                    if osp.exists(chi_path1):
                        chi = np.load(chi_path1)
                    elif osp.exists(chi_path2):
                        chi = np.load(chi_path2)
                    else:
                        raise Exception(f'chi does not exist: {chi_path1}, {chi_path2}')
                    chi = torch.tensor(chi, dtype = torch.float32)
                    graph.energy = x @ chi @ x.t()

                if self.output.startswith('energy_sym'):
                    graph.energy = (graph.energy + graph.energy.t()) / 2
                if self.output.startswith('energy_sym_diag'):
                    D = calculate_D(graph.diag_chis_continuous)
                    graph.energy += torch.tensor(D, dtype = torch.float32)

            del graph.diag_chis_continuous
            del graph.diag_chis_continuous_mlp

            torch.save(graph, self.processed_paths[i])

            # record degree
            if self.verbose:
                deg = np.array(torch_geometric.utils.degree(graph.edge_index[0],
                                                            graph.num_nodes))
                self.degree_list.append(deg)

    def process_x_psi(self, raw_folder):
        '''
        Helper function to load the appropriate particle type matrix and
        apply any necessary preprocessing.
        '''
        x_file = osp.join(raw_folder, 'x.npy')
        if osp.exists(x_file):
            x = np.load(x_file)
            if self.crop is not None:
                x = x[self.crop[0]:self.crop[1], :]

            psi_file = osp.join(raw_folder, 'psi.npy')
            if osp.exists(psi_file):
                psi = np.load(psi_file)
                if self.crop is not None:
                    psi = psi[self.crop[0]:self.crop[1], :]
            else:
                psi = x.copy()
            x = torch.tensor(x, dtype = torch.float32)
            psi = torch.tensor(psi, dtype = torch.float32)
        else:
            x = None
            psi = None
        return x, psi

    def process_y(self, raw_folder):
        '''
        Helper function to load the appropriate contact map and apply any
        necessary preprocessing.
        '''
        if self.y_preprocessing is None:
            y = np.load(osp.join(raw_folder, 'y.npy')).astype(np.float64)
            if self.y_norm == 'max':
                y /= np.max(y)
            elif self.y_norm == 'mean':
                y /= np.mean(np.diagonal(y))
        elif self.y_preprocessing == 'log':
            y = np.load(osp.join(raw_folder, 'y.npy')).astype(np.float64)
            if self.y_norm == 'max':
                y /= np.max(y)
            elif self.y_norm == 'mean':
                y /= np.mean(np.diagonal(y))
            y = np.log(y+1)
        else:
            assert self.y_norm is None
            y_path = osp.join(raw_folder, f'y_{self.y_preprocessing}.npy')
            if osp.exists(y_path):
                y = np.load(y_path)
            raise Exception(f"Unknown preprocessing: {self.y_preprocessing} or y_path missing: {y_path}")

        if self.crop is not None:
            y = y[self.crop[0]:self.crop[1], self.crop[0]:self.crop[1]]

        if self.max_diagonal is not None:
            y = np.tril(y, self.max_diagonal)
            y = np.triu(y, -self.max_diagonal)

        if self.y_log_transform is not None:
            assert not self.y_preprocessing.endswith('log'), "don't use log twice in a row"
            if self.y_log_transform == 'ln':
                y = np.log(y)
            elif self.y_log_transform.isdigit():
                val = int(self.y_log_transform)
                if val == 2:
                    y = np.log2(y)
                elif val == 10:
                    y = np.log10(y)
                else:
                    raise Exception(f'Unaccepted log base: {val}')
            else:
                raise Exception(f'Unrecognized log transform: {self.y_log_transform}')

            y[np.isinf(y)] = 0 # since we didn't add a constant to y before the log

        if self.sparsify_threshold is not None:
            y[np.abs(y) < self.sparsify_threshold] = 0
        if self.sparsify_threshold_upper is not None:
            y[np.abs(y) > self.sparsify_threshold_upper] = 0


        # self.plotDegreeProfile(y)
        y = torch.tensor(y, dtype = torch.float32)
        return y

    def process_diag_params(self, raw_folder):
        path = osp.join(raw_folder, 'diag_chis_continuous.npy')
        if osp.exists(path):
            diag_chis_gt = torch.tensor(np.load(path), dtype = torch.float32)
        else:
            diag_chis_gt = None
            if self.output is not None:
                raise Exception(f'chi_diag not found for {raw_folder}')

        if self.mlp_model_id is None:
            diag_chis_mlp = None
        else:
            # extract sample info
            sample = osp.split(raw_folder)[1]
            sample_id = int(sample[6:])
            sample_path_split = osp.normpath(raw_folder).split(os.sep)

            model_path = f'/home/erschultz/sequences_to_contact_maps/results/MLP/{self.mlp_model_id}'
            argparse_path = osp.join(model_path, 'argparse.txt')
            with open(argparse_path, 'r') as f:
                for line in f:
                    if line == '--data_folder\n':
                        break
                data_folder = f.readline().strip()
                mlp_dataset = osp.split(data_folder)[1]

            # set up argparse options
            parser = get_base_parser()
            sys.argv = [sys.argv[0]] # delete args from get_params, otherwise gnn opt will try and use them
            opt = parser.parse_args(['@{}'.format(argparse_path)])
            opt.id = int(self.mlp_model_id)
            output_mode = opt.output_mode
            opt = finalize_opt(opt, parser, local = True, debug = True)
            opt.data_folder = osp.join('/',*sample_path_split[:-2]) # use sample_dataset not mlp_dataset
            opt.log_file = sys.stdout # change
            opt.output_mode = None # None for prediction mode
            opt.crop = (0, self.m)

            # get model
            model = get_model(opt, False).to(opt.device)
            model_name = osp.join(opt.ofile_folder, 'model.pt')
            if osp.exists(model_name):
                save_dict = torch.load(model_name, map_location=torch.device('cpu'))
                model.load_state_dict(save_dict['model_state_dict'])
            else:
                raise Exception('Model does not exist: {}'.format(model_name))
            model.eval()

            # get dataset
            dataset = DiagFunctions(opt.data_folder, opt.crop, opt.preprocessing_norm,
                                    opt.log_preprocessing, opt.y_zero_diag_count,
                                    opt.output_mode,
                                    names = False, samples = [sample_id])

            # get prediction
            for i, x in enumerate(dataset):
                x = x[0]
                x = x.to(opt.device)
                yhat = model(x)
                yhat = yhat.cpu().detach().numpy()
                yhat = yhat.reshape((-1)).astype(np.float64)

            if 'bond_length' in output_mode:
                bond_length = yhat[-1]
                yhat = yhat[:-1]
                with open('bond_length.txt', 'w') as f:
                    f.write(str(bond_length))
                print('MLP bond_length:', bond_length)

            assert output_mode.startswith('diag_chi_step') or output_mode.startswith('diag_chi_continuous')
            diag_chis_mlp = yhat

        return diag_chis_gt, diag_chis_mlp

    def get(self, index):
         data = torch.load(self.processed_paths[index])
         return data

    def len(self):
        return len(self.raw_file_names)

    @property
    def weighted_degree(self):
        return torch.sum(self.contact_map, axis = 1)

    def filter_to_topk(self, y, top_k):
        # any entry whose absolute value is not in the top_k will be set to 0, row-wise
        yabs = np.abs(y)
        k = self.m - top_k
        z = np.argpartition(yabs, k, axis = -1)
        z = z[:, :k]
        y[np.arange(self.m)[:,None], z] = 0
        y = y.T # convert to col_wise filtering
        return y

    def generate_edge_index(self):
        edge_index = self.contact_map.nonzero().t()
        self.num_edges_list.append(edge_index.shape[1])
        if self.split_neg_pos:
            if self.y_log_transform:
                split_val = 0
            else:
                split_val = 1
            pos_edge_index = (self.contact_map > split_val).nonzero().t()
            neg_edge_index = (self.contact_map < split_val).nonzero().t()
        else:
            pos_edge_index = None
            neg_edge_index = None

        return edge_index, pos_edge_index, neg_edge_index

    def plotDegreeProfile(self):
        # y is weighted adjacency matrix
        ycopy = self.contact_map.copy()
        ycopy[y > 0] = 1
        ycopy[y < 0] = 1

        ypos = y.copy()
        ypos[y > 0] = 1
        ypos[y < 0] = 0

        yneg = y.copy()
        yneg[y > 0] = 0
        yneg[y < 0] = -1

        deg = np.sum(ycopy, axis = 0)
        degpos = np.sum(ypos, axis = 0)
        degneg = np.sum(yneg, axis = 0)
        print('min: ', np.min(deg), 'max: ', np.max(deg), ss.mode(deg))
        print('min: ', np.min(degpos), 'max: ', np.max(degpos), ss.mode(degpos))
        print('min: ', np.min(degneg), 'max: ', np.max(degneg), ss.mode(degneg))
        plt.hist(deg, bins = 100, label = 'deg')
        plt.ylabel('count', fontsize=16)
        plt.xlabel('degree', fontsize=16)
        plt.legend()
        plt.show()
        plt.close()


def main():
    g = ContactsGraph('dataset_04_18_21', root_name = 'graphs0', output = 'energy',
                        crop = [0, 15], m = 15)
    rmtree(g.root)


if __name__ == '__main__':
    main()
