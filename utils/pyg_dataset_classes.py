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

from .dataset_classes import make_dataset


class ContactsGraph(torch_geometric.data.Dataset):
    # How to backprop through model after converting to GNN:
    # https://github.com/rusty1s/pytorch_geometric/issues/1511
    def __init__(self, dirname, root_name = None, m = 1024, y_preprocessing = 'diag',
                y_log_transform = None, y_norm = 'instance', min_subtraction = True,
                use_node_features = True,
                sparsify_threshold = None, sparsify_threshold_upper = None,
                split_neg_pos_edges = False,
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
            y_norm: type of normalization ('instance', 'batch')
            min_subtraction: True to subtract min during normalization
            use_node_features: True to use bead labels as node features
            sparsify_threshold: lower threshold for sparsifying contact map (None to skip)
            sparsify_threshold_upper: upper threshold for sparsifying contact map (None to skip)
            split_neg_pos_edges: True to split negative and positive edges for training
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
        self.sparsify_threshold = sparsify_threshold
        self.sparsify_threshold_upper = sparsify_threshold_upper
        self.split_neg_pos = split_neg_pos_edges
        self.output = output
        self.crop = crop
        self.samples = None
        self.degree_list = [] # created in self.process()
        self.verbose = verbose
        self.file_paths = make_dataset(self.dirname, maxSample = max_sample, samples = samples)

        if self.y_norm == 'batch':
            assert y_preprocessing is not None, "use instance normalization instead"
            min_max = np.load(osp.join(dirname, f"y_{y_preprocessing}_min_max.npy"))
            print("min, max: ", min_max)
            self.ymin = min_max[0]
            self.ymax = min_max[1]
        else:
            self.ymin = 0
            self.ymax = 1

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

        if verbose and self.degree_list:
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
            edge_index, pos_edge_index, neg_edge_index = self.generate_edge_index()

            if self.use_node_features:
                graph = torch_geometric.data.Data(x = x, edge_index = edge_index)
            else:
                graph = torch_geometric.data.Data(x = None, edge_index = edge_index)

            graph.minmax = torch.tensor([self.ymin, self.ymax])
            graph.path = raw_folder
            graph.num_nodes = self.m
            graph.pos_edge_index = pos_edge_index
            graph.neg_edge_index = neg_edge_index
            graph.weighted_degree = self.weighted_degree
            # graph.weighted_degree needed for pre_transform -  delete later to save RAM
            graph.contact_map = self.contact_map

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

                if self.output == 'energy_sym':
                    graph.energy = (graph.energy + graph.energy.t()) / 2
            elif self.output == 'diag_chi':
                chi_diag = None
                config_file = osp.join(raw_folder, 'config.json')
                if osp.exists(config_file):
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                        if "diag_chis" in config:
                            chi_diag = np.array(config["diag_chis"])
                if chi_diag is None:
                    raise Exception(f'chi_diag not found for {config_file}')
                D = np.zeros((self.m, self.m))
                k = len(chi_diag)
                for d in range(self.m):
                    rng = np.arange(self.m-d)
                    diag_chi = chi_diag[math.floor(d/(self.m/k))]
                    D[rng, rng+d] = diag_chi
                graph.y = torch.tensor(D, dtype = torch.float32)

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
            y_path = osp.join(raw_folder, 'y.npy')
        elif self.y_preprocessing == 'diag':
            y_path = osp.join(raw_folder, 'y_diag.npy')
        elif self.y_preprocessing == '1000_diag':
            y_path = osp.join(raw_folder, 'y1000_diag.npy')
        elif self.y_preprocessing == '2500_diag':
            y_path = osp.join(raw_folder, 'y2500_diag.npy')
        elif self.y_preprocessing == '5000_diag':
            y_path = osp.join(raw_folder, 'y5000_diag.npy')
        elif self.y_preprocessing == 'prcnt':
            y_path = osp.join(raw_folder, 'y_prcnt.npy')
        elif self.y_preprocessing == 'diag_batch':
            y_path = osp.join(raw_folder, 'y_diag_batch.npy')
        else:
            raise Exception("Unknown preprocessing: {}".format(self.y_preprocessing))

        y = np.load(y_path)
        if self.crop is not None:
            y = y[self.crop[0]:self.crop[1], self.crop[0]:self.crop[1]]

        if self.y_norm == 'instance':
            self.ymax = np.max(y)
            self.ymin = np.min(y)

        # if y_norm is batch this uses batch parameters from init,
        # if y_norm is None, this does nothing
        if self.min_subtraction:
            y = (y - self.ymin) / (self.ymax - self.ymin)
        else:
            y = y / self.ymax

        if self.y_log_transform is not None:
            if self.y_log_transform == 'ln':
                y = np.log(y+1e-8)
            elif self.y_log_transform.isdigit():
                val = int(self.y_log_transform)
                if val == 2:
                    y = np.log2(y+1e-8)
                elif val == 10:
                    y = np.log10(y+1e-8)
                else:
                    raise Exception(f'Unaccepted log base: {val}')
            else:
                raise Exception(f'Unrecognized log transform: {self.y_log_transform}')

        if self.sparsify_threshold is not None:
            y[np.abs(y) < self.sparsify_threshold] = 0
        if self.sparsify_threshold_upper is not None:
            y[np.abs(y) > self.sparsify_threshold_upper] = 0


        # self.plotDegreeProfile(y)
        y = torch.tensor(y, dtype = torch.float32)
        return y

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
