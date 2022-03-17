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
import seaborn as sns
import torch
import torch_geometric.data
import torch_geometric.transforms
import torch_geometric.utils
from torch.utils.data import Dataset
from torch_scatter import scatter_max, scatter_mean, scatter_min, scatter_std


def make_dataset(dir, minSample = 0, maxSample = float('inf'), verbose = False, samples = None):
    """
    Make list data file paths.

    Inputs:
        dir: data source directory
        minSample: ignore samples < minSample
        maxSample: ignore samples > maxSample
        verbose: True for verbose mode
        samples: list/set of samples to include, None for all samples

    Outputs:
        data_file_arr: list of data file paths
    """
    data_file_arr = []
    samples_dir = osp.join(dir, 'samples')
    for file in os.listdir(samples_dir):
        if not file.startswith('sample'):
            if verbose:
                print("Skipping {}".format(file))
        else:
            sample_id = int(file[6:])
            if sample_id < minSample:
                continue
            if sample_id > maxSample:
                continue
            if samples is None or sample_id in samples:
                data_file = osp.join(samples_dir, file)
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
        super(Sequences2Contacts, self).__init__()
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

class ContactsGraph(torch_geometric.data.Dataset):
    # How to backprop through model after converting to GNN: https://github.com/rusty1s/pytorch_geometric/issues/1511
    def __init__(self, dirname, root_name = None, m = 1024, y_preprocessing = 'diag', y_log_transform = False,
                y_norm = 'instance', min_subtraction = True, use_node_features = True, use_edge_weights = True,
                sparsify_threshold = None, sparsify_threshold_upper = None, top_k = None,
                weighted_LDP = False, split_neg_pos_edges = False, degree = False, weighted_degree = False,
                split_neg_pos_edges_for_feature_augmentation = False,
                transform = None, pre_transform = None,
                output = 'contact', crop = None, samples = None,
                ofile = sys.stdout, verbose = True):
        t0 = time.time()
        self.m = m
        self.dirname = dirname
        self.y_preprocessing = y_preprocessing
        self.y_log_transform = y_log_transform
        self.y_norm = y_norm
        self.min_subtraction = min_subtraction
        self.use_node_features = use_node_features
        self.use_edge_weights = use_edge_weights
        self.sparsify_threshold = sparsify_threshold
        self.sparsify_threshold_upper = sparsify_threshold_upper
        self.top_k = top_k
        self.weighted_LDP = weighted_LDP
        self.split_neg_pos = split_neg_pos_edges
        self.degree = degree
        self.weighted_degree = weighted_degree
        self.split_neg_pos_edges_for_feature_augmentation = split_neg_pos_edges_for_feature_augmentation
        self.output = output
        self.crop = crop
        self.samples = None
        self.degree_list = [] # created in self.process()
        self.verbose = verbose
        if self.weighted_LDP and self.top_k is None and self.sparsify_threshold is None and self.sparsify_threshold_upper is None:
            print('Warning: using LDP without any sparsification')

        if self.y_norm == 'batch':
            assert y_preprocessing is not None, "use instance normalization instead"
            min_max = np.load(osp.join(dirname, "y_{}_min_max.npy".format(y_preprocessing)))
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
                    # format is graphs### where ### is number
                    val = int(file[6:])
                    if val > max_val:
                        max_val = val
            self.root = osp.join(dirname, 'graphs{}'.format(max_val+1))
        else:
            # use exsting graph data folder
            self.root = osp.join(dirname, root_name)
        super(ContactsGraph, self).__init__(self.root, transform, pre_transform)

        if verbose:
            print('Dataset construction time: {} minutes'.format(np.round((time.time() - t0) / 60, 3)), file = ofile)

        if verbose:
            self.degree_list = np.array(self.degree_list)
            mean_deg = np.round(np.mean(self.degree_list, axis = 1), 2)
            std_deg = np.round(np.std(self.degree_list, axis = 1), 2)
            print('Mean degree: {} +- {}\n'.format(mean_deg, std_deg), file = ofile)

    @property
    def raw_file_names(self):
        return make_dataset(self.dirname, samples = self.samples)

    @property
    def processed_file_names(self):
        return ['graph_{}.pt'.format(i) for i in range(self.len())]

    def process(self):
        for i, raw_folder in enumerate(self.raw_file_names):
            sample = int(osp.split(raw_folder)[1][6:])
            x, psi = self.process_x_psi(raw_folder)
            y = self.process_y(raw_folder)
            edge_index, pos_edge_index, neg_edge_index, edge_weight = self.sparsify_adj_mat(y)

            if self.use_node_features:
                graph = torch_geometric.data.Data(x = x, edge_index = edge_index, edge_attr = edge_weight, y = x)
            else:
                graph = torch_geometric.data.Data(x = None, edge_index = edge_index, edge_attr = edge_weight, y = x)

            graph.minmax = torch.tensor([self.ymin, self.ymax])
            graph.path = raw_folder
            graph.num_nodes = self.m
            graph.pos_edge_index = pos_edge_index
            graph.neg_edge_index = neg_edge_index

            if self.weighted_LDP:
                graph = self.weightedLocalDegreeProfile(graph, y)
            if self.degree or self.weighted_degree:
                self.concatDegree(graph, y, self.weighted_degree)
            if self.pre_transform is not None:
                graph = self.pre_transform(graph)

            if self.output == 'contact':
                graph.contact_map = y
                # TODO double check that this works in core_test_train
                # y = torch_geometric.utils.to_dense_adj(data.edge_index, edge_attr = data.edge_attr,
                                                        # batch = data.batch,
                                                        # max_num_nodes = opt.n)
            elif self.output == 'energy':
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
                        raise Exception('chi does not exist: {}, {}'.format(chi_path1, chi_path2))
                    chi = torch.tensor(chi, dtype = torch.float32)
                    graph.energy = x @ chi @ x.t()

            torch.save(graph, self.processed_paths[i])

            # record degree
            if self.verbose:
                deg = np.array(torch_geometric.utils.degree(graph.edge_index[0], num_nodes = self.m))
                self.degree_list.append(deg)

    def process_x_psi(self, raw_folder):
        '''Helper function to load the appropriate particle type matrix and apply any necessary preprocessing.'''
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
        '''Helper function to load the appropriate contact map and apply any necessary preprocessing.'''
        if self.y_preprocessing is None:
            y_path = osp.join(raw_folder, 'y.npy')
        elif self.y_preprocessing == 'diag':
            y_path = osp.join(raw_folder, 'y_diag.npy')
        elif self.y_preprocessing == 'prcnt':
            y_path = osp.join(raw_folder, 'y_prcnt.npy')
        elif self.y_preprocessing == 'diag_instance':
            y_path = osp.join(raw_folder, 'y_diag_instance.npy')
        else:
            raise Exception("Unknown preprocessing: {}".format(self.y_preprocessing))

        y = np.load(y_path)
        if self.crop is not None:
            y = y[self.crop[0]:self.crop[1], self.crop[0]:self.crop[1]]

        if self.y_log_transform:
            y = np.log10(y)

        if self.y_norm == 'instance':
            self.ymax = np.max(y)
            self.ymin = np.min(y)

        # if y_norm is batch this uses batch parameters from init, if y_norm is None, this does nothing
        if self.min_subtraction:
            y = (y - self.ymin) / (self.ymax - self.ymin)
        else:
            y = y / self.ymax

        if self.sparsify_threshold is not None:
            y[np.abs(y) < self.sparsify_threshold] = 0
        if self.sparsify_threshold_upper is not None:
            y[np.abs(y) > self.sparsify_threshold_upper] = 0

        if self.top_k is not None:
            y = self.filter_to_topk(y)

        # self.plotDegreeProfile(y)
        y = torch.tensor(y, dtype = torch.float32)
        return y

    def get(self, index):
         data = torch.load(self.processed_paths[index])
         return data

    def len(self):
        return len(self.raw_file_names)

    def get_weighted_degree(self, y):
        return torch.sum(y, axis = 1)

    def filter_to_topk(self, y):
        # any entry whose absolute value is not in the topk will be set to 0, row-wise
        yabs = np.abs(y)
        k = self.m - self.top_k
        z = np.argpartition(yabs, k, axis = -1)
        z = z[:, :k]
        y[np.arange(self.m)[:,None], z] = 0
        y = y.T # convert to col_wise filtering
        return y

    def sparsify_adj_mat(self, y):
        edge_index = y.nonzero().t()
        if self.split_neg_pos:
            assert not self.use_edge_weights, "not supported"
            pos_edge_index = (y > 0).nonzero().t()
            neg_edge_index = (y < 0).nonzero().t()
        else:
            pos_edge_index = None
            neg_edge_index = None

        if self.use_edge_weights:
            row, col = edge_index
            edge_weight = y[row, col]
        else:
            edge_weight = None

        return edge_index, pos_edge_index, neg_edge_index, edge_weight

    def weightedLocalDegreeProfile(self, data, y):
        '''
        Weighted version of Local Degree Profile (LDP) from https://arxiv.org/abs/1811.03508

        Reference code: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/local_degree_profile.html#LocalDegreeProfile

        Appends LDP features to feature vector.
        '''
        row, col = data.edge_index
        N = data.num_nodes

        deg = self.get_weighted_degree(y)
        deg_col = deg[col]

        min_deg, _ = scatter_min(deg_col, row, dim_size=N)
        min_deg[min_deg > 10000] = 0
        max_deg, _ = scatter_max(deg_col, row, dim_size=N)
        max_deg[max_deg < -10000] = 0
        mean_deg = scatter_mean(deg_col, row, dim_size=N)
        std_deg = scatter_std(deg_col, row, dim_size=N)

        x = torch.stack([deg, min_deg, max_deg, mean_deg, std_deg], dim=1)

        if data.x is not None:
            data.x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([data.x, x], dim=-1)
        else:
            data.x = x

        return data

    def concatDegree(self, data, y, weighted):
        if weighted:
            deg = self.get_weighted_degree(y)
            if self.split_neg_pos_edges_for_feature_augmentation:
                ypos = torch.clone(y)
                ypos[y < 0] = 0
                pos_deg = self.get_weighted_degree(ypos)
                del ypos
                yneg = torch.clone(y)
                yneg[y > 0] = 0
                neg_deg = self.get_weighted_degree(yneg)
                del yneg
        else:
            deg = torch_geometric.utils.degree(data.edge_index[0], num_nodes = self.m)
            if self.split_neg_pos_edges_for_feature_augmentation:
                pos_deg = torch_geometric.utils.degree(data.pos_edge_index[0], num_nodes = self.m)
                neg_deg = torch_geometric.utils.degree(data.neg_edge_index[0], num_nodes = self.m)

        deg = deg / torch.max(deg)
        if self.split_neg_pos_edges_for_feature_augmentation:
            if torch.max(pos_deg) > 0:
                # this condition failed during testing with small subgraphs
                # shouldn't be a concern otherwise
                pos_deg = pos_deg / torch.max(pos_deg)
            if torch.max(neg_deg) > 0:
                neg_deg = neg_deg / torch.max(neg_deg)
            deg = torch.stack([deg, pos_deg, neg_deg], dim=1)
        else:
            deg = torch.stack([deg], dim=1)

        if data.x is not None:
            data.x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([graph.x, x], dim=-1)
        else:
            data.x = deg

        return data

    def plotDegreeProfile(self, y):
        # y is weighted adjacency matrix
        ycopy = y.copy()
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

def main():
    g = ContactsGraph('dataset_04_18_21', root_name = 'graphs0', output = 'energy', crop = [0, 15], m = 15)
    rmtree(g.root)


if __name__ == '__main__':
    main()
