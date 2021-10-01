import os
import os.path as osp
from shutil import rmtree

import torch
from torch.utils.data import Dataset
import torch_geometric.data
import torch_geometric.transforms
from torch_scatter import scatter_min, scatter_max, scatter_mean, scatter_std
import torch_geometric.utils

import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm
import seaborn as sns

import time
import numpy as np
import scipy.stats as ss

def make_dataset(dir, minSample = 0):
    data_file_arr = []
    samples_dir = osp.join(dir, 'samples')
    for file in os.listdir(samples_dir):
        if not file.startswith('sample'):
            print("Skipping {}".format(file))
        else:
            sample_id = int(file[6:])
            if sample_id < minSample:
                print("Skipping {}".format(file))
            else:
                data_file = osp.join(samples_dir, file)
                if os.listdir(data_file): # not empty
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

class Sequences2Contacts(Dataset):
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
        self.minmax = minmax
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
        if self.minmax:
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
                relabel_11_to_00 = False, output = 'contact', crop = None):
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
        self.relabel_11_to_00 = relabel_11_to_00
        self.output = output
        self.crop = crop
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
        print('graph init time: {}\n'.format(np.round(time.time() - t0, 3)))

    @property
    def raw_file_names(self):
        return make_dataset(self.dirname)

    @property
    def processed_file_names(self):
        return ['graph_{}.pt'.format(i) for i in range(self.len())]

    def process(self):
        for i, raw_folder in enumerate(self.raw_file_names):
            x = self.process_x(raw_folder)
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

    def process_x(self, raw_folder):
        '''Helper function to load the appropriate particle type matrix and apply any necessary preprocessing.'''
        x = np.load(osp.join(raw_folder, 'x.npy'))
        if self.relabel_11_to_00:
            m, k = x.shape
            ind = np.where((x == np.ones(k)).all(axis = 1))
            x[ind] = 0
        if self.crop is not None:
            x = x[self.crop[0]:self.crop[1], :]
        x = torch.tensor(x, dtype = torch.float32)
        return x

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
        # plt.hist(degpos, bins = 50, label = 'pos')
        # plt.hist(degneg, bins = 50, label = 'neg')
        plt.ylabel('count', fontsize=16)
        plt.xlabel('degree', fontsize=16)
        plt.legend()
        # plt.show()
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
    t2 = torch_geometric.transforms.Constant()
    t = torch_geometric.transforms.Compose([t2])
    t0 = time.time()
    for i in range(1):
        g = ContactsGraph('dataset_04_18_21', root_name = 'graphs0', top_k = 100, weighted_LDP = True)
        print(g[0].x[:, 2:])
        rmtree(g.root)
    print('tot time', time.time() - t0)


if __name__ == '__main__':
    main()
