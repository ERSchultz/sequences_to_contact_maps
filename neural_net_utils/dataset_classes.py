import os
import os.path as osp
from shutil import rmtree

import torch
from torch.utils.data import Dataset
import torch_geometric.data
import torch_geometric.transforms
from torch_scatter import scatter_min, scatter_max, scatter_mean, scatter_std
from torch_geometric.utils import degree

import time
import numpy as np

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
    def __init__(self, dirname, root_name = None, n = 1024, y_preprocessing = 'diag',
                y_norm = 'instance', min_subtraction = True, use_node_features = True,
                sparsify_threshold = None, top_k = None, weighted_LDP = False,
                transform = None, pre_transform = None):
        t0 = time.time()
        self.n = n
        self.dirname = dirname
        self.y_preprocessing = y_preprocessing
        self.y_norm = y_norm
        self.min_subtraction = min_subtraction
        self.use_node_features = use_node_features
        self.sparsify_threshold = sparsify_threshold
        self.weighted_LDP = weighted_LDP
        self.top_k = top_k

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
            self.root = osp.join(dirname, root_name)
        super(ContactsGraph, self).__init__(self.root, transform, pre_transform)
        print('graph init time: {}'.format(np.round(time.time() - t0, 3)))

    @property
    def raw_file_names(self):
        return make_dataset(self.dirname)

    @property
    def processed_file_names(self):
        return ['graph_{}.pt'.format(i) for i in range(self.len())]

    def process(self):
        for i, raw_folder in enumerate(self.raw_file_names):
            if self.y_preprocessing is None:
                y_path = osp.join(raw_folder, 'y.npy')
            elif self.y_preprocessing == 'diag':
                y_path = osp.join(raw_folder, 'y_diag.npy')
            elif self.y_preprocessing == 'prcnt':
                y_path = osp.join(raw_folder, 'y_prcnt.npy')
            elif self.y_preprocessing == 'diag_instance':
                y_path = osp.join(raw_folder, 'y_diag_instance.npy')
            else:
                raise Exception("Warning: Unknown preprocessing: {}".format(self.y_preprocessing))
            y = np.load(y_path)
            if self.y_norm == 'instance':
                self.ymax = np.max(y)
                self.ymin = np.min(y)

            # if y_norm is batch this uses batch parameters from init, if y_norm is None, this does nothing
            if self.min_subtraction:
                y = (y - self.ymin) / (self.ymax - self.ymin)
            else:
                y = y / self.ymax

            if self.sparsify_threshold is not None:
                y[y < self.sparsify_threshold] = 0

            if self.top_k is not None:
                self.filter_to_topk(y)

            y = torch.tensor(y, dtype = torch.float32)
            edge_index, edge_weight = self.sparsify_adj_mat(y)
            x = torch.tensor(np.load(osp.join(raw_folder, 'x.npy')), dtype = torch.float32)
            if self.use_node_features:
                graph = torch_geometric.data.Data(x = x, edge_index = edge_index, edge_attr = edge_weight, y = x)
            else:
                graph = torch_geometric.data.Data(x = None, edge_index = edge_index, edge_attr = edge_weight, y = x)
            graph.minmax = torch.tensor([self.ymin, self.ymax])
            graph.path = raw_folder
            graph.num_nodes = self.n
            if self.weighted_LDP:
                if not self.top_k and not self.sparsify_threshold:
                    print('Warning: using LDP without any sparsification')
                graph = self.weightedLocalDegreeProfile(graph, y)
            torch.save(graph, self.processed_paths[i])

    def get(self, index):
         data = torch.load(self.processed_paths[index])
         return data

    def len(self):
        return len(self.raw_file_names)

    def weighted_degree(self, y):
        return torch.sum(y, axis = 1)

    def filter_to_topk(self, y):
        # any entry not in the topk will be set to 0, row-wise
        k = self.n - self.top_k
        z = np.argpartition(y, k)
        z = z[:, :k]
        y[np.arange(self.n)[:,None], z] = 0

    def sparsify_adj_mat(self, y):
        edge_index = (y > 0).nonzero().t()
        row, col = edge_index
        edge_weight = y[row, col]
        return edge_index, edge_weight

    def weightedLocalDegreeProfile(self, data, y):
        '''
        Weighted version of Local Degree Profile (LDP) from https://arxiv.org/abs/1811.03508

        Reference code: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/local_degree_profile.html#LocalDegreeProfile
        '''
        row, col = data.edge_index
        N = data.num_nodes

        deg = self.weighted_degree(y)
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

class Sequences(Dataset):
    def __init__(self, dirname, crop, names = False, min_sample = 0):
        super(Sequences, self).__init__()
        self.crop = crop
        self.names = names
        self.paths = sorted(make_dataset(dirname, minSample = min_sample))

    def __getitem__(self, index):
        x_path = osp.join(self.paths[index], 'x.npy')
        x = np.load(x_path)
        if self.crop is not None:
            x = x[self.crop[0]:self.crop[1], :]

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
