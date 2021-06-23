import os.path as osp
import sys
abspath = osp.abspath(__file__)
dname = osp.dirname(abspath)
sys.path.insert(0, dname)

import torch
from torch.utils.data import Dataset
from utils import make_dataset
import torch_geometric.data
import numpy as np

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
    # How to backprop through model after convertign to GNN: https://github.com/rusty1s/pytorch_geometric/issues/1511
    def __init__(self, dirname, y_preprocessing, y_norm, min_subtraction, transform = None, pre_transform = None):
        self.dirname = dirname
        self.y_preprocessing = y_preprocessing
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

        print(self.processed_file_names)

        super(ContactsGraph, self).__init__(dirname, transform, pre_transform)

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
            y = torch.tensor(np.load(y_path), dtype = torch.float32)
            if self.y_norm == 'instance':
                self.ymax = torch.max(y)
                self.ymin = torch.min(y)

            # if y_norm is batch this uses batch parameters from init, if y_norm is None, this does nothing
            if self.min_subtraction:
                y = (y - self.ymin) / (self.ymax - self.ymin)
            else:
                y = y / self.ymax

            x = torch.tensor(np.load(osp.join(raw_folder, 'x.npy')), dtype = torch.float32)
            edge_index = (y > 0).nonzero().t()
            row, col = edge_index
            edge_weight = y[row, col]
            graph = torch_geometric.data.Data(x = x, edge_index = edge_index, edge_attr = edge_weight, y = x)
            graph.minmax = torch.tensor([self.ymin, self.ymax])
            graph.path = raw_folder
            torch.save(graph, osp.join(self.processed_dir, 'graph_{}.pt'.format(i)))

    def get(self, index):
         data = torch.load(self.processed_paths[index])

         return data

    def len(self):
        return len(self.raw_file_names)


def main():
    g = ContactsGraph('dataset_04_18_21', None, None, True)
    graph = g[0]
    x, edge_index, edge_attr  = graph.x, graph.edge_index, graph.edge_attr


if __name__ == '__main__':
    main()
