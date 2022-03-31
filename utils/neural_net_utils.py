import os.path as osp

import numpy as np
import torch
import torch_geometric
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree
from torch_scatter import scatter_max, scatter_mean, scatter_min, scatter_std

from .dataset_classes import Sequences, SequencesContacts
from .gnn_dataset_classes import ContactsGraph
from .networks import get_model


## model functions ##
def load_saved_model(opt, verbose = True):
    model = get_model(opt, verbose)
    model.to(opt.device)
    model_name = osp.join(opt.ofile_folder, 'model.pt')
    if osp.exists(model_name):
        save_dict = torch.load(model_name, map_location=torch.device('cpu'))
        model.load_state_dict(save_dict['model_state_dict'])
        train_loss_arr = save_dict['train_loss']
        val_loss_arr = save_dict['val_loss']
        if verbose:
            print('Model is loaded: {}'.format(model_name), file = opt.log_file)
    else:
        raise Exception('Model does not exist: {}'.format(model_name))
    model.eval()

    return model, train_loss_arr, val_loss_arr

## dataset functions ##
def get_dataset(opt, names = False, minmax = False, verbose = True):
    if opt.GNN_mode:
        if opt.split_sizes is not None and -1 not in opt.split_sizes:
            max_sample = np.sum(opt.split_sizes)
        else:
            max_sample = float('inf')

        dataset = ContactsGraph(opt.data_folder, opt.root_name, opt.m, opt.y_preprocessing,
                                opt.y_log_transform, opt.y_norm, opt.min_subtraction,
                                opt.use_node_features, opt.use_edge_weights,
                                opt.sparsify_threshold, opt.sparsify_threshold_upper,
                                opt.top_k, opt.split_neg_pos_edges,
                                opt.transforms_processed, opt.pre_transforms_processed,
                                opt.output_mode, opt.crop, opt.log_file, verbose,
                                max_sample)
        opt.root = dataset.root
        print('\n'*3)
    elif opt.autoencoder_mode and opt.output_mode == 'sequence':
        dataset = Sequences(opt.data_folder, opt.crop, opt.x_reshape, names)
        opt.root = None
    else:
        dataset = SequencesContacts(opt.data_folder, opt.toxx, opt.toxx_mode,
                                    opt.y_preprocessing, opt.y_norm,
                                    opt.x_reshape, opt.ydtype, opt.y_reshape,
                                    opt.crop, opt.min_subtraction, names, minmax)
        opt.root = None

    return dataset

def get_data_loaders(dataset, opt):
    train_dataset, val_dataset, test_dataset = split_dataset(dataset, opt)
    if opt.verbose:
        print('dataset lengths: ', len(train_dataset), len(val_dataset), len(test_dataset))

    if opt.GNN_mode:
        dataloader_fn = torch_geometric.loader.DataLoader
    else:
        dataloader_fn = DataLoader
    train_dataloader = dataloader_fn(train_dataset, batch_size = opt.batch_size,
                                    shuffle = opt.shuffle, num_workers = opt.num_workers)
    if len(val_dataset) > 0:
        val_dataloader = dataloader_fn(val_dataset, batch_size = opt.batch_size,
                                        shuffle = opt.shuffle, num_workers = opt.num_workers)
    else:
        val_dataloader = None

    if len(test_dataset) > 0:
        test_dataloader = dataloader_fn(test_dataset, batch_size = opt.batch_size,
                                        shuffle = opt.shuffle, num_workers = opt.num_workers)
    else:
        test_dataloader = None

    return train_dataloader, val_dataloader, test_dataloader

def split_dataset(dataset, opt):
    """Splits input dataset into proportions specified by split."""
    opt.N = len(dataset)
    if opt.split_percents is not None:
        assert sum(opt.split_percents) - 1 < 1e-5, f"split doesn't sum to 1: {opt.split_percents}"
        opt.testN = math.floor(opt.N * opt.split_percents[2])
        opt.valN = math.floor(opt.N * opt.split_percents[1])
        opt.trainN = opt.N - opt.testN - opt.valN
    else:
        assert opt.split_sizes is not None
        assert opt.split_sizes.count(-1) < 2, "can be at most 1 entry set to -1"

        opt.trainN, opt.valN, opt.testN = opt.split_sizes
        if opt.trainN == -1:
            opt.trainN = opt.N - opt.testN - opt.valN
        elif opt.valN == -1:
            opt.valN = opt.N - opt.trainN - opt.testN
        elif opt.testN == -1:
            opt.testN = opt.N - opt.trainN - opt.valN

    if opt.verbose:
        print('split sizes:', opt.trainN, opt.valN, opt.testN, opt.N)

    if opt.random_split:
        return torch.utils.data.random_split(dataset, [opt.trainN, opt.valN, opt.testN],
                                            torch.Generator().manual_seed(opt.seed))
    else:
        test_dataset = dataset[:opt.testN]
        val_dataset = dataset[opt.testN:opt.testN+opt.valN]
        train_dataset = dataset[opt.testN+opt.valN:opt.testN+opt.valN+opt.trainN]
        return train_dataset, val_dataset, test_dataset

# pytorch helper functions
def optimizer_to(optim, device = None):
    # https://discuss.pytorch.org/t/moving-optimizer-from-cpu-to-gpu/96068/2
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            if device is not None:
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
            else:
                print(param.data.get_device())
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    if device is not None:
                        subparam.data = subparam.data.to(device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(device)
                    else:
                        print(subparam.data.get_device())

# pytorch geometric functions
class WeightedLocalDegreeProfile(BaseTransform):
    '''
    Weighted version of Local Degree Profile (LDP) from https://arxiv.org/abs/1811.03508
    Appends WLDP features to feature vector.

    Reference code: https://pytorch-geometric.readthedocs.io/en/latest/_modules/
        torch_geometric/transforms/local_degree_profile.html#LocalDegreeProfile
    '''
    def __call__(self, data):
        row, col = data.edge_index
        N = data.num_nodes

        # weighted_degree must exist
        deg = data.weighted_degree
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

class Degree(BaseTransform):
    '''
    Appends degree features to feature vector.

    Reference code: https://pytorch-geometric.readthedocs.io/en/latest/_modules/
        torch_geometric/transforms/target_indegree.html#TargetIndegree
    '''
    def __init__(self, norm = True, max_val = None, weighted = False,
                split_edges = False, split_val = 0):
        self.norm = norm
        self.max_val = max_val
        self.weighted = weighted
        self.split_edges = split_edges
        self.split_val = split_val

    def __call__(self, data):
        if self.weighted:
            deg = data.weighted_degree
            if self.split_edges:
                ypos = torch.clone(data.contact_map)
                ypos[ypos < self.split_val] = 0
                pos_deg = torch.sum(ypos, axis = 1)
                del ypos
                yneg = torch.clone(data.contact_map)
                yneg[yneg > self.split_val] = 0
                neg_deg = torch.sum(yneg, axis = 1)
                del yneg
        else:
            deg = degree(data.edge_index[0], data.num_nodes)
            if self.split_edges:
                pos_deg = degree(data.pos_edge_index[0], data.num_nodes)
                neg_deg = degree(data.neg_edge_index[0], data.num_nodes)

        if self.norm:
            deg /= (deg.max() if self.max_val is None else self.max_val)
            if self.split_edges:
                # if statement is a safety check to avoid divide by 0
                pos_deg /= (pos_deg.max() if pos_deg.max() > 0 else 1)

                neg_deg /= (neg_deg.max() if neg_deg.max() > 0 else neg_deg.min())

        if self.split_edges:
            deg = torch.stack([deg, pos_deg, neg_deg], dim=1)
        else:
            deg = torch.stack([deg], dim=1)

        if data.x is not None:
            data.x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([data.x, deg], dim=-1)
        else:
            data.x = deg

        return data

class AdjPCATransform(BaseTransform):
    '''Appends values from top k PCs of adjacency matrix to feature vector.'''
    def __init__(self, k = 5):
        self.k = k

    def __call__(self, data):
        pca = PCA(n_components = self.k)
        y_trans = pca.fit_transform(torch.clone(data.contact_map))

        y_trans = torch.tensor(y_trans, dtype = torch.float32)

        if data.x is not None:
            data.x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([data.x, y_trans], dim=-1)
        else:
            data.x = y_trans

        return data

class AdjTransform(BaseTransform):
    '''Appends rows of adjacency matrix to feature vector.'''
    def __call__(self, data):
        if data.x is not None:
            data.x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([data.x, torch.clone(data.contact_map)], dim=-1)
        else:
            data.x = torch.clone(data.contact_map)

        return data
