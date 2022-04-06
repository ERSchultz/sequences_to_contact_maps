import os.path as osp

import numpy as np
import torch
import torch_geometric
from torch.utils.data import DataLoader

from .dataset_classes import Sequences, SequencesContacts
from .networks import get_model
from .pyg_dataset_classes import ContactsGraph


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
                                opt.use_node_features, opt.use_edge_weights, opt.use_edge_attr,
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
