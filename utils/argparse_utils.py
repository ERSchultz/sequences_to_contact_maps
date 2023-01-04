import argparse
import csv
import multiprocessing
import os
import os.path as osp
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms

from .pyg_fns import (AdjPCATransform, AdjPCs, AdjTransform, ContactDistance,
                      Degree, DiagonalParameterDistance, GeneticDistance,
                      GeneticPosition, NoiseLevel, OneHotGeneticPosition,
                      WeightedLocalDegreeProfile)
from .utils import DiagonalPreprocessing

sys.path.insert(0, '/home/erschultz/SignNet-BasisNet/Alchemy')
from sign_net.transform import EVDTransform


def get_base_parser():
    '''Helper function that returns base parser'''
    parser = argparse.ArgumentParser(description='Base parser', fromfile_prefix_chars='@',
                                    allow_abbrev = False)
    AC = ArgparserConverter()

    # GNN pre-processing args
    parser.add_argument('--GNN_mode', type=AC.str2bool, default=False,
                        help='True to use GNNs (uses pytorch_geometric in core_test_train)')
    parser.add_argument('--transforms', type=AC.str2list, default=[],
                        help='list of transforms to use for GNN')
    parser.add_argument('--pre_transforms', type=AC.str2list, default=[],
                        help='list of pre-transforms to use for GNN')
    parser.add_argument('--sparsify_threshold', type=AC.str2float,
                        help='remove all edges with weight < threshold (None to do nothing)')
    parser.add_argument('--sparsify_threshold_upper', type=AC.str2float,
                        help='remove all edges with weight < threshold (None to do nothing)')
    parser.add_argument('--top_k', type=AC.str2int, default=None,
                        help='filter to top k largest edges per node (None to do nothing) - DEPRECATED')
    parser.add_argument('--use_node_features', type=AC.str2bool, default=False,
                        help='True to use node features for GNN models')
    parser.add_argument('--use_edge_weights', type=AC.str2bool, default=True,
                        help='True to use edge weights in GNN')
    parser.add_argument('--use_edge_attr', type=AC.str2bool, default=False,
                        help='True to use edge attr in GNN')
    parser.add_argument('--keep_zero_edges', type=AC.str2bool, default=False,
                        help='True to keep edges with zero weight')

    # pre-processing args
    parser.add_argument('--data_folder', type=AC.str2list, default='dataset_04_18_21',
                        help='Location of data')
    parser.add_argument('--scratch', type=str, default='/scratch/midway2/erschultz',
                        help='Location of scratch dir')
    parser.add_argument('--root_name', type=AC.str2None,
                        help='name of file to save graph data (leave as None to create root automatically)'
                            '(root is the directory path - defined later)')
    parser.add_argument('--delete_root', type=AC.str2bool, default=True,
                        help='True to delete root directory after runtime')
    parser.add_argument('--toxx', type=AC.str2bool, default=False,
                        help='True if x should be converted to 2D image')
    parser.add_argument('--toxx_mode', type=str, default='mean',
                        help='mode for toxx (default mean)')
    parser.add_argument('--y_preprocessing', type=AC.str2None,
                        help='type of pre-processing for contact map')
    parser.add_argument('--y_zero_diag_count', type=int, default=0,
                        help='number of diagonals of y set to 0')
    parser.add_argument('--log_preprocessing', type=AC.str2None,
                        help='type of log transform input data (None to skip)')
    parser.add_argument('--output_preprocesing', type=AC.str2None,
                        help='type of preprocessing for output')
    parser.add_argument('--kr', type=AC.str2bool,
                        help='True to use KnightRuiz balancing algorithm')
    parser.add_argument('--mean_filt', type=AC.str2int,
                        help='mean_filt: apply mean filter of width <mean_filt> (None to skip)')
    parser.add_argument('--rescale', type=AC.str2int,
                        help='rescale contact map by factor of <rescale> (None to skip)')
    parser.add_argument('--gated', type=AC.str2bool, default=False,
                        help='True to use gated connection')
    parser.add_argument('--preprocessing_norm', type=AC.str2None, default='batch',
                        help='type of [0,1] normalization for input data')
    parser.add_argument('--min_subtraction', type=AC.str2bool, default=True,
                        help='if min subtraction should be used for preprocessing_norm')
    parser.add_argument('--x_reshape', type=AC.str2bool, default=True,
                        help='True if x should be considered a 1D image')
    parser.add_argument('--ydtype', type=AC.str2dtype, default='float32',
                        help='torch data type for y')
    parser.add_argument('--y_reshape', type=AC.str2bool, default=True,
                        help='True if y should be considered a 2D image')
    parser.add_argument('--crop', type=AC.str2list,
                        help='size of crop to apply to image - format: <leftcrop-rightcrop>')
    parser.add_argument('--classes', type=int, default=10,
                        help='number of classes in percentile normalization')
    parser.add_argument('--use_scratch', type=AC.str2bool, default=False,
                        help='True to move data to scratch')
    parser.add_argument('--use_scratch_parallel', type=AC.str2bool, default=False,
                        help='True to move data in parallel (use_scratch must be True)')

    # dataloader args
    parser.add_argument('--split_percents', type=AC.str2list,
                        help='Train, val, test split for dataset (percents)')
    parser.add_argument('--split_sizes', type=AC.str2list, default=[-1, 200, 0],
                        help='Train, val, test split for dataset (counts), -1 for remainder')
    parser.add_argument('--random_split', type=AC.str2bool, default=False,
                        help='True to use random train, val, test split')
    parser.add_argument('--shuffle', type=AC.str2bool, default=True,
                        help='Whether or not to shuffle dataset')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of threads for data loader to use')

    # train args
    parser.add_argument('--start_epoch', type=int, default=1,
                        help='Starting epoch')
    parser.add_argument('--n_epochs', type=int, default=2,
                        help='Number of epochs to train for')
    parser.add_argument('--save_mod', type=int, default=5,
                        help='How often to save')
    parser.add_argument('--print_mod', type=int, default=2,
                        help='How often to print')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate. Default=0.001')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight Decay. Default=0.0')
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of gpus')
    parser.add_argument('--milestones', type=AC.str2list, default=[2],
                        help='Milestones for lr decay - format: <milestone1-milestone2>')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Gamma for lr scheduler')
    parser.add_argument('--loss', type=str, default='mse',
                        help='Type of loss to use: options: {"mse", "cross_entropy"}')
    parser.add_argument('--w_reg', type=AC.str2None,
                        help='Type of regularization to use for W, options: {"l1", "l2"}')
    parser.add_argument('--reg_lambda', type=float, default=1e-4,
                        help='regularization strength for w_reg')
    parser.add_argument('--autoencoder_mode', type=AC.str2bool, default=False,
                        help='True to use input as target output (i.e. autoencoder)')
    parser.add_argument('--verbose', type=AC.str2bool, default=False,
                        help='True to print')
    parser.add_argument('--print_params', type=AC.str2bool, default=True,
                        help='True to print parameters after training')
    parser.add_argument('--output_mode', type=str, default='contact',
                        help='data structure of output {"contact", "sequence", "energy"}')

    # model args
    parser.add_argument('--model_type', type=str, default='test',
                        help='Type of model')
    parser.add_argument('--id', type=AC.str2int,
                        help='id of model')
    parser.add_argument('--pretrained', type=AC.str2bool, default=False,
                        help='True if using a pretrained model')
    parser.add_argument('--resume_training', type=AC.str2bool, default=False,
                        help='True if resuming training of a partially trained model')
    parser.add_argument('--k', type=AC.str2int,
                        help='Number of input epigenetic marks')
    parser.add_argument('--m', type=int, default=1024,
                        help='Number of particles')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed to use. Default: 42')
    parser.add_argument('--act', type=AC.str2None, default='relu',
                        help='default activation') # TODO impelement throughout
    parser.add_argument('--inner_act', type=AC.str2None,
                        help='default activation (not used for all networks)')
    parser.add_argument('--out_act', type=AC.str2None,
                        help='activation of final layer')
    parser.add_argument('--training_norm', type=AC.str2None,
                        help='norm during training (batch, instance, or None)')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout probability')
    parser.add_argument('--parameter_sharing', type=AC.str2bool, default=False,
                        help='true to use parameter sharing in autoencoder blocks')
    parser.add_argument('--use_bias', type=AC.str2bool, default=True,
                        help='true to use bias (only implemented in ContactGNN and MLP)')

    # GNN model args
    parser.add_argument('--use_sign_net', type=AC.str2bool, default=False,
                        help='True to use sign net architecture')
    parser.add_argument('--message_passing', type=str, default='GCN',
                        help='type of message passing algorithm')
    parser.add_argument('--head_architecture', type=AC.str2None,
                        help='type of head architecture')
    parser.add_argument('--head_architecture_2', type=AC.str2None,
                        help='2nd type of head architecture')
    parser.add_argument('--head_hidden_sizes_list', type=AC.str2list,
                        help='List of hidden sizes for convolutional layers')
    parser.add_argument('--encoder_hidden_sizes_list', type=AC.str2list,
                        help='hidden sizes for node encoder')
    parser.add_argument('--inner_hidden_sizes_list', type=AC.str2list,
                        help='hidden sizes for inner architecture')
    parser.add_argument('--edge_encoder_hidden_sizes_list', type=AC.str2list,
                        help='hidden sizes for edge encoder')
    parser.add_argument('--update_hidden_sizes_list', type=AC.str2list,
                        help='hidden sizes for update step of MPGNN')
    parser.add_argument('--head_act', type=AC.str2None, default='relu',
                        help='activation function for head network')
    parser.add_argument('--num_heads', type=AC.str2int, default=1,
                        help='number of attention heads for relevant MPGNN')
    parser.add_argument('--concat_heads', type=AC.str2bool, default=True,
                        help='False to average instead of concat attention heads')
    parser.add_argument('--max_diagonal', type=AC.str2int,
                        help='Maximum diagonal to consider')
    parser.add_argument('--mlp_model_id', type=AC.str2int,
                        help='Model ID for MLP diagonal parameters')

    # SimpleEpiNet args
    parser.add_argument('--kernel_w_list', type=AC.str2list,
                        help='List of kernel widths of convolutional layers')
    parser.add_argument('--hidden_sizes_list', type=AC.str2list,
                        help='List of hidden sizes for convolutional layers')

    # UNet args
    parser.add_argument('--nf', type=int, help='Number of filters')

    # DeepC args
    parser.add_argument('--dilation_list', type=AC.str2list,
                        help='List of dilations for dilated convolutional layers')

    # Akita args
    parser.add_argument('--dilation_list_trunk', type=AC.str2list,
                        help='List of dilations for dilated convolutional layers of trunk')
    parser.add_argument('--bottleneck', type=int,
                        help='Number of filters in bottleneck (must be <= hidden_size_dilation_trunk)')
    parser.add_argument('--dilation_list_head', type=AC.str2list,
                        help='List of dilations for dilated convolutional layers of head')
    parser.add_argument('--down_sampling', type=AC.str2None,
                        help='type of down sampling to use')

    # post-processing args
    parser.add_argument('--plot', type=AC.str2bool, default=True,
                        help='True to plot result figures')
    parser.add_argument('--plot_predictions', type=AC.str2bool, default=True,
                        help='True to plot predictions')

    return parser

def finalize_opt(opt, parser, windows = False, local = False, debug = False):
    '''
    Helper function to processes command line arguments.

    Inputs:
        opt (options): parsed command line arguments from parser.parse_args()
        parser: instance of argparse.ArgumentParser() - used to re-parse if needed
        windows: True for windows file path
        local: True to override copy_data_to_scratch
        debug: True for debug mode (won't throw warning for resume_training)

    Outputs:
        opt
    '''

    # set up output folders/files
    if windows:
        model_type_root = 'C:/Users/Eric/OneDrive/Documents/Research/Coding'
    else:
        model_type_root = '/home/erschultz'
    model_type_folder = osp.join(model_type_root, 'sequences_to_contact_maps', 'results',
                                opt.model_type)

    if opt.resume_training:
        assert opt.id is not None

    if opt.id is None:
        if not osp.exists(model_type_folder):
            os.mkdir(model_type_folder, mode = 0o755)
            opt.id = 1
        else:
            max_id = 0
            for filename in os.listdir(model_type_folder):
                if filename.isnumeric():
                    id = int(filename)
                    if id > max_id:
                        max_id = id
            opt.id = max_id + 1
    else:
        txt_file = osp.join(model_type_folder, str(opt.id), 'argparse.txt')
        if osp.exists(txt_file):
            assert opt.resume_training or debug, f"issue with id={opt.id}"
            id_copy = opt.id
            args = sys.argv.copy() # need to copy if running finalize_opt multiple times
            args.insert(1, f'@{txt_file}')
            args.pop(0) # remove program name
            opt = parser.parse_args(args) # parse again
            # by inserting at position 1, the original arguments will override the txt file
            opt.id = id_copy

    opt.ofile_folder = osp.join(model_type_folder, str(opt.id))
    if not osp.exists(opt.ofile_folder):
        os.mkdir(opt.ofile_folder, mode = 0o755)
    opt.log_file_path = osp.join(opt.ofile_folder, 'out.log')
    opt.log_file = open(opt.log_file_path, 'a')

    param_file_path = osp.join(opt.ofile_folder, 'params.log')
    opt.param_file = open(param_file_path, 'a')

    # configure other model params
    assert (opt.split_percents is None) ^ (opt.split_sizes is None)

    opt.split_neg_pos_edges = False
    if opt.message_passing.lower() == 'signedconv':
        opt.split_neg_pos_edges = True
    elif opt.message_passing.lower() == 'gat':
        assert not opt.use_edge_weights

    # configure loss
    if opt.loss == 'mse':
        opt.criterion = F.mse_loss
        opt.channels = 1
    elif opt.loss == 'huber':
        opt.criterion = F.huber_loss
        opt.channels = 1
    elif opt.loss == 'cross_entropy':
        assert opt.out_act is None, "Cannot use output activation with cross entropy"
        assert not opt.GNN_mode, 'cross_entropy not tested for GNN'
        assert opt.y_preprocessing == 'prcnt', 'must use percentile preprocessing with cross entropy'
        assert opt.preprocessing_norm is None, 'Cannot normalize with cross entropy'
        opt.channels = opt.classes
        opt.y_reshape = False
        opt.criterion = F.cross_entropy
        opt.ydtype = torch.int64
    elif opt.loss == 'BCE':
        assert opt.out_act is None, "Cannot use output activation with BCE"
        if opt.output_mode == 'contact':
            assert opt.preprocessing_norm is not None, 'must use some sort of preprocessing_norm'
        opt.criterion = F.binary_cross_entropy_with_logits
    else:
        raise Exception(f'Invalid loss: {repr(opt.loss)}')

    # check mode
    if opt.model_type.startswith('GNNAutoencoder') or opt.model_type.startswith('ContactGNN'):
        assert opt.GNN_mode, 'mode should be GNN'

    # configure GNN transforms
    if opt.GNN_mode:
        opt.node_feature_size = 0
        if opt.use_node_features:
            assert opt.k is not None
            opt.node_feature_size += opt.k
        else:
            assert (len(opt.transforms) + len(opt.pre_transforms)) > 0, f"need feature augmentation for id={opt.id}"

    if opt.rescale is not None:
        assert opt.rescale != 0, f'{opt.id}'
        opt.input_m = int(opt.m / opt.rescale)
    else:
        opt.input_m = opt.m
    if opt.crop is not None:
        opt.m = opt.crop[1] - opt.crop[0]
        opt.input_m = opt.crop[1] - opt.crop[0]

    # transforms
    process_transforms(opt)

    # move data to scratch
    if opt.use_scratch and not local:
        copy_data_to_scratch(opt)

    # configure cuda
    if opt.gpus > 1:
        opt.cuda = True
        opt.use_parallel = True
        opt.gpu_ids = []
        for ii in range(6):
            try:
                torch.cuda.get_device_properties(ii)
                print(str(ii), file = opt.log_file)
                opt.gpu_ids.append(ii)
            except AssertionError:
                print('Not ' + str(ii) + "!", file = opt.log_file)
    elif opt.gpus == 1:
        opt.cuda = True
        opt.use_parallel = False
    else:
        opt.cuda = False
        opt.use_parallel = False

    if opt.cuda and not torch.cuda.is_available():
        if not local:
            print('Warning: falling back to cpu', file = opt.log_file)
        opt.cuda = False
        opt.use_parallel = False

    opt.device = torch.device('cuda' if opt.cuda else 'cpu')

    # Set random seeds
    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)

    opt.log_file.close() # save any writes so far
    opt.log_file = open(opt.log_file_path, 'a')
    return opt

def process_transforms(opt):
    # collect these for printing purposes (see opt2list)
    opt.edge_transforms = []
    opt.node_transforms = []

    if opt.log_preprocessing is not None:
        default_split_value = 0
    else:
        default_split_value = 1
    opt.edge_dim = 0

    # transforms
    transforms_processed = []
    for t_str in opt.transforms:
        t_str = t_str.lower().split('_')
        if t_str[0] == 'constant':
            opt.node_transforms.append(torch_geometric.transforms.Constant())
            transforms_processed.append(torch_geometric.transforms.Constant())
            opt.node_feature_size += 1
        elif t_str[0] == 'sparse':
            opt.node_transforms.append(torch_geometric.transforms.ToSparseTensor())
            transforms_processed.append(torch_geometric.transforms.ToSparseTensor())
        else:
            raise Exception("Invalid transform {}".format(t_str))
    if len(transforms_processed) > 0:
        opt.transforms_processed = torch_geometric.transforms.Compose(transforms_processed)
    else:
        opt.transforms_processed = None

    # pre-transforms
    processed = []
    opt.diag = False # True if y_diag is needed for transform
    for t_str in opt.pre_transforms:
        t_str = t_str.lower().split('_')
        if t_str[0] == 'constant':
            transform = torch_geometric.transforms.Constant()
            opt.node_transforms.append(transform)
            opt.node_feature_size += 1
        elif t_str[0] == 'weightedldp':
            transform = WeightedLocalDegreeProfile()
            opt.node_transforms.append(transform)
            if (opt.top_k is None and
                opt.sparsify_threshold is None and
                opt.sparsify_threshold_upper is None):
                print('Warning: using LDP without any sparsification')
            opt.node_feature_size += 5
        elif t_str[0] == 'degree' or t_str[0] == 'weighteddegree':
            if t_str[0] == 'weighteddegree':
                weighted = True
            else:
                weighted = False
            opt.node_feature_size += 1
            split = False
            norm = True
            max_value = None
            split_value = default_split_value
            for mode_str in t_str[1:]:
                if mode_str == 'diag':
                    opt.diag = True
                if mode_str.startswith('split'):
                    if len(mode_str) > 5:
                        assert mode_str[5:].isnumeric()
                        split_value = float(mode_str[5:])
                    split = True
                    opt.node_feature_size += 1
                if mode_str.startswith('max'):
                    assert mode_str[3:].isnumeric()
                    max_value = float(mode_str[3:])
            transform = Degree(split_val = split_value, diag = opt.diag,
                            split_edges = split, max_val = max_value,
                            weighted = weighted)
            opt.node_transforms.append(transform)
        elif t_str[0] == 'onehotdegree':
            opt.node_feature_size += opt.m + 1
            transform = torch_geometric.transforms.OneHotDegree(opt.m)
            opt.node_transforms.append(transform)
        elif t_str[0] == 'adj':
            opt.node_feature_size += opt.m
            processed.append(AdjTransform())
            opt.node_transforms.append(AdjTransform())
        elif t_str[0] == 'adjpca':
            k = 8
            diag = False
            for mode_str in t_str[1:]:
                if mode_str == 'diag':
                    diag = True
                    opt.diag = True
                elif mode_str.isdigit():
                    k = int(mode_str)
            opt.node_feature_size += k
            transform = AdjPCATransform(k, diag)
            opt.node_transforms.append(transform)
        elif t_str[0] == 'adjpcs':
            opt.diag = True
            norm = False
            k = 8
            for mode_str in t_str[1:]:
                if mode_str == 'norm':
                    norm = True
                elif mode_str.isdigit():
                    k = int(mode_str)
            opt.node_feature_size += k
            transform = AdjPCs(k, norm)
            opt.node_transforms.append(transform)
        elif t_str[0] == 'contactdistance':
            opt.edge_transforms.append(f'ContactDistance')
            assert opt.use_edge_attr or opt.use_edge_weights
            if opt.use_edge_attr:
                opt.edge_dim += 1
            norm = False
            for mode_str in t_str[1:]:
                if mode_str == 'norm':
                    norm = True

            transform = ContactDistance(norm = norm,
                                        split_edges = opt.split_neg_pos_edges,
                                        convert_to_attr = opt.use_edge_attr)
            opt.edge_transforms.append(transform)
        elif t_str[0] == 'geneticdistance':
            assert opt.use_edge_attr or opt.use_edge_weights
            if opt.use_edge_attr:
                opt.edge_dim += 1
            log = False
            log10 = False
            norm = False
            for mode_str in t_str[1:]:
                if mode_str == 'log':
                    log = True
                if mode_str == 'log10':
                    log10 = True
                if mode_str == 'norm':
                    norm = True

            transform = GeneticDistance(split_edges = opt.split_neg_pos_edges,
                                        convert_to_attr = opt.use_edge_attr,
                                        log = log, log10 = log10, norm = norm)
            opt.edge_transforms.append(transform)
        elif t_str[0] == 'geneticposition':
            center = False
            norm = False
            for mode_str in t_str[1:]:
                if mode_str == 'center':
                    center = True
                elif mode_str == 'norm':
                    norm = True

            opt.node_feature_size += 1
            transform = GeneticPosition(center = center, norm = norm)
            opt.node_transforms.append(transform)
        elif t_str[0] == 'onehotgeneticposition':
            transform = OneHotGeneticPosition()
            opt.node_feature_size += opt.m
            opt.node_transforms.append(transform)
        elif t_str[0] == 'noiselevel':
            inverse = False
            for mode_str in t_str[1:]:
                if mode_str == 'inverse':
                    inverse = True
            transform = NoiseLevel(inverse)
            opt.node_transforms.append(transform)
            opt.node_feature_size += 1
        elif t_str[0] == 'diagonalparameterdistance':
            assert opt.use_edge_attr or opt.use_edge_weights

            if len(t_str) > 1 and t_str[1].isdigit():
                mlp_id = int(t_str[1])
            else:
                mlp_id = None
            if opt.use_edge_attr:
                opt.edge_dim += 1

            transform = DiagonalParameterDistance(split_edges = opt.split_neg_pos_edges,
                                            convert_to_attr = opt.use_edge_attr, id = mlp_id)
            opt.edge_transforms.append(transform)
        elif t_str[0] == 'evd':
            transform = EVDTransform('sym')
            opt.node_transforms.append(transform)
        else:
            raise Exception(f'Unrecognized transform: {t_str} for id={opt.id}')
        processed.append(transform)

    if len(processed) > 0:
        opt.pre_transforms_processed = torch_geometric.transforms.Compose(processed)
    else:
        opt.pre_transforms_processed = None

    # these are used in opt2list for making results table
    opt.edge_transforms = sorted([repr(i) for i in opt.edge_transforms])
    opt.node_transforms = sorted([repr(i) for i in opt.node_transforms])
    # see pre_transforms_processed for more complete description of transforms

def copy_data_to_scratch(opt):
    t0 = time.time()
    # initialize scratch path
    if not osp.exists(opt.scratch):
        os.mkdir(opt.scratch, mode = 0o700)


    datasets = [osp.split(d)[-1] for d in opt.data_folder]
    data_folder_str ='-'.join(datasets)
    scratch_path = osp.join(opt.scratch, data_folder_str)
    if not osp.exists(scratch_path):
        os.mkdir(scratch_path, mode = 0o700)

    # initialize samples folder
    if not osp.exists(osp.join(scratch_path, 'samples')):
        os.mkdir(osp.join(scratch_path, 'samples'), mode = 0o700)

    for data_folder in opt.data_folder:
        # transfer summary files
        for file in os.listdir(data_folder):
            file_dir = osp.join(data_folder, file)
            scratch_file_dir = osp.join(scratch_path, file)
            if file.endswith('npy') and not osp.exists(scratch_file_dir):
                shutil.copyfile(file_dir, scratch_file_dir)

        # transfer sample data
        samples = os.listdir(osp.join(data_folder, 'samples'))
        if opt.use_scratch_parallel:
            mapping = []
            for sample in samples:
                mapping.append([sample, data_folder, scratch_path, opt.toxx, opt.y_preprocessing, opt.output_mode])
            with multiprocessing.Pool(opt.num_workers) as p:
                 p.starmap(copy_data_to_scratch_inner, mapping)
        else:
            for sample in samples:
                copy_data_to_scratch_inner(sample, data_folder, scratch_path, opt.toxx, opt.y_preprocessing, opt.output_mode)

    opt.data_folder = [scratch_path]

    tf = time.time()
    delta_t = np.round(tf - t0, 0)
    print("Took {} seconds to move data to scratch".format(delta_t), file = opt.log_file)

def copy_data_to_scratch_inner(sample, data_folder, scratch_path, toxx, y_preprocessing, output_mode):
    sample_dir = osp.join(data_folder, 'samples', sample)
    if not osp.isdir(sample_dir):
        return

    scratch_sample_dir = osp.join(scratch_path, 'samples', sample)
    if not osp.exists(scratch_sample_dir):
        os.mkdir(scratch_sample_dir, mode = 0o700)
    for file in os.listdir(sample_dir):
        move_file = False
        # change to True to move file
        # defaults to not moving file (saves space on scratch and move time)

        if '.' not in file:
            continue

        fname, ftype = file.split('.')

        if file == 'xx.npy' and toxx:
            # only need xx.npy if toxx is True
            move_file = True
        elif file == 'y.npy' and (y_preprocessing is None or y_preprocessing.startswith('log') or output_mode.startswith('diag')):
            move_file = True
        elif y_preprocessing is not None and fname.endswith(y_preprocessing) and ftype == 'npy':
            # need file corresponding to y_preprocessing
            move_file = True
        elif (file == 's.npy' or file == 'e.npy') and output_mode.startswith('energy'):
            # only need s.npy if neural net output is energy
            move_file = True
        elif file == 'config.json' and output_mode.startswith('diag'):
            # need to config to get diag_chi
            move_file = True
        elif file == 'params.log' and output_mode.startswith('diag_param'):
            # need params log to get diag_chi params
            move_file = True
        elif file == 'diag_chis_continuous.npy' and (output_mode.startswith('diag') or output_mode.startswith('energy')):
            # need to load this file regardless of which version of diag
            move_file = True

        if move_file:
            source_file = osp.join(sample_dir, file)
            destination_file = osp.join(scratch_sample_dir, file)
            if not osp.exists(destination_file):
                shutil.copyfile(source_file, destination_file)

    # if y_preprocessing is not None and y_preprocessing.startswith('sweep'):
    #     sweep, *y_preprocessing = y_preprocessing.split('_')
    #     sweep = int(sweep[5:])
    #     if isinstance(y_preprocessing, list):
    #         y_preprocessing = '_'.join(y_preprocessing)
    #
    #     ifile = osp.join(sample_dir, f'data_out/contacts{sweep}.txt')
    #     if osp.exists(ifile):
    #         y = np.loadtxt(ifile)
    #         if y_preprocessing == 'log':
    #             y_to_move = np.log(y+1)
    #         elif y_preprocessing == 'log_diag':
    #             y_log = np.log(y+1)
    #             meanDist = DiagonalPreprocessing.genomic_distance_statistics(y_log)
    #             y_to_move = DiagonalPreprocessing.process(y_log, meanDist)
    #         elif y_preprocessing == 'diag':
    #             meanDist = DiagonalPreprocessing.genomic_distance_statistics(y)
    #             y_to_move = DiagonalPreprocessing.process(y, meanDist)
    #
    #         np.save(osp.join(scratch_sample_dir, f'y_sweep{sweep}_{y_preprocessing}'), y_to_move)

def argparse_setup(local = False):
    """Helper function set up parser."""
    parser = get_base_parser()
    opt = parser.parse_args()
    return finalize_opt(opt, parser, local)

def save_args(opt):
    with open(osp.join(opt.ofile_folder, 'argparse.txt'), 'w') as f:
        for arg in sys.argv[1:]: # skip the program file
            f.write(arg + '\n')

def opt2list(opt):
    data_folder = '-'.join([osp.split(d)[1] for d in opt.data_folder])
    opt_list = [opt.model_type, opt.id, data_folder, opt.preprocessing_norm,
        opt.y_preprocessing, opt.output_preprocesing, opt.mean_filt, opt.rescale,
        opt.kr, opt.min_subtraction, opt.log_preprocessing, opt.crop]
    opt_list.append(opt.split_percents if opt.split_percents is not None else opt.split_sizes)
    opt_list.extend([opt.shuffle, opt.batch_size, opt.num_workers, opt.n_epochs, opt.lr,
        opt.weight_decay, opt.w_reg,
        opt.milestones, opt.gamma, opt.loss,
        opt.k, opt.m, opt.seed, opt.act, opt.inner_act,
        opt.head_act, opt.out_act, opt.training_norm])
    if opt.GNN_mode:
        opt_list.extend([opt.use_node_features, opt.use_edge_weights, opt.use_edge_attr,
                        opt.node_transforms, opt.edge_transforms,
                        opt.sparsify_threshold, opt.sparsify_threshold_upper, opt.max_diagonal,
                        opt.hidden_sizes_list, opt.message_passing, opt.update_hidden_sizes_list,
                        f'{opt.head_architecture}+{opt.head_architecture_2}', opt.head_hidden_sizes_list])

    if opt.model_type == 'simpleEpiNet':
        opt_list.extend([opt.kernel_w_list, opt.hidden_sizes_list])
    elif opt.model_type == 'UNet':
        opt_list.extend([opt.nf, opt.toxx, opt.toxx_mode])
    elif opt.model_type == 'Akita':
        opt_list.extend([opt.kernel_w_list, opt.hidden_sizes_list, opt.dilation_list_trunk,
                        opt.bottleneck, opt.dilation_list_head, opt.down_sampling])
    elif opt.model_type == 'DeepC':
        opt_list.extend([opt.kernel_w_list, opt.hidden_sizes_list, opt.dilation_list])
    elif opt.model_type == 'test':
        opt_list.extend([opt.kernel_w_list, opt.hidden_sizes_list, opt.dilation_list_trunk,
                        opt.bottleneck, opt.dilation_list_head, opt.nf])
    elif opt.model_type.startswith('GNNAutoencoder'):
        opt_list.extend([opt.head_act, opt.parameter_sharing])
    elif opt.model_type.startswith('ContactGNN'):
        pass
    elif opt.model_type == 'SequenceFCAutoencoder':
        opt_list.extend([opt.hidden_sizes_list, opt.parameter_sharing])
    elif opt.model_type == 'MLP':
        opt_list.extend([opt.y_zero_diag_count, opt.hidden_sizes_list])
    else:
        raise Exception("Unknown model type: {}".format(opt.model_type))

    opt_list.append(opt.output_mode)

    return opt_list

def save_opt(opt, ofile):
    if not osp.exists(ofile):
        with open(ofile, 'w', newline = '') as f:
            wr = csv.writer(f)
            opt_list = get_opt_header(opt.model_type, opt.GNN_mode)
            wr.writerow(opt_list)
    with open(ofile, 'a') as f:
        wr = csv.writer(f)
        opt_list = opt2list(opt)
        wr.writerow(opt_list)

def get_opt_header(model_type, GNN_mode):
    opt_list = ['model_type', 'id',  'dataset', 'preprocessing_norm',
        'y_preprocessing', 'output_preprocesing', 'mean_filt', 'rescale',
        'kr', 'min_subtraction', 'log_preprocessing', 'crop', 'split', 'shuffle',
        'batch_size', 'num_workers', 'n_epochs', 'lr', 'weight_decay', 'w reg', 'milestones',
        'gamma', 'loss', 'k', 'm',
        'seed', 'act', 'inner_act', 'head_act', 'out_act',
        'training_norm']
    if GNN_mode:
        opt_list.extend(['use_node_features','use_edge_weights', 'use_edge_attr',
                        'node_transforms', 'edge_transforms',
                        'sparsify_threshold', 'sparsify_threshold_upper', 'max_diagonal',
                        'hidden_sizes_list', 'message_passing', 'update_hidden_sizes_list',
                        'head_architecture', 'head_hidden_sizes_list'])
    if model_type == 'simpleEpiNet':
        opt_list.extend(['kernel_w_list', 'hidden_sizes_list'])
    elif model_type == 'UNet':
        opt_list.extend(['nf','toxx', 'toxx_mode'])
    elif model_type == 'Akita':
        opt_list.extend(['kernel_w_list', 'hidden_sizes_list', 'dilation_list_trunk',
                        'bottleneck', 'dilation_list_head', 'down_sampling'])
    elif model_type == 'DeepC':
        opt_list.extend(['kernel_w_list', 'hidden_sizes_list', 'dilation_list'])
    elif model_type == 'test':
        opt_list.extend(['kernel_w_list', 'hidden_sizes_list', 'dilation_list_trunk',
                        'bottleneck', 'dilation_list_head', 'nf'])
    elif model_type == 'GNNAutoencoder':
        opt_list.extend(['head_act', 'head_hidden_sizes_list'])
    elif model_type.startswith('ContactGNN'):
        pass
    elif model_type == 'SequenceFCAutoencoder':
        opt_list.extend(['hidden_sizes_list', 'parameter_sharing'])
    elif model_type == 'MLP':
        opt_list.extend(['y_zero_diag_count', 'hidden_sizes_list'])
    else:
        raise Exception("Unknown model type: {}".format(model_type))

    opt_list.append('output_mode')

    return opt_list

class ArgparserConverter():
    @staticmethod
    def str2None(v):
        """
        Helper function for argparser, converts str to None if str == 'none'

        Returns the string otherwise.

        Inputs:
            v: string
        """
        if v is None:
            return v
        elif isinstance(v, str):
            if v.lower() == 'none':
                return None
            else:
                return v
        else:
            raise argparse.ArgumentTypeError('String value expected.')

    @staticmethod
    def str2int(v):
        """
        Helper function for argparser, converts str to int if possible.

        Inputs:
            v: string
        """
        if v is None:
            return v
        elif isinstance(v, str):
            if v.lower() == 'none':
                return None
            elif v.isnumeric():
                return int(v)
            elif v[0] == '-' and v[1:].isnumeric():
                return int(v)
            else:
                raise argparse.ArgumentTypeError('none or int expected not {}'.format(v))
        else:
            raise argparse.ArgumentTypeError('String value expected.')

    @staticmethod
    def str2float(v):
        """
        Helper function for argparser, converts str to float if possible.

        Inputs:
            v: string
        """
        if v is None:
            return v
        elif isinstance(v, str):
            if v.lower() == 'none':
                return None
            elif v.replace('.', '').replace('-', '').isnumeric():
                return float(v)
            else:
                raise argparse.ArgumentTypeError('none or float expected not {}'.format(v))
        else:
            raise argparse.ArgumentTypeError('String value expected.')

    @staticmethod
    def str2bool(v):
        """
        Helper function for argparser, converts str to boolean for various string inputs.
        https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

        Inputs:
            v: string
        """
        if isinstance(v, bool):
           return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    def is_float(v) -> bool:
        try:
            float(v)
            return True
        except ValueError:
            return False

    @staticmethod
    def str2list(v, sep = '-'):
        """
        Helper function for argparser, converts str to list by splitting on sep.
        Empty string will be mapped to -1.

        Example for sep = '-': "-i-j-k" -> [-1, i, j, k]

        Inputs:
            v: string
            sep: separator
        """
        if v is None:
            return None
        elif isinstance(v, str):
            if v.lower() == 'none':
                return None
            elif v.lower() == 'empty':
                return []
            else:
                result = [i for i in v.split(sep)]
                for i, val in enumerate(result):
                    if val.isnumeric():
                        result[i] = int(val)
                    elif ArgparserConverter.is_float(val):
                        result[i] = float(val)
                    elif val == '':
                        result[i] = -1
                return result
        else:
            raise argparse.ArgumentTypeError('str value expected.')

    @staticmethod
    def str2list2D(v, sep1 = '\\', sep2 = '&'):
        """
        Helper function for argparser, converts str to list by splitting on sep1, then on sep2.

        Example for sep1 = '\\', sep2 = '&': "i & j \\ k & l" -> [[i, j], [k, l]]

        Inputs:
            v: string (any spaces will be ignored)
            sep: separator
        """
        if v is None:
            return None
        elif isinstance(v, str):
            if v.lower() == 'none':
                return None
            elif v.lower() in {'nonlinear', 'polynomial'}:
                return v.lower()
            else:
                v = v.replace(' ', '') # get rid of spaces
                result = [i.split(sep2) for i in v.split(sep1)]
                result = np.array(result, dtype=float)
                return result
        else:
            raise argparse.ArgumentTypeError('str value expected.')

    @staticmethod
    def str2dtype(v):
        """
        Helper function for argparser, converts str to torch dtype.

        Inputs:
            v: string
        """
        if isinstance(v, str):
            if v == 'float32':
                return torch.float32
            elif v == 'float64':
                return torch.float64
            elif v == 'int32':
                return torch.int32
            elif v == 'int64':
                return torch.int64
            else:
                raise Exception('Unkown str: {}'.format(v))
        else:
            raise argparse.ArgumentTypeError('str value expected.')

    def list2str(v, sep = '-'):
        """
        Helper function to convert list to string.

        Inputs:
            v: list
        """
        if isinstance(v, list):
            return sep.join([str(i) for i in v])
        else:
            raise Exception('list value expected.')

    @staticmethod
    def float2str(v):
        """
        Helper function to convert float to str in si notation.

        Inputs:
            v: float
        """
        # TODO make this more robust
        if isinstance(v, float):
            vstr = "{:.1e}".format(v)
            if vstr[2] == '0':
                # converts 1.0e-04 to 1e-04
                vstr = vstr[0:1] + vstr[3:]
            if vstr[-2] == '0':
                # converts 1e-04 to 1e-4
                vstr = vstr[0:-2] + vstr[-1]
        else:
            raise Exception('float value expected.')
        return vstr
