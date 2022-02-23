import argparse
import csv
import os
import os.path as osp
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms


def getBaseParser():
    '''Helper function that returns base parser'''
    parser = argparse.ArgumentParser(description='Base parser', fromfile_prefix_chars='@')

    # GNN pre-processing args
    parser.add_argument('--GNN_mode', type=str2bool, default=False, help='True to use GNNs (uses pytorch_geometric in core_test_train)')
    parser.add_argument('--transforms', type=str2list, help='list of transforms to use for GNN')
    parser.add_argument('--pre_transforms', type=str2list, help='list of pre-transforms to use for GNN')
    parser.add_argument('--sparsify_threshold', type=str2float, help='remove all edges with weight < threshold (None to do nothing)')
    parser.add_argument('--sparsify_threshold_upper', type=str2float, help='remove all edges with weight < threshold (None to do nothing)')
    parser.add_argument('--top_k', type=str2int, help='filter to top k largest edges per node (None to do nothing)')
    parser.add_argument('--use_node_features', type=str2bool, default=False, help='True to use node features for GNN models')
    parser.add_argument('--use_edge_weights', type=str2bool, default=True, help='True to use edge weights in GNN')
    parser.add_argument('--relabel_11_to_00',type=str2bool, default=False, help='True to relabel [1,1] particles as [0,0] particles')
    parser.add_argument('--split_neg_pos_edges_for_feature_augmentation', type=str2bool, default=False, help='True to split edges for feature augmentation')

    # pre-processing args
    parser.add_argument('--data_folder', type=str, default='dataset_04_18_21', help='Location of data')
    parser.add_argument('--scratch', type=str, default='/scratch/midway2/erschultz', help='Location of scratch dir')
    parser.add_argument('--root_name', type=str2None, help='name of file to save graph data (leave as None to create root automatically) (root is the directory path - defined later)')
    parser.add_argument('--delete_root', type=str2bool, default=True, help='True to delete root directory after runtime')
    parser.add_argument('--toxx', type=str2bool, default=False, help='True if x should be converted to 2D image')
    parser.add_argument('--toxx_mode', type=str, default='mean', help='mode for toxx (default mean)')
    parser.add_argument('--y_preprocessing', type=str2None, default='diag', help='type of pre-processing for y')
    parser.add_argument('--y_log_transform', type=str2bool, default=False, help='True to log transform y')
    parser.add_argument('--y_norm', type=str2None, default='batch', help='type of [0,1] normalization for y')
    parser.add_argument('--min_subtraction', type=str2bool, default=True, help='if min subtraction should be used for y_norm')
    parser.add_argument('--x_reshape', type=str2bool, default=True, help='True if x should be considered a 1D image')
    parser.add_argument('--ydtype', type=str2dtype, default='float32', help='torch data type for y')
    parser.add_argument('--y_reshape', type=str2bool, default=True, help='True if y should be considered a 2D image')
    parser.add_argument('--crop', type=str2list, help='size of crop to apply to image - format: <leftcrop-rightcrop>')
    parser.add_argument('--classes', type=int, default=10, help='number of classes in percentile normalization')
    parser.add_argument('--use_scratch', type=str2bool, default=False, help='True to move data to scratch')

    # dataloader args
    parser.add_argument('--split_percents', type=str2list, help='Train, val, test split for dataset (percents)')
    parser.add_argument('--split_sizes', type=str2list, default=[-1, 200, 0], help='Train, val, test split for dataset (counts), -1 for remainder')
    parser.add_argument('--random_split', type=str2bool, default=False, help='True to use random train, val, test split')
    parser.add_argument('--shuffle', type=str2bool, default=True, help='Whether or not to shuffle dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of threads for data loader to use')

    # train args
    parser.add_argument('--start_epoch', type=int, default=1, help='Starting epoch')
    parser.add_argument('--n_epochs', type=int, default=2, help='Number of epochs to train for')
    parser.add_argument('--save_mod', type=int, default=5, help='How often to save')
    parser.add_argument('--print_mod', type=int, default=2, help='How often to print')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning eate. Default=0.001')
    parser.add_argument('--gpus', type=int, default=1, help='Number of gpus')
    parser.add_argument('--milestones', type=str2list, default=[2], help='Milestones for lr decay - format: <milestone1-milestone2>')
    parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for lr decay')
    parser.add_argument('--loss', type=str, default='mse', help='Type of loss to use: options: {"mse", "cross_entropy"}')
    parser.add_argument('--autoencoder_mode', type=str2bool, default=False, help='True to use input as target output (i.e. autoencoder)')
    parser.add_argument('--verbose', type=str2bool, default=False)
    parser.add_argument('--print_params', type=str2bool, default=True, help='True to print parameters after training')
    parser.add_argument('--output_mode', type=str, default='contact', help='data structure of output {"contact", "sequence", "energy"}')

    # model args
    parser.add_argument('--model_type', type=str, default='test', help='Type of model')
    parser.add_argument('--id', type=int, help='id of model')
    parser.add_argument('--pretrained', type=str2bool, default=False, help='True if using a pretrained model')
    parser.add_argument('--resume_training', type=str2bool, default=False, help='True if resuming traning of a partially trained model')
    parser.add_argument('--ifile_folder', type=str, help='Location of input file for pretrained model')
    parser.add_argument('--ifile', type=str, help='Name of input file for pretrained model')
    parser.add_argument('--k', type=int, default=2, help='Number of epigenetic marks')
    parser.add_argument('--m', type=int, default=1024, help='Number of particles')
    parser.add_argument('--seed', type=int, default=42, help='random seed to use. Default: 42')
    parser.add_argument('--act', type=str2None, default='relu', help='default activation') # TODO impelement throughout
    parser.add_argument('--inner_act', type=str2None, help='default activation (not used for all networks)')
    parser.add_argument('--out_act', type=str2None, help='activation of final layer')
    parser.add_argument('--training_norm', type=str2None, help='norm during training (batch, instance, or None)')
    parser.add_argument('--parameter_sharing', type=str2bool, default=False, help='true to use parameter sharing in autoencoder blocks')
    parser.add_argument('--use_bias', type=str2bool, default=True, help='true to use bias (only implemented in ContactGNN)')

    # GNN model args
    parser.add_argument('--message_passing', type=str, default='GCN', help='type of message passing algorithm')
    parser.add_argument('--head_architecture', type=str2None, help='type of head architecture')
    parser.add_argument('--head_hidden_sizes_list', type=str2list, help='List of hidden sizes for convolutional layers')
    parser.add_argument('--encoder_hidden_sizes_list', type=str2list, help='hidden sizes for encoder')
    parser.add_argument('--update_hidden_sizes_list', type=str2list, help='hidden sizes for update step of MPGNN')
    parser.add_argument('--head_act', type=str2None, default='relu', help='activation function for head network')

    # SimpleEpiNet args
    parser.add_argument('--kernel_w_list', type=str2list, help='List of kernel widths of convolutional layers')
    parser.add_argument('--hidden_sizes_list', type=str2list, help='List of hidden sizes for convolutional layers')

    # UNet args
    parser.add_argument('--nf', type=int, help='Number of filters')

    # DeepC args
    parser.add_argument('--dilation_list', type=str2list, help='List of dilations for dilated convolutional layers')

    # Akita args
    parser.add_argument('--dilation_list_trunk', type=str2list, help='List of dilations for dilated convolutional layers of trunk')
    parser.add_argument('--bottleneck', type=int, help='Number of filters in bottleneck (must be <= hidden_size_dilation_trunk)')
    parser.add_argument('--dilation_list_head', type=str2list, help='List of dilations for dilated convolutional layers of head')
    parser.add_argument('--down_sampling', type=str2None, help='type of down sampling to use')

    # post-processing args
    parser.add_argument('--plot', type=str2bool, default=True, help='True to plot result figures')
    parser.add_argument('--plot_predictions', type=str2bool, default=True, help='True to plot predictions')

    return parser

def finalizeOpt(opt, parser, local = False):
    '''
    Helper function to processes command line arguments.

    Inputs:
        opt (options): parsed command line arguments from parser.parse_args()
        parser: instance of argparse.ArgumentParser() - used to re-parse if needed
        local: True to overide some commands when working locally

    Outputs:
        opt
    '''

    # set up output folders/files
    if local:
        model_type_folder = osp.join('/home/eric/sequences_to_contact_maps/results', opt.model_type)
    else:
        model_type_folder = osp.join('/home/erschultz/sequences_to_contact_maps/results', opt.model_type)

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
        assert osp.exists(txt_file), "{} does not exist".format(txt_file)
        id_copy = opt.id
        args = sys.argv.copy() # need to copy if running finalizeOpt multiple times
        args.insert(1, '@{}'.format(txt_file))
        args.pop(0) # remove program name
        opt = parser.parse_args(args) # parse again
        # by inserting at position 1, the original arguments will override the txt file
        opt.id = id_copy

    opt.ofile_folder = osp.join(model_type_folder, str(opt.id))
    if not osp.exists(opt.ofile_folder):
        os.mkdir(opt.ofile_folder, mode = 0o755)
    log_file_path = osp.join(opt.ofile_folder, 'out.log')
    opt.log_file = open(log_file_path, 'a')

    param_file_path = osp.join(opt.ofile_folder, 'params.log')
    opt.param_file = open(param_file_path, 'a')

    # configure other model params
    assert opt.split_percents is not None or opt.split_counts is not None, "both can't be None"
    assert opt.split_percents is None or opt.split_counts is None, "one must be None"

    if opt.y_log_transform:
        assert opt.y_norm is None, "don't use log transform with y norm"

    opt.split_neg_pos_edges = False
    if opt.message_passing.lower() == 'signedconv':
        opt.split_neg_pos_edges = True
        if opt.use_edge_weights:
            opt.use_edge_weights = False
            print('Setting use_edge_weights to False', file = opt.log_file)

    # configure loss
    if opt.loss == 'mse':
        opt.criterion = F.mse_loss
        opt.channels = 1
    elif opt.loss == 'cross_entropy':
        assert opt.out_act is None, "Cannot use output activation with cross entropy"
        assert not opt.GNN_mode, 'cross_entropy not tested for GNN'
        assert opt.y_preprocessing == 'prcnt', 'must use percentile preprocessing with cross entropy'
        assert opt.y_norm is None, 'Cannot normalize with cross entropy'
        opt.channels = opt.classes
        opt.y_reshape = False
        opt.criterion = F.cross_entropy
        opt.ydtype = torch.int64
    elif opt.loss == 'BCE':
        assert opt.out_act is None, "Cannot use output activation with BCE"
        if opt.output_mode == 'contact':
            assert opt.y_norm is not None, 'must use some sort of y_norm'
        opt.criterion = F.binary_cross_entropy_with_logits
    else:
        raise Exception('Invalid loss: {}'.format(repr(opt.loss)))

    # check mode
    if opt.model_type.startswith('GNNAutoencoder') or opt.model_type.startswith('ContactGNN'):
        assert opt.GNN_mode, 'mode should be GNN'

    # configure GNN transforms
    if opt.GNN_mode:
        opt.node_feature_size = 0
        if opt.use_node_features:
            opt.node_feature_size += opt.k
        else:
            assert opt.transforms is not None or opt.pre_transforms is not None, "need feature augmentation"

    opt.transforms_processed = None
    if opt.transforms is not None:
        transforms_processed = []
        for t_str in opt.transforms:
            if t_str.lower() == 'constant':
                transforms_processed.append(torch_geometric.transforms.Constant())
                opt.node_feature_size += 1
            else:
                raise Exception("Invalid transform {}".format(t_str))
        if len(transforms_processed) > 0:
            opt.transforms_processed = torch_geometric.transforms.Compose(transforms_processed)

    opt.weighted_LDP = False
    opt.degree = False
    opt.weighted_degree = False
    opt.pre_transforms_processed = None
    if opt.pre_transforms is not None:
        pre_transforms_processed = []
        for t_str in opt.pre_transforms:
            if t_str.lower() == 'constant':
                pre_transforms_processed.append(torch_geometric.transforms.Constant())
                opt.node_feature_size += 1
            elif t_str.lower() == 'weighted_ldp':
                # don't append to pre_transforms
                # instead set flag to True
                opt.weighted_LDP = True
                opt.node_feature_size += 5
            elif t_str.lower() == 'degree':
                opt.node_feature_size += 1
                if opt.split_neg_pos_edges_for_feature_augmentation:
                    opt.node_feature_size += 2
                    # additional feature for neg and pos
                opt.degree = True
            elif t_str.lower() == 'weighted_degree':
                opt.node_feature_size += 1
                if opt.split_neg_pos_edges_for_feature_augmentation:
                    opt.node_feature_size += 2
                    # additional feature for neg and pos
                opt.weighted_degree = True
            elif t_str.lower() == 'onehotdegree':
                opt.node_feature_size += opt.m + 1
                pre_transforms_processed.append(torch_geometric.transforms.OneHotDegree(opt.m))
            else:
                raise Exception("Invalid transform {}".format(t_str))
        if len(pre_transforms_processed) > 0:
            opt.pre_transforms_processed = torch_geometric.transforms.Compose(pre_transforms_processed)

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

    return opt

def copy_data_to_scratch(opt):
    t0 = time.time()
    # initialize scratch path
    scratch_path = osp.join(opt.scratch, osp.split(opt.data_folder)[-1])
    if not osp.exists(scratch_path):
        os.mkdir(scratch_path, mode = 0o700)

    # transfer summary files
    for file in os.listdir(opt.data_folder):
        file_dir = osp.join(opt.data_folder, file)
        scratch_file_dir = osp.join(scratch_path, file)
        if file.endswith('npy') and not osp.exists(scratch_file_dir):
            shutil.copyfile(file_dir, scratch_file_dir)

    # initialize samples folder
    if not osp.exists(osp.join(scratch_path, 'samples')):
        os.mkdir(osp.join(scratch_path, 'samples'), mode = 0o700)

    # transfer sample data
    for sample in os.listdir(osp.join(opt.data_folder, 'samples')):
        sample_dir = osp.join(opt.data_folder, 'samples', sample)
        scratch_sample_dir = osp.join(scratch_path, 'samples', sample)
        if not osp.exists(scratch_sample_dir):
            os.mkdir(scratch_sample_dir, mode = 0o700)
        for file in os.listdir(sample_dir):
            # skip transferring certain files if not needed (saves space on scratch and move time)
            if file == 'xx.npy' and not opt.toxx:
                # only need xx.npy if toxx is True
                pass
            elif file == 'y_prcnt.npy' and opt.y_preprocessing != 'prcnt':
                # only need y_prcnt.npy if using percentile preprocessing
                pass
            elif file == 's.npy' and opt.output_mode != 'energy':
                # only need s.npy if neural net output is energy
                pass
            elif file.endswith('npy'):
                # only move .npy files
                source_file = osp.join(sample_dir, file)
                destination_file = osp.join(scratch_sample_dir, file)
                if not osp.exists(destination_file):
                    shutil.copyfile(source_file, destination_file)

    opt.data_folder = scratch_path

    tf = time.time()
    delta_t = np.round(tf - t0, 0)
    print("Took {} seconds to move data to scratch".format(delta_t), file = opt.log_file)

def argparseSetup(local = False):
    """Helper function set up parser."""
    parser = getBaseParser()
    opt = parser.parse_args()
    return finalizeOpt(opt, parser, local)

def save_args(opt):
    with open(osp.join(opt.ofile_folder, 'argparse.txt'), 'w') as f:
        for arg in sys.argv[1:]: # skip the program file
            f.write(arg + '\n')

def opt2list(opt):
    opt_list = [opt.model_type, opt.id, opt.data_folder, opt.y_preprocessing,
        opt.y_norm, opt.min_subtraction, opt.y_log_transform, opt.crop, opt.split_percents, opt.shuffle,
        opt.batch_size, opt.num_workers, opt.n_epochs, opt.lr, opt.gpus, opt.milestones,
        opt.gamma, opt.loss, opt.pretrained, opt.resume_training, opt.ifile_folder, opt.ifile, opt.k, opt.m,
        opt.seed, opt.act, opt.inner_act, opt.head_act, opt.out_act,
        opt.training_norm, opt.relabel_11_to_00]
    if opt.GNN_mode:
        opt_list.extend([opt.use_node_features, opt.use_edge_weights, opt.transforms, opt.pre_transforms, opt.split_neg_pos_edges_for_feature_augmentation, opt.sparsify_threshold, opt.sparsify_threshold_upper, opt.top_k,
                        opt.hidden_sizes_list, opt.message_passing, opt.head_architecture, opt.head_hidden_sizes_list])

    if opt.model_type == 'simpleEpiNet':
        opt_list.extend([opt.kernel_w_list, opt.hidden_sizes_list])
    elif opt.model_type == 'UNet':
        opt_list.extend([opt.nf, opt.toxx, opt.toxx_mode])
    elif opt.model_type == 'Akita':
        opt_list.extend([opt.kernel_w_list, opt.hidden_sizes_list, opt.dilation_list_trunk, opt.bottleneck, opt.dilation_list_head, opt.down_sampling])
    elif opt.model_type == 'DeepC':
        opt_list.extend([opt.kernel_w_list, opt.hidden_sizes_list, opt.dilation_list])
    elif opt.model_type == 'test':
        opt_list.extend([opt.kernel_w_list, opt.hidden_sizes_list, opt.dilation_list_trunk, opt.bottleneck, opt.dilation_list_head, opt.nf])
    elif opt.model_type.startswith('GNNAutoencoder'):
        opt_list.extend([opt.head_act, opt.parameter_sharing])
    elif opt.model_type.startswith('ContactGNN'):
        pass
    elif opt.model_type == 'SequenceFCAutoencoder':
        opt_list.extend([opt.hidden_sizes_list, opt.parameter_sharing])
    else:
        raise Exception("Unknown model type: {}".format(opt.model_type))

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
    opt_list = ['model_type', 'id',  'data_folder', 'y_preprocessing',
        'y_norm', 'min_subtraction', 'y_log_transform', 'crop', 'split', 'shuffle',
        'batch_size', 'num_workers', 'n_epochs', 'lr', 'gpus', 'milestones',
        'gamma', 'loss', 'pretrained', 'resume_training', 'ifile_folder', 'ifile', 'k', 'm',
        'seed', 'act', 'inner_act', 'head_act', 'out_act',
        'training_norm', 'relabel_11_to_00']
    if GNN_mode:
        opt_list.extend(['use_node_features','use_edge_weights', 'transforms', 'pre_transforms', 'split_neg_pos_edges_for_feature_augmentation','sparsify_threshold', 'sparsify_threshold_upper', 'top_k',
                        'hidden_sizes_list', 'message_passing', 'head_architecture', 'head_hidden_sizes_list'])
    if model_type == 'simpleEpiNet':
        opt_list.extend(['kernel_w_list', 'hidden_sizes_list'])
    elif model_type == 'UNet':
        opt_list.extend(['nf','toxx', 'toxx_mode'])
    elif model_type == 'Akita':
        opt_list.extend(['kernel_w_list', 'hidden_sizes_list', 'dilation_list_trunk', 'bottleneck', 'dilation_list_head', 'down_sampling'])
    elif model_type == 'DeepC':
        opt_list.extend(['kernel_w_list', 'hidden_sizes_list', 'dilation_list'])
    elif model_type == 'test':
        opt_list.extend(['kernel_w_list', 'hidden_sizes_list', 'dilation_list_trunk', 'bottleneck', 'dilation_list_head', 'nf'])
    elif model_type == 'GNNAutoencoder':
        opt_list.extend(['head_act', 'head_hidden_sizes_list'])
    elif model_type.startswith('ContactGNN'):
        pass
    elif model_type == 'SequenceFCAutoencoder':
        opt_list.extend(['hidden_sizes_list', 'parameter_sharing'])
    else:
        raise Exception("Unknown model type: {}".format(model_type))

    return opt_list

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
        else:
            raise argparse.ArgumentTypeError('none or int expected not {}'.format(v))
    else:
        raise argparse.ArgumentTypeError('String value expected.')

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
        elif v.replace('.', '').isnumeric():
            return float(v)
        else:
            raise argparse.ArgumentTypeError('none or float expected not {}'.format(v))
    else:
        raise argparse.ArgumentTypeError('String value expected.')

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

def str2list(v, sep = '-'):
    """
    Helper function for argparser, converts str to list by splitting on sep.

    Exmaple for sep = '-': "i-j-k" -> [i,j,k]

    Inputs:
        v: string
        sep: separator
    """
    if v is None:
        return None
    elif isinstance(v, str):
        if v.lower() == 'none':
            return None
        else:
            result = [i for i in v.split(sep)]
            for i, val in enumerate(result):
                if val.isnumeric():
                    result[i] = int(val)
            return result
    else:
        raise argparse.ArgumentTypeError('str value expected.')

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

def test():
    s = '0.176'
    f = str2float(s)
    print(f)

if __name__ == '__main__':
    test()
