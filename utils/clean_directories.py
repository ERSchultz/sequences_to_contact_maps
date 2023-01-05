import argparse
import os.path as osp
import sys
from shutil import rmtree


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
            return result
    else:
        raise argparse.ArgumentTypeError('str value expected.')



def clean_directories(data_folder = 'dataset_04_18_21', GNN_path = None,
                    GNN_file_name = None, ofile = sys.stdout):
    parser = argparse.ArgumentParser(description='Simple parser')
    parser.add_argument('--data_folder', type=str2list, default=data_folder,
                        help='Location of data')
    parser.add_argument('--GNN_file_name', type=str, default=str(GNN_file_name),
                        help='name of file graph data was saved to')
    parser.add_argument('--GNN_path', type=str, default=GNN_path,
                        help='path to graph data')
    parser.add_argument('--clean_scratch', action='store_true',
                        help='True clean scratch')
    parser.add_argument('--scratch', type=str, default=None)
    parser.add_argument('--use_scratch', type=str2bool, default=True)
    opt, _ = parser.parse_known_args()

    if isinstance(opt.data_folder, list):
        opt.data_folder = opt.data_folder[0]

    if opt.use_scratch and opt.scratch is not None:
        opt.data_folder = osp.join(opt.scratch, osp.split(opt.data_folder)[-1])

    if opt.clean_scratch:
        rmtree(opt.data_folder)
    else:
        if opt.GNN_path is None and opt.GNN_file_name is not None:
            opt.GNN_path = osp.join(opt.data_folder, opt.GNN_file_name)

        if osp.exists(opt.GNN_path):
            print(f'Removing {opt.GNN_path}', file = ofile)
            rmtree(opt.GNN_path)
        else:
            print(f'{opt.GNN_path} does not exist - cannot remove', file = ofile)


if __name__ == '__main__':
    clean_directories()
