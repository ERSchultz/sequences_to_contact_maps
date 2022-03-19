import argparse
import os.path as osp
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



def clean_directories(data_folder = 'dataset_04_18_21', GNN_path = None, GNN_file_name = None):
    parser = argparse.ArgumentParser(description='Simple parser')
    parser.add_argument('--data_folder', type=str, default=data_folder,
                        help='Location of data')
    parser.add_argument('--GNN_file_name', type=str, default=str(GNN_file_name),
                        help='name of file graph data was saved to')
    parser.add_argument('--GNN_path', type=str, default=GNN_path,
                        help='path to graph data')
    parser.add_argument('--clean_scratch', type=str2bool, default=False,
                        help='True clean scratch')
    parser.add_argument('--scratch', type=str, default=None)
    opt = parser.parse_args()

    if opt.scratch is not None:
        opt.data_folder = osp.join(opt.scratch, osp.split(opt.data_folder)[-1])

    if opt.clean_scratch and opt.scratch:
        rmtree(opt.data_folder)
    else:
        if opt.GNN_path is None:
            opt.GNN_path = osp.join(opt.data_folder, opt.GNN_file_name)

        if osp.exists(opt.GNN_path):
            print('Removing {}'.format(opt.GNN_path))
            rmtree(opt.GNN_path)
        else:
            print('{} does not exist - cannot remove'.format(opt.GNN_path))


if __name__ == '__main__':
    clean_directories()
