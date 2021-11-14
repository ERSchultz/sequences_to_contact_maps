import os.path as osp
from shutil import rmtree

from neural_net_utils.argparseSetup import str2bool
import argparse

def main(data_folder = 'dataset_04_18_21', root_name = None, root = None):
    parser = argparse.ArgumentParser(description='Simple parser')
    parser.add_argument('--data_folder', type=str, default=data_folder, help='Location of data')
    parser.add_argument('--root_name', type=str, default=str(root_name), help='name of file graph data was saved to')
    parser.add_argument('--root', type=str, default=root, help='name of root')
    parser.add_argument('--use_scratch', type=str2bool, default=False, help='True if data was moved to scratch')
    opt = parser.parse_args()

    if opt.use_scratch:
        opt.data_folder = osp.join('/scratch/midway2/erschultz', osp.split(opt.data_folder)[-1])

    if opt.root is None:
        opt.root = osp.join(opt.data_folder, opt.root_name)

    if osp.exists(opt.root):
        print('Removing {}'.format(opt.root))
        rmtree(opt.root)
    else:
        print('{} does not exist - cannot remove'.format(opt.root))


if __name__ == '__main__':
    main()
