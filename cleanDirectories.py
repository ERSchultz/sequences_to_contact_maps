import os.path as osp
from shutil import rmtree

from neural_net_utils.argparseSetup import str2bool
import argparse

def main():
    parser = argparse.ArgumentParser(description='Simple parser')
    parser.add_argument('--data_folder', type=str, default='dataset_04_18_21', help='Location of data')
    parser.add_argument('--root_name', type=str, help='name of file graph data was saved to')
    parser.add_argument('--use_scratch', type=str2bool, default=False, help='True if data was moved to scratch')
    opt = parser.parse_args()

    if opt.use_scratch:
        opt.data_folder = osp.join('/scratch/midway2/erschultz', osp.split(opt.data_folder)[-1])

    root = osp.join(opt.data_folder, opt.root_name)

    if osp.exists(root):
        print('Removing {}'.format(root))
        rmtree(root)
    else:
        print('{} does not exist'.format(root))


if __name__ == '__main__':
    main()
