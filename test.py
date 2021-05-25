import numpy as np
import torch
import torch.nn as nn
import time
import csv
from neural_net_utils.base_networks import *
from neural_net_utils.networks import *
from neural_net_utils.utils import *
from neural_net_utils.dataset_classes import *

def test_num_workers():
    opt = argparseSetup() # get default args
    opt.data_folder = "/../../../project2/depablo/erschultz/dataset_04_18_21"
    opt.cuda = True
    opt.device = torch.device('cuda')
    opt.y_preprocessing = 'diag'
    opt.y_norm = 'batch'
    dataset = Sequences2Contacts(opt.data_folder, opt.toxx, opt.y_preprocessing,
                                        opt.y_norm, opt.x_reshape, opt.ydtype,
                                        opt.y_reshape, opt.crop)

    b_arr = np.array([1, 2, 4, 8, 16, 32])
    w_arr = np.array([0,1,2,3,4,5,6,7,8])
    results = np.zeros((len(b_arr), len(w_arr)))
    for i, b in enumerate(b_arr):
        print(b)
        for j, w in enumerate(w_arr):
            t0 = time.time()
            opt.batch_size = int(b)
            opt.num_workers = w
            _, val_dataloader, _ = getDataLoaders(dataset, opt)
            for x, y in val_dataloader:
                x = x.to(opt.device)
                y = y.to(opt.device)
            results[i, j] = time.time() - t0

    print(np.round(results, 1))


def main():
    # test_num_workers()
    print('0.1'.isnumeric())
    # raise Exception('test')
    L = 5
    M = 3
    N = 12
    empty_list = [[[None]*N]*M]*L
    print(empty_list)
    print(empty_list[2][2 ][4])






if __name__ == '__main__':
    main()
