import numpy as np
import torch
import torch.nn as nn
import time
import csv
import os
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

def cleanup():
    dir = "/../../../project2/depablo/erschultz/dataset_04_18_21/samples"
    for sample in os.listdir(dir):
        sample_path = os.path.join(dir, sample)
        if os.path.isdir(sample_path):
            for file in os.listdir(sample_path):
                f_path = os.path.join(sample_path, file)
                if os.path.isdir(f_path):
                    for file2 in os.listdir(f_path):
                        f2_path = os.path.join(f_path, file2)
                        os.remove(f2_path)
                        print('Delete', f2_path)
                    os.rmdir(f_path)
                    print('Delete', f_path)
                else:
                    pass
        else:
            pass

def makeargeparefiles():
    with open('results\\UNet\\experiments.csv', newline = '') as f:
        reader = csv.reader(f)
        for line in f:
            print(line.strip().split(','))



def main():
    makeargeparefiles()






if __name__ == '__main__':
    main()
