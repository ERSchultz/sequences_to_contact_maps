import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import csv
import os
from neural_net_utils.base_networks import *
from neural_net_utils.networks import *
from neural_net_utils.utils import *
from neural_net_utils.dataset_classes import *
from core_test_train import core_test_train, getModel

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
    dir = "results"
    for type in os.listdir(dir):
        type_path = os.path.join(dir, type)
        if os.path.isdir(type_path):
            for id in os.listdir(type_path):
                id_path = os.path.join(type_path, id)
                if os.path.isdir(id_path):
                    for file in os.listdir(id_path):
                        f_path = os.path.join(id_path, file)
                        if os.path.isdir(f_path) and file.startswith('sample'):
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

def debugModel(model_type):
    opt = argparseSetup()
    opt.mode = 'debugging'
    opt.model_type = model_type

    # Preprocessing
    if model_type == 'UNet':
        opt.toxx = True
        opt.toxx_mode = 'mean'
        opt.x_reshape = False

    # architecture
    opt.k = 2
    opt.crop = None
    opt.n = 1024
    opt.y_preprocessing = 'diag'
    opt.y_norm = 'batch'
    opt.loss = 'mse'

    if model_type == 'Akita':
        opt.kernel_w_list=str2list('5-5-5')
        opt.hidden_sizes_list=str2list('4-6-8')
        opt.dilation_list_trunk=str2list('2-4-8-16')
        opt.bottleneck=4
        opt.dilation_list_head=str2list('2-4-8-16')
        opt.out_act=nn.ReLU(True)
        opt.training_norm='batch'
        opt.down_sampling='conv'
    elif model_type == 'UNet':
        opt.nf = 8
        opt.out_act = 'sigmoid'
        opt.training_norm = 'batch'
    elif model_type == 'DeepC':
        opt.kernel_w_list=str2list('5-5-5')
        opt.hidden_sizes_list=str2list('32-64-128')
        opt.dilation_list=str2list('2-4-8-16-32-64-128-256-512')



    # hyperparameters
    opt.n_epochs = 1
    opt.lr = 1e-3
    opt.batch_size = 1
    opt.milestones = str2list('1')
    opt.gamma = 0.1

    # other
    opt.plot = True
    opt.plot_predictions = True
    opt.verbose = False
    if opt.cuda:
        opt.data_folder = "/../../../project2/depablo/erschultz/dataset_04_18_21"
    else:
        opt.data_folder = "dataset_04_18_21"

    model = getModel(opt)
    opt.model_type = 'test'
    core_test_train(model, opt)




def main():
    # cleanup()
    debugModel('Akita')



if __name__ == '__main__':
    main()
