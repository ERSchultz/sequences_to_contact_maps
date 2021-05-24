import torch
import torch.nn as nn
import torch.nn.functional as F
from neural_net_utils.networks import UNet
from neural_net_utils.dataset_classes import Sequences2Contacts
from neural_net_utils.utils import argparseSetup, str2list
from neural_net_utils.core_test_train import core_test_train
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import argparse

def main():
    opt = argparseSetup()
    opt.mode = 'debugging'
    if opt.mode == 'debugging':
        opt.mode = 'real'
        # Preprocessing
        opt.toxx = True
        opt.toxx_mode = 'mean'
        opt.x_reshape = False

        # architecture
        opt.k=2

        opt.n=1024
        opt.y_preprocessing='diag'
        opt.y_norm='instance'
        opt.nf = 8

        # hyperparameters
        opt.n_epochs=1
        opt.lr=0.1
        opt.batch_size=4
        opt.milestones=str2list('1')
        opt.gamma=0.1

        # other
        opt.verbose = False
        opt.plot = False
        opt.plot_predictions = False
        opt.data_folder = 'dataset_04_18_21'
        opt.save_mod = 1
    print(opt)


    if opt.loss == 'cross_entropy':
        assert opt.y_preprocessing == 'prcnt', 'must use percentile preprocessing with cross entropy'
        assert opt.y_norm == None, 'Cannot normalize with cross entropy'
        opt.y_reshape = False
        opt.criterion = F.cross_entropy
        opt.ydtype = torch.int64
        opt.out_act = None
        model = UNet(nf_in = 2, nf_out = 10, nf = opt.nf, out_act = None) # activation combined into loss
    elif opt.loss == 'mse':
        opt.criterion = F.mse_loss
        opt.out_act = 'sigmoid'
        model = UNet(nf_in = 2, nf_out = 1, nf = opt.nf, out_act = nn.Sigmoid())
        opt.y_reshape = True
        opt.ydtype = torch.float32
    else:
        print('Invalid loss: {}'.format(opt.loss))
    core_test_train(model, opt)


if __name__ == '__main__':
    main()
