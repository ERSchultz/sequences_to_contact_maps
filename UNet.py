import torch
import torch.nn as nn
import torch.nn.functional as F
from neural_net_utils.networks import UNet
from neural_net_utils.dataset_classes import Sequences2Contacts
from neural_net_utils.utils import argparseSetup
from neural_net_utils.core_test_train import core_test_train
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import argparse

def main():
    opt = argparseSetup()
    print(opt)

    if opt.loss == 'cross_entropy':
        assert opt.y_preprocessing == 'prcnt', 'must use percentile normalization with cross entropy'
        opt.y_reshape = False
        opt.criterion = F.cross_entropy
        opt.ydtype = torch.int64
        model = UNet(nf_in = 2, nf_out = 10, nf = opt.nf, out_act = None) # activation combined into loss
    elif opt.loss == 'mse':
        opt.criterion = F.mse_loss
        model = UNet(nf_in = 2, nf_out = 1, nf = opt.nf, out_act = nn.Sigmoid())
        opt.y_reshape = True
        opt.ydtype = torch.float32
    else:
        print('Invalid loss: {}'.format(opt.loss))

    core_test_train(model, opt)


if __name__ == '__main__':
    main()
