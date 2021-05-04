import torch.nn as nn
import torch.nn.functional as F
from neural_net_utils.networks import UNet
from neural_net_utils.dataset_classes import Sequences2Contacts
from neural_net_utils.utils import getBaseParser
from neural_net_utils.core_test_train import core_test_train
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import argparse

def argparseSetup():
    parser = getBaseParser()

    # dataloader args
    parser.add_argument('--batch_size', type=int, default=8, help='Training batch size')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of threads for data loader to use')

    # model args
    parser.add_argument('--nf', type=int, default=8, help='Number of filters')

    opt = parser.parse_args()

    # add arg for reshape
    if opt.y_norm == 'prcnt':
        opt.y_reshape = False
    else:
        opt.y_reshape = True

    return configureOptForCuda(opt)

def main():
    opt = argparseSetup()
    print(opt)

    seq2ContactData = Sequences2Contacts(opt.data_folder, toxx = True,
                                        y_norm = opt.y_norm,
                                        y_reshape = opt.y_reshape, crop = opt.crop)

    # Set up model
    if opt.y_norm == 'diag':
        model = UNet(nf_in = 2, nf_out = 1, nf = opt.nf, out_act = nn.Sigmoid())
        criterion = F.mse_loss
    elif opt.y_norm == 'prcnt':
        model = UNet(nf_in = 2, nf_out = 10, nf = opt.nf, out_act = None)
        criterion = F.cross_entropy

    core_test_train(seq2ContactData, model, criterion, opt)


if __name__ == '__main__':
    main()
