import torch
import torch.nn as nn
import torch.nn.functional as F
from neural_net_utils.networks import UNet
from neural_net_utils.dataset_classes import Sequences2Contacts
from neural_net_utils.utils import getBaseParser, finalizeParser
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
    parser.add_argument('--loss', type=str, default='mse', help='Type of loss to use: options: {"mse", "cross_entropy"}')

    return finalizeParser(parser)

def main():
    opt = argparseSetup()
    print(opt)

    opt.loss = 'mse'
    opt.y_norm = 'diag'
    if opt.loss == 'cross_entropy':
        assert opt.y_norm == 'prcnt', 'must use percentile normalization with cross entropy'
        opt.y_reshape = False
        opt.criterion = F.cross_entropy
        opt.ydtype = torch.int64
        model = UNet(nf_in = 2, nf_out = 10, nf = opt.nf, out_act = None)
    else:
        opt.criterion = F.mse_loss
        model = UNet(nf_in = 2, nf_out = 1, nf = opt.nf, out_act = None)
        opt.y_reshape = True
        opt.ydtype = torch.float32


    seq2ContactData = Sequences2Contacts(opt.data_folder, toxx = True,
                                        y_norm = opt.y_norm, ydtype = opt.ydtype,
                                        y_reshape = opt.y_reshape, crop = opt.crop)

    core_test_train(seq2ContactData, model, opt)


if __name__ == '__main__':
    main()
