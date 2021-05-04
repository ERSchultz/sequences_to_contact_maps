import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from neural_net_utils.networks import SimpleEpiNet
from neural_net_utils.dataset_classes import Sequences2Contacts
from neural_net_utils.utils import getBaseParser, configureOptForCuda
from neural_net_utils.core_test_train import core_test_train
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import argparse

def argparseSetup():
    parser = getBaseParser()

    # dataloader args
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of threads for data loader to use')

    # model args
    parser.add_argument('--height', type=int, default=1, help='Height of convolutional kernel')
    parser.add_argument('--hidden', type=int, default=4, help='Number of variables in hidden layers')
    parser.add_argument('--latent', type=int, default=2, help='Final number of latent variables')

    opt = parser.parse_args()
    return configureOptForCuda(opt)

def main():
    opt = argparseSetup()
    print(opt)

    seq2ContactData = Sequences2Contacts(opt.data_folder, toxx = False,
                                        y_norm = opt.y_norm, crop = opt.crop)


    # Set up model
    model = SimpleEpiNet(n = opt.n, h = opt.height, k = opt.k,
                        hidden_size = opt.hidden, latent_size = opt.latent)
    criterion = F.mse_loss

    core_test_train(seq2ContactData, model, criterion, opt)


if __name__ == '__main__':
    main()
