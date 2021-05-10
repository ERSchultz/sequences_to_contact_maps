import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from neural_net_utils.networks import SimpleEpiNet
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

    # Set up model
    model = SimpleEpiNet(opt.n, opt.k, opt.kernel_w_list, opt.hidden_sizes_list)
    if opt.loss == 'mse':
        opt.criterion = F.mse_loss
    else:
        print('Invalid loss: {}'.format(opt.loss))

    core_test_train(model, opt)


if __name__ == '__main__':
    main()
