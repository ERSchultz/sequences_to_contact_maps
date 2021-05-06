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

    seq2ContactData = Sequences2Contacts(opt.data_folder, toxx = opt.toxx, y_preprocessing = opt.y_preprocessing,
                                        y_norm = opt.y_norm, x_reshape = opt.x_reshape,
                                        crop = opt.crop)


    # Set up model
    model = SimpleEpiNet(opt.n, opt.k, opt.kernel_w_list, opt.hidden_sizes_list)
    if opt.loss == 'mse':
        opt.criterion = F.mse_loss
    else:
        print('Invalid loss: {}'.format(opt.loss))

    core_test_train(seq2ContactData, model, opt)


if __name__ == '__main__':
    main()
