import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from neural_net_utils.networks import *
from neural_net_utils.dataset_classes import *
from neural_net_utils.utils import getDataLoaders, plotModelFromArrays, getBaseParser
from neural_net_utils.core_test_train import *
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

    if opt.milestones is not None:
        opt.milestons = opt.milestones.split('-')

    # configure cuda
    if opt.gpus > 1:
        opt.cuda = True
        opt.use_parallel = True
        opt.gpu_ids = []
        for ii in range(6):
            try:
                torch.cuda.get_device_properties(ii)
                print(str(ii))
                opt.gpu_ids.append(ii)
            except AssertionError:
                print('Not ' + str(ii) + "!")
    elif opt.gpus == 1:
        opt.cuda = True
        opt.use_parallel = False
    else:
        opt.cuda = False
        opt.use_parallel = False

    if opt.cuda and not torch.cuda.is_available():
        print('Warning: falling back to cpu')
        opt.cuda = False
        opt.use_parallel = False

    opt.device = torch.device('cuda' if opt.cuda else 'cpu')
    return opt

def main():
    opt = argparseSetup()
    t0 = time.time()
    seq2ContactData = Sequences2Contacts(opt.data_folder, n = 1024, k = opt.k, toxx = True)
    train_dataloader, val_dataloader, test_dataloader = getDataLoaders(seq2ContactData,
                                                                        batch_size = opt.batch_size,
                                                                        num_workers = opt.num_workers,
                                                                        seed = opt.seed)

    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)

    # Set up model
    model = UNet(nf_in = 2, nf_out = 1, nf = opt.nf, out_act = nn.Sigmoid())
    if opt.pretrained:
        model_name = os.path.join(opt.ifile_folder, opt.ifile + '.pt')
        if os.path.exists(model_name):
            if opt.cuda:
                save_dict = torch.load(model_name)
            else:
                save_dict = torch.load(model_name, map_location = 'cpu')
            model.load_state_dict(save_dict['model_state_dict'])
            print('Pre-trained model is loaded.')

    optimizer = optim.Adam(model.parameters(), lr = opt.lr)
    if opt.milestones is not None:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = opt.milestones,
                                                    gamma = opt.gamma, verbose = True)
    else:
        scheduler = None
    criterion = F.mse_loss

    if opt.use_parallel:
        model = torch.nn.DataParallel(model, device_ids = opt.gpu_ids)

    if opt.cuda:
        model.to(opt.device)

    train_loss_arr = train(train_dataloader, model, optimizer,
            criterion, device = opt.device, save_location = os.path.join(opt.ofile_folder, opt.ofile + '.pt'),
            n_epochs = opt.n_epochs, start_epoch = opt.start_epoch, use_parallel = opt.use_parallel,
            scheduler = scheduler, save_mod = opt.save_mod, print_mod = opt.print_mod)
    val_loss = test(val_dataloader, model, optimizer, criterion, opt.device)

    print('Total time: {}'.format(time.time() - t0))
    print('Val loss: {}'.format(val_loss))
    plotModelFromArrays(train_loss_arr, os.path.join(opt.ofile_folder, opt.ofile + ' train_val_loss.png'), val_loss)

if __name__ == '__main__':
    main()
