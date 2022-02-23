import locale
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

locale.setlocale(locale.LC_ALL, '')

from .argparseSetup import argparseSetup, save_args
from .cleanDirectories import cleanDirectories
from .dataset_classes import *
from .networks import *
from .plotting_functions import plotModelFromArrays, plotting_script
from .utils import comparePCA, getDataLoaders, getDataset, getModel


def main():
    opt = argparseSetup()
    model = getModel(opt)

    core_test_train(model, opt)

def core_test_train(model, opt):
    print(opt, end = '\n\n', file = opt.log_file)
    print(opt)
    print(opt.ofile_folder, end = '\n\n')
    save_args(opt)

    # split dataset
    dataset = getDataset(opt)
    train_dataloader, val_dataloader, test_dataloader = getDataLoaders(dataset, opt)

    if opt.pretrained:
        model_name = os.path.join(opt.ifile_folder, opt.ifile + '.pt')
        if os.path.exists(model_name):
            if opt.cuda:
                save_dict = torch.load(model_name)
            else:
                save_dict = torch.load(model_name, map_location = 'cpu')
            model.load_state_dict(save_dict['model_state_dict'])
            print('Pre-trained model is loaded.', file = opt.log_file)

    # Set up model and scheduler
    opt.optimizer = optim.Adam(model.parameters(), lr = opt.lr)
    if opt.milestones is not None:
        opt.scheduler = optim.lr_scheduler.MultiStepLR(opt.optimizer, milestones = opt.milestones,
                                                    gamma = opt.gamma, verbose = opt.verbose)
    else:
        opt.scheduler = None

    if opt.use_parallel:
        model = torch.nn.DataParallel(model, device_ids = opt.gpu_ids)

    if opt.cuda:
        model.to(opt.device)

    if opt.print_params:
        print('#### INITIAL PARAMETERS ####', file = opt.param_file)
        for k,p in model.named_parameters():
            print(k, p.shape, file = opt.param_file)
            print(p, '\n', file = opt.param_file)
        print('\n', file = opt.param_file)

    t0 = time.time()
    print("#### TRAINING/VALIDATION ####", file = opt.log_file)
    train_loss_arr, val_loss_arr = train(train_dataloader, val_dataloader, model, opt, ofile = opt.log_file)

    tot_pars = 0
    if opt.print_params:
        print('#### FINAL PARAMETERS ####', file = opt.param_file)
    for k,p in model.named_parameters():
        tot_pars += p.numel()
        if opt.verbose:
            print(k, p.numel(), p.shape, file = opt.log_file)
            print(p, '\ngrad: ', p.grad, '\n', file = opt.log_file)
        if opt.print_params:
            print(k, p.numel(), p.shape, file = opt.param_file)
            print(p, '\ngrad: ', p.grad, '\n', file = opt.param_file)
    print('\nTotal parameters: {}'.format(locale.format_string("%d", tot_pars, grouping = True)), file = opt.log_file)
    tot_time = (time.time() - t0) / 60 / 60 # hours
    print('Total training + validation time: {} hours'.format(np.round(tot_time), 2), file = opt.log_file)
    print('Final val loss: {}\n'.format(val_loss_arr[-1]), file = opt.log_file)

    if opt.GNN_mode:
        plotting_script(model, opt, train_loss_arr, val_loss_arr, dataset)
    else:
        plotting_script(model, opt, train_loss_arr, val_loss_arr)
        # don't pass dataset

    # cleanup
    opt.log_file.close()
    if opt.root is not None and opt.delete_root:
        # opt.root is set in utils.getDataset
        cleanDirectories(root = opt.root)

def train(train_loader, val_dataloader, model, opt, ofile = sys.stdout):
    train_loss = []
    val_loss = []
    for e in range(opt.start_epoch, opt.n_epochs+1):
        if opt.verbose:
            print('Epoch:', e)
        model.train()
        avg_loss = 0
        for t, data in enumerate(train_loader):
            if opt.verbose:
                print('Iteration: ', t)
            opt.optimizer.zero_grad()

            if opt.GNN_mode:
                data = data.to(opt.device)
                if opt.autoencoder_mode:
                    y = data.contact_map
                    y = torch.reshape(y, (-1, opt.m, opt.m))
                elif opt.output_mode == 'energy':
                    y = data.energy
                    y = torch.reshape(y, (-1, opt.m, opt.m))
                else:
                    y = data.y
                yhat = model(data)
            elif opt.autoencoder_mode and opt.output_mode == 'sequence':
                x = data[0]
                x = x.to(opt.device)
                y = x
                yhat = model(x)
            else:
                x, y = data
                x = x.to(opt.device)
                y = y.to(opt.device)
                yhat = model(x)
            if opt.verbose:
                if 'x' in locals():
                    print('x', x, x.shape)
                if opt.GNN_mode:
                    print('x', data.x, data.x.shape)
                print('y', y, y.shape)
                print('yhat', yhat, yhat.shape)
            loss = opt.criterion(yhat, y)
            avg_loss += loss.item()
            loss.backward()
            opt.optimizer.step()
        avg_loss /= (t+1)
        train_loss.append(avg_loss)

        if opt.scheduler is not None:
            opt.scheduler.step()
        if e % opt.print_mod == 0 or e == opt.n_epochs:
            print('Epoch {}, loss = {:.4f}'.format(e, avg_loss), file = ofile)
            print_val_loss = True
        else:
            print_val_loss = False
        val_loss.append(test(val_dataloader, model, opt, print_val_loss, ofile))

        if e % opt.save_mod == 0:
            if opt.use_parallel:
                model_state = model.module.state_dict()
            else:
                model_state = model.state_dict()
            save_dict = {'model_state_dict': model_state,
                        'epoch': e,
                        'optimizer_state_dict': opt.optimizer.state_dict(),
                        'scheduler_state_dict': None,
                        'train_loss': train_loss,
                        'val_loss': val_loss}
            if opt.scheduler is not None:
                save_dict['scheduler_state_dict'] = opt.scheduler.state_dict()
            torch.save(save_dict, os.path.join(opt.ofile_folder, 'model.pt'))

    return train_loss, val_loss

def test(loader, model, opt, toprint, ofile = sys.stdout):
    assert loader is not None, 'loader is None - check train/val/test split'
    model.eval()
    avg_loss = 0
    with torch.no_grad():
        for t, data in enumerate(loader):
            if opt.GNN_mode:
                data = data.to(opt.device)
                if opt.autoencoder_mode:
                    y = data.contact_map
                    y = torch.reshape(y, (-1, opt.m, opt.m))
                elif opt.output_mode == 'energy':
                    y = data.energy
                    y = torch.reshape(y, (-1, opt.m, opt.m))
                else:
                    y = data.y
                yhat = model(data)
            elif opt.autoencoder_mode and opt.output_mode == 'sequence':
                x = data[0]
                x = x.to(opt.device)
                y = x
                yhat = model(x)
            else:
                x, y = data
                x = x.to(opt.device)
                y = y.to(opt.device)
                yhat = model(x)
            loss = opt.criterion(yhat, y)
            avg_loss += loss.item()
    avg_loss /= (t+1)
    if toprint:
        print('Mean test/val loss: {:.4f}\n'.format(avg_loss), file = ofile)
    # TODO quartiles and median loss
    return avg_loss


if __name__ == '__main__':
    main()
