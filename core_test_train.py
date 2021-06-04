import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import time
from neural_net_utils.networks import *
from neural_net_utils.utils import getDataLoaders, comparePCA, save_opt, save_args, argparseSetup, getModel
from plotting_functions import plotting_script, plotModelFromArrays
from neural_net_utils.dataset_classes import Sequences2Contacts

def main():
    opt = argparseSetup()
    model = getModel(opt)

    core_test_train(model, opt)

def core_test_train(model, opt):
    print(opt, end = '\n\n', file = opt.log_file)
    print(opt)
    print(opt.ofile_folder, end = '\n\n')
    save_opt(opt, os.path.join('results', opt.model_type, 'experiments.csv'))
    save_args(opt)

    # split dataset
    dataset = Sequences2Contacts(opt.data_folder, opt.toxx, opt.toxx_mode, opt.y_preprocessing,
                                        opt.y_norm, opt.x_reshape, opt.ydtype,
                                        opt.y_reshape, opt.crop, opt.min_subtraction)
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
    optimizer = optim.Adam(model.parameters(), lr = opt.lr)
    if opt.milestones is not None:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = opt.milestones,
                                                    gamma = opt.gamma, verbose = True)
    else:
        scheduler = None

    if opt.use_parallel:
        model = torch.nn.DataParallel(model, device_ids = opt.gpu_ids)

    if opt.cuda:
        model.to(opt.device)


    t0 = time.time()
    train_loss_arr, val_loss_arr = train(train_dataloader, val_dataloader, model, optimizer,
            opt.criterion, device = opt.device, save_location = os.path.join(opt.ofile_folder, 'model.pt'),
            n_epochs = opt.n_epochs, start_epoch = opt.start_epoch, use_parallel = opt.use_parallel,
            scheduler = scheduler, save_mod = opt.save_mod, print_mod = opt.print_mod, verbose = opt.verbose,
            ofile = opt.log_file)

    tot_pars = 0
    for k,p in model.named_parameters():
        tot_pars += p.numel()
    print('Total parameters: {}'.format(tot_pars), file = opt.log_file)
    print('Total time: {}'.format(time.time() - t0), file = opt.log_file)
    print('Final val loss: {}\n'.format(val_loss_arr[-1]), file = opt.log_file)

    plotting_script(model, opt, train_loss_arr, val_loss_arr)

    opt.log_file.close()

def train(train_loader, val_dataloader, model, optimizer, criterion, device, save_location,
        n_epochs, start_epoch, use_parallel, scheduler, save_mod, print_mod, verbose, ofile = sys.stdout):
    train_loss = []
    val_loss = []
    to_device_time = 0
    test_time = 0
    forward_time = 0
    backward_time = 0
    for e in range(start_epoch, n_epochs+1):
        if verbose:
            print('Epoch:', e)
        model.train()
        avg_loss = 0
        for t, (x,y) in enumerate(train_loader):
            if verbose:
                print('Iteration: ', t)
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()

            yhat = model(x)
            if verbose:
                print('y', y)
                print('yhat', yhat)
            loss = criterion(yhat, y)
            avg_loss += loss.item()
            loss.backward()
            optimizer.step()
        avg_loss /= (t+1)
        train_loss.append(avg_loss)

        if scheduler is not None:
            scheduler.step()
        if e % print_mod == 0 or e == n_epochs:
            print('Epoch {}, loss = {:.4f}'.format(e, avg_loss), file = ofile)
            print_val_loss = True
        else:
            print_val_loss = False
        val_loss.append(test(val_dataloader, model, optimizer, criterion, device, print_val_loss, ofile))

        if e % save_mod == 0:
            if use_parallel:
                model_state = model.module.state_dict()
            else:
                model_state = model.state_dict()
            save_dict = {'model_state_dict': model_state,
                        'epoch': e,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': None,
                        'train_loss': train_loss,
                        'val_loss': val_loss}
            if scheduler is not None:
                save_dict['scheduler_state_dict'] = scheduler.state_dict()
            torch.save(save_dict, save_location)

    print('\nto_device_time: ', to_device_time, file = ofile)
    print('test_time: ', test_time, file = ofile)
    print('forward_time: ', forward_time, file = ofile)
    print('backward_time: {}\n'.format(backward_time), file = ofile)

    return train_loss, val_loss

def test(loader, model, optimizer, criterion, device, toprint, ofile = sys.stdout):
    model.eval()
    avg_loss = 0
    with torch.no_grad():
        for t, (x,y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            yhat = model(x)
            loss = criterion(yhat, y)
            avg_loss += loss.item()
    avg_loss /= (t+1)
    if toprint:
        print('Mean test/val loss: {:.4f}'.format(avg_loss), file = ofile)
    # TODO quartiles and median loss
    return avg_loss


if __name__ == '__main__':
    main()
