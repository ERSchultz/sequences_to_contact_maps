import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.data
import os
import sys
import time
from neural_net_utils.networks import *
from neural_net_utils.utils import getDataLoaders, comparePCA, save_opt, save_args, argparseSetup, getModel
from plotting_functions import plotting_script, plotModelFromArrays
from neural_net_utils.dataset_classes import *

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
    if opt.mode == 'GNN':
        dataset = ContactsGraph(opt.data_folder, opt.y_preprocessing,
                                            opt.y_norm, opt.min_subtraction)
    else:
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


    t0 = time.time()
    train_loss_arr, val_loss_arr = train(train_dataloader, val_dataloader, model, opt, ofile = opt.log_file)

    tot_pars = 0
    for k,p in model.named_parameters():
        if opt.verbose:
            print(k, p.numel())
        tot_pars += p.numel()
    print('Total parameters: {}'.format(tot_pars), file = opt.log_file)
    print('Total time: {}'.format(time.time() - t0), file = opt.log_file)
    print('Final val loss: {}\n'.format(val_loss_arr[-1]), file = opt.log_file)

    plotting_script(model, opt, train_loss_arr, val_loss_arr)

    opt.log_file.close()

def train(train_loader, val_dataloader, model, opt, ofile = sys.stdout):
    train_loss = []
    val_loss = []
    for e in range(opt.start_epoch, opt.n_epochs+1):
        if opt.verbose:
            print('Epoch:', e)
        model.train()
        avg_loss = 0
        for t, data in enumerate(train_loader):
            data = data.to(opt.device)
            if opt.verbose:
                print('Iteration: ', t)
            opt.optimizer.zero_grad()

            if opt.mode == 'GNN':
                y = torch_geometric.utils.to_dense_adj(data.edge_index, edge_attr = data.edge_attr,
                                                        batch = data.batch,
                                                        max_num_nodes = opt.n)
                yhat = model(data)
            else:
                x, y = data
                yhat = model(x)
            if opt.verbose:
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
    model.eval()
    avg_loss = 0
    with torch.no_grad():
        for t, data in enumerate(loader):
            data = data.to(opt.device)
            opt.optimizer.zero_grad()
            if opt.mode == 'GNN':
                y = torch_geometric.utils.to_dense_adj(data.edge_index, edge_attr = data.edge_attr,
                                                        batch = data.batch,
                                                        max_num_nodes = opt.n)
                yhat = model(data)
            else:
                x, y = data
                yhat = model(x)
            if opt.verbose:
                print('y', y, y.shape)
                print('yhat', yhat, yhat.shape)
            loss = opt.criterion(yhat, y)
            avg_loss += loss.item()
    avg_loss /= (t+1)
    if toprint:
        print('Mean test/val loss: {:.4f}\n'.format(avg_loss), file = ofile)
    # TODO quartiles and median loss
    return avg_loss


if __name__ == '__main__':
    main()
