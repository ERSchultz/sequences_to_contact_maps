import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from neural_net_utils.utils import getDataLoaders, comparePCA
from neural_net_utils.plotting_functions import plotting_script, plotModelFromArrays
from neural_net_utils.dataset_classes import Sequences2Contacts

def core_test_train(model, opt):
    # Set random seeds
    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)

    # split dataset
    dataset = Sequences2Contacts(opt.data_folder, opt.toxx, opt.y_preprocessing,
                                        opt.y_norm, opt.x_reshape, opt.ydtype,
                                        opt.y_reshape, opt.crop)
    train_dataloader, val_dataloader, test_dataloader = getDataLoaders(dataset, opt)

    if opt.pretrained:
        model_name = os.path.join(opt.ifile_folder, opt.ifile + '.pt')
        if os.path.exists(model_name):
            if opt.cuda:
                save_dict = torch.load(model_name)
            else:
                save_dict = torch.load(model_name, map_location = 'cpu')
            model.load_state_dict(save_dict['model_state_dict'])
            print('Pre-trained model is loaded.')

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
            opt.criterion, device = opt.device, save_location = os.path.join(opt.ofile_folder, opt.ofile + '.pt'),
            n_epochs = opt.n_epochs, start_epoch = opt.start_epoch, use_parallel = opt.use_parallel,
            scheduler = scheduler, save_mod = opt.save_mod, print_mod = opt.print_mod, verbose = opt.verbose)

    tot_pars = 0
    for k,p in model.named_parameters():
        tot_pars += p.numel()
    print('Total parameters: {}'.format(tot_pars))
    print('Total time: {}'.format(time.time() - t0))
    print('Final val loss: {}\n'.format(val_loss_arr[-1]))

    if opt.plot:
        imageSubPath = os.path.join('images', opt.ofile)
        if not os.path.exists(imageSubPath):
            os.mkdir(imageSubPath, mode = 0o755)

        imagePath = os.path.join(imageSubPath, 'train_val_loss.png')
        plotModelFromArrays(train_loss_arr, val_loss_arr, imagePath, opt)

        plotting_script(model, opt)

def train(train_loader, val_dataloader, model, optimizer, criterion, device, save_location,
        n_epochs, start_epoch, use_parallel, scheduler, save_mod, print_mod, verbose):
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
            if torch.cuda.is_available():
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                x = x.to(device)
                y = y.to(device)
                torch.cuda.synchronize()
                end.record()
                to_device_time += start.elapsed_time(end)
            else:
                t0 = time.time()
                x = x.to(device)
                y = y.to(device)
                to_device_time += time.time() - t0
            optimizer.zero_grad()

            if torch.cuda.is_available():
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                yhat = model(x)
                torch.cuda.synchronize()
                end.record()
                forward_time += start.elapsed_time(end)
            else:
                t0 = time.time()
                yhat = model(x)
                forward_time += time.time() - t0


            if verbose:
                print('y', y)
                print('yhat', yhat)
            loss = criterion(yhat, y)
            avg_loss += loss.item()

            if torch.cuda.is_available():
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                loss.backward()
                torch.cuda.synchronize()
                end.record()
                backward_time += start.elapsed_time(end)
            else:
                t0 = time.time()
                loss.backward()
                backward_time += time.time() - t0

            optimizer.step()
        avg_loss /= (t+1)
        train_loss.append(avg_loss)

        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            val_loss.append(test(val_dataloader, model, optimizer, criterion, device))
            torch.cuda.synchronize()
            end.record()
            test_time += start.elapsed_time(end)
        else:
            t0 = time.time()
            val_loss.append(test(val_dataloader, model, optimizer, criterion, device))
            test_time += time.time() - t0

        if scheduler is not None:
            scheduler.step()
        if e % print_mod == 0 or e == n_epochs:
            print('Epoch {}, loss = {:.4f}'.format(e, avg_loss))
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

    print('to_device_time: ', to_device_time)
    print('test_time: ', test_time)
    print('forward_time: ', forward_time)
    print('backward_time: ', backward_time)

    return train_loss, val_loss

def test(loader, model, optimizer, criterion, device):
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
    print('\nMean test loss: {:.4f}'.format(avg_loss))
    # TODO quartiles and median loss
    return avg_loss
