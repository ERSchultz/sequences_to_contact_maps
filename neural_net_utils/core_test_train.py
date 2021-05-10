import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from neural_net_utils.utils import getDataLoaders, comparePCA
from neural_net_utils.plotting_functions import plotModelFromArrays, plotDistanceStratifiedPearsonCorrelation
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
            scheduler = scheduler, save_mod = opt.save_mod, print_mod = opt.print_mod)

    print('Total time: {}'.format(time.time() - t0))
    print('Final val loss: {}'.format(val_loss_arr[-1]))

    imageSubPath = os.path.join('images', opt.ofile)
    if not os.path.exists(imageSubPath):
        os.mkdir(imageSubPath, mode = 0o755)

    imagePath = os.path.join(imageSubPath, 'train_val_loss.png')
    plotModelFromArrays(train_loss_arr, val_loss_arr, imagePath, opt)

    # get new val_dataloader
    dataset = Sequences2Contacts(opt.data_folder, opt.toxx, opt.y_preprocessing,
                                        opt.y_norm, opt.x_reshape, opt.ydtype,
                                        opt.y_reshape, opt.crop)
                                         # TODO make this unnecessary
    opt.batchsize = 1
    _, val_dataloader, _ = getDataLoaders(dataset, opt)

    comparePCA(val_dataloader, model, opt)

    imagePath = os.path.join(imageSubPath, 'distance_pearson.png')
    plotDistanceStratifiedPearsonCorrelation(val_dataloader, model, imagePath, opt)
    print()

    imagePath = os.path.join(imageSubPath, 'per_class_acc.png')
    plotPerClassAccuracy(val_dataloader, opt, imagePath)
    print()

    plotPredictions(val_dataloader, model, opt)
    print('\n'*3)

def train(train_loader, val_dataloader, model, optimizer, criterion, device, save_location,
        n_epochs, start_epoch, use_parallel, scheduler, save_mod, print_mod):
    train_loss = []
    val_loss = []
    for e in range(start_epoch, n_epochs+1):
        model.train()
        avg_loss = 0
        for t, (x,y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            yhat = model(x)
            loss = criterion(yhat, y)
            avg_loss += loss.item()
            loss.backward()
            optimizer.step()
        avg_loss /= (t+1)
        train_loss.append(avg_loss)

        val_loss.append(test(val_dataloader, model, optimizer, criterion, device))

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
