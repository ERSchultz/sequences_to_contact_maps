import torch
import torch.nn as nn

def train(train_loader, model, optimizer, criterion, device, save_location,
        n_epochs, start_epoch, use_parallel, scheduler, save_mod, print_mod, ):
    train_loss = []
    for e in range(start_epoch, n_epochs+1):
        model.train()
        avg_loss = 0
        for t, (x,y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            yhat = model(x)
            print(yhat.shape, yhat.dtype)
            print(y.shape, y.dtype)
            loss = criterion(yhat, y)
            avg_loss += loss.item()
            loss.backward()
            optimizer.step()
        avg_loss /= (t+1)
        train_loss.append(avg_loss)

        if scheduler is not None:
            scheduler.step()
        if e % print_mod == 0:
            print('Epoch {}, loss = {:.4f}'.format(e, avg_loss))
        if e % save_mod == 0:
            if use_parallel:
                model_state = model.module.state_dict()
            else:
                model_state = model.state_dict()
            if scheduler is not None:
                torch.save({'model_state_dict': model_state,
                            'epoch': e,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'train_loss': train_loss},
                            save_location)
            else:
                torch.save({'model_state_dict': model_state,
                            'epoch': e,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': None,
                            'train_loss': train_loss},
                            save_location)

    return train_loss

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
