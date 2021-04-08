import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision.transforms as transforms
from neural_net_utils.dataset_classes import *
from neural_net_utils.networks import *
from neural_net_utils.utils import getDataLoaders
import numpy as np
import matplotlib.pyplot as plt
import time

def trainVAE(train_loader, model, optimizer, device, save_location,
        epochs, save_mod = 5, print_mod = 2):
    train_loss = []
    for e in range(epochs):
        model.train()
        avg_loss = 0
        for t, y in enumerate(train_loader):
            y = y.to(device)
            optimizer.zero_grad()
            loss = model.loss(y)
            avg_loss += loss.item()
            loss.backward()
            optimizer.step()
        avg_loss = avg_loss/(t+1)
        if e % print_mod == 0:
            print('Epoch %d, avg_loss = %.4f' % (e, avg_loss))
        if e % save_mod == 0:
            torch.save({'model_state_dict': model.state_dict(),
                        'epoch': e,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_loss},
                        save_location)
        train_loss.append(avg_loss)

    return train_loss


def main(dir, epochs = 1000, device = 'cuda:0'):
    t0 = time.time()
    contactData = Contacts(dir)
    for i in contactData:
        print(i)
    train_dataloader = DataLoader(contactData, batch_size = 64,
                                    shuffle = True, num_workers = 0)

    if device == 'cuda:0' and not torch.cuda.is_available():
        print('Warning: falling back to cpu')
        device = 'cpu'

    model = VAE(1024, 200, 25)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr = 1e-3) # default beta TODO

    train_loss = trainVAE(train_dataloader, model, optimizer,
            device, save_location = 'models/VAE_model1.pt', epochs = epochs)

    print('Total time: {}'.format(time.time() - t0))

    plt.plot(np.arange(0, epochs), train_loss, label = 'train_loss')
    plt.legend()
    plt.savefig('images/train_loss.png')
    plt.close()

if __name__ == '__main__':
    clusterdir = '../../../project2/depablo/skyhl/dataset_04_06_21'
    mydir = 'dataset_04_06_21'
    main(mydir, 10)
