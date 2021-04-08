from torch.utils.data import Dataset, DataLoader
from neural_net_utils.utils import *
import torch

class Sequences2Contacts(Dataset):
    def __init__(self, dirname, n, k, toxx, y_diag_norm = True):
        super(Sequences2Contacts, self).__init__()
        self.n = n
        self.k = k
        self.toxx = toxx
        self.y_diag_norm = y_diag_norm
        self.paths = sorted(make_dataset(dirname))

    def __getitem__(self, index):
        y_path = self.paths[index] + '/y.npy'
        y = np.load(y_path)
        if self.y_diag_norm:
            y = diagonal_normalize(y.astype(np.float64))
        y = y.reshape(1, self.n, self.n)
        y = y / np.max(y)

        if self.toxx:
            x_path = self.paths[index] + '/xx.npy'
            x = np.load(x_path)
        else:
            x_path = self.paths[index] + '/x.npy'
            x = np.load(x_path)
            x = x.reshape(1, self.n, self.k)

        return torch.Tensor(x), torch.Tensor(y)

    def __len__(self):
        return len(self.paths)

class Contacts(Dataset):
    def __init__(self, dirname, n, y_diag_norm = True):
        super(Contacts, self).__init__()
        self.n = n
        self.y_diag_norm = y_diag_norm
        self.paths = sorted(make_dataset(dirname))

    def __getitem__(self, index):
        y_path = self.paths[index] + '/y.npy'
        y = np.load(y_path)
        if self.y_diag_norm:
            y = diagonal_normalize(y.astype(np.float64))
        y = y.reshape(1, self.n, self.n)
        y = y / np.max(y)
        return y

    def __len__(self):
        return len(self.paths)
