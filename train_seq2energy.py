import os
import os.path as osp
import sys

import torch
import torch.functional as F
import torch.optim as optim

import numpy as np

from neural_net_utils.utils import *
from neural_net_utils.argparseSetup import *
from neural_net_utils.networks import seq2Energy

sys.path.insert(1, '/home/eric/TICG-chromatin/scripts')
sys.path.insert(1, 'C:\\Users\\Eric\\OneDrive\\Documents\\Research\\Coding\\TICG-chromatin\\scripts')
from get_seq import relabel_seq

def train_seq2energy(gnn_opt):
    gnn_model = getModel(gnn_opt, verbose = False)
    gnn_model.to(gnn_opt.device)
    model_name = osp.join(gnn_opt.ofile_folder, 'model.pt')
    if osp.exists(model_name):
        save_dict = torch.load(model_name, map_location=torch.device('cpu'))
        gnn_model.load_state_dict(save_dict['model_state_dict'])
        train_loss_arr = save_dict['train_loss']
        val_loss_arr = save_dict['val_loss']
        print('Model is loaded: {}'.format(model_name))
    else:
        raise Exception('Model does not exist: {}'.format(model_name))
    gnn_model.eval()

    gnn_opt.batch_size = 1 # batch size must be 1
    dataset = getDataset(gnn_opt, True, True, verbose = False)
    _, val_dataloader, _ = getDataLoaders(dataset, gnn_opt)
    for i, data in enumerate(val_dataloader):
        assert gnn_opt.GNN_mode and not gnn_opt.autoencoder_mode
        data = data.to(gnn_opt.device)
        path = data.path[0]
        chis = np.load(osp.join(path, 'chis.npy'))
        chis = torch.Tensor(chis)
        print(chis)
        x = np.load(osp.join(path, 'x.npy'))
        x = torch.Tensor(x)

        seq = data.y
        # seq = torch.tensor(relabel_seq(seq.cpu().detach().numpy(), 'D-AB')).to(gnn_opt.device)

        m, k = seq.shape

        s = gnn_model(data)
        if gnn_opt.loss == 'BCE':
            # using BCE with logits loss, which combines sigmoid into loss
            # so need to do sigmoid here
            s = torch.sigmoid(s)
        s = s.detach()
        s = torch.reshape(s, (m, m))
        # s = seq @ chis @ seq.T
        # s = s.to(gnn_opt.device)

        model = seq2Energy(k)
        model.to(gnn_opt.device)
        model.train()
        optimizer = optim.Adam(model.parameters(), lr = 1e-1)
        criterion = F.mse_loss
        loss_arr = []
        for _ in range(1000):
            optimizer.zero_grad()
            s_hat = model.forward(seq)
            loss = criterion(s_hat, s)
            loss.backward()
            loss_arr.append(loss.item())
            optimizer.step()

        print('Final parameters: ')
        for name, p in model.named_parameters():
            print(name, p, p.shape)
            print(p.grad)
        print('Final loss: {}\n'.format(loss_arr[-1]))




def main():
    opt = argparseSetup()
    print(opt, '\n')
    train_seq2energy(opt)

    # cleanup
    if opt.root is not None and opt.delete_root:
        rmtree(opt.root)

if __name__ == '__main__':
    main()
