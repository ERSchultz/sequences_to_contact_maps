import os.path as osp

import numpy as np
import scipy
import torch
import torch.nn.functional as F


def mse_standardize(input, target):
    input_center = input - torch.mean(input)
    target_center = target - torch.mean(target)
    input_norm = input_center / torch.max(torch.abs(input_center))
    target_norm = target_center / torch.max(torch.abs(target_center))
    return F.mse_loss(input_norm, target_norm)

def mse_center(input, target):
    input_center = input - torch.mean(input)
    target_center = target - torch.mean(target)
    return F.mse_loss(input_center, target_center)

def mse_and_mse_center(input, target, lambda1=1, lambda2=1, split_loss=False):
    mse1 = lambda1 * F.mse_loss(input, target)
    mse2 = lambda2 * mse_center(input, target)
    if split_loss:
        return mse1, mse2
    return mse1 + mse2

def mse_log(input, target):
    input_log = torch.sign(input) * torch.log(torch.abs(input) + 1)
    target_log = torch.sign(target) * torch.log(torch.abs(target) + 1)
    return F.mse_loss(input_log, target_log)

def mse_center_log(input, target):
    input_center = input - torch.mean(input)
    target_center = target - torch.mean(target)

    input_log = torch.sign(input_center) * torch.log(torch.abs(input_center) + 1)
    target_log = torch.sign(target_center) * torch.log(torch.abs(target_center) + 1)
    return F.mse_loss(input_log, target_log)

def mse_log_and_mse_center_log(input, target, lambda1=1, lambda2=1, split_loss=False):
    mse1 = lambda1 * mse_log(input, target)
    mse2 = lambda2 * mse_center_log(input, target)
    if split_loss:
        return mse1, mse2
    return mse1 + mse2

def mse_kth_diagonal(input, target, k):
    return F.mse_loss(torch.diagonal(input, k), torch.diagonal(target, k))

def mse_top_k_diagonals(input, target, k):
    loss = 0
    for i in range(1, k+1):
        loss += mse_kth_diagonal(input, target, i)

    return loss / k

class MSE_log_scc():
    def __init__(self, m):
        weights = scipy.linalg.toeplitz(np.arange(0, m))
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def __call__(input, target):
        input_log = torch.sign(input) * torch.log(torch.abs(input) + 1)
        target_log = torch.sign(target) * torch.log(torch.abs(target) + 1)
        diff = input_log - target_log
        error = torch.multiply(diff, self.weights)
        return torch.mean(torch.square(error))

class MSE_and_MSE_log():
    def __init__(self, lambda1, lambda2):
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def __call__(self, input, target, split_loss=False):
        mse1 = self.lambda1 * F.mse_loss(input, target)
        mse2 = self.lambda2 * mse_log(input, target)
        if split_loss:
            return mse1, mse2
        return mse1 + mse2

class MSE_log_and_MSE_kth_diagonal():
    def __init__(self, k, lambda1, lambda2):
        self.k = k
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def __call__(self, input, target, split_loss=False):
        mse1 = self.lambda1 * mse_log(input, target)
        mse2 = self.lambda2 * mse_kth_diagonal(input, target, self.k)
        if split_loss:
            return mse1, mse2
        return mse1 + mse2

class MSE_log_and_MSE_top_k_diagonals():
    def __init__(self, k, lambda1, lambda2):
        self.k = k
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def __call__(self, input, target, split_loss=False):
        mse1 = self.lambda1 * mse_log(input, target)
        mse2 = self.lambda2 * mse_top_k_diagonals(input, target, self.k)
        if split_loss:
            return mse1, mse2
        return mse1 + mse2

def test():
    GNN_ID = 496
    dir = '/home/erschultz/sequences_to_contact_maps/results/ContactGNNEnergy/'
    gnn_dir = osp.join(dir, f'{GNN_ID}')
    assert osp.exists(gnn_dir)
    dataset = 'dataset_09_28_23'
    sample = 2452
    s_dir = osp.join(gnn_dir, f'{dataset}_sample{sample}/sample{sample}')
    # e = np.loadtxt(osp.join(s_dir, 'energy.txt'))
    e = scipy.linalg.toeplitz(np.arange(0, 512))
    ehat = np.loadtxt(osp.join(s_dir, 'energy_hat.txt'))
    e = e[:10, :10]
    ehat = ehat[:10,:10]
    print(e)
    print(e.shape)
    e = torch.tensor(e)
    ehat = torch.tensor(ehat)
    # mse1, mse2 = mse_and_mse_center(ehat, e, split_loss = True)
    # print(mse1, mse2)
    #
    # mse3 = mse_log(e, ehat)
    # print(mse3)

    # mse4 = mse_kth_diagonal(e, ehat, 2)

    # mse5 = mse_top_k_diagonals(e, ehat, 2)
    # print(mse5)
    #
    # ind = torch.triu_indices(10, 10, -3)
    # print(e)
    # print(ind)
    # print(e[ind[0], ind[1]])

    metric = MSE_log_scc(len(e))
    mse6 = metric(e, ehat)
    print(mse6)


if __name__ == '__main__':
    test()
