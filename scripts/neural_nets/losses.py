import os.path as osp

import numpy as np
import scipy
import torch
import torch.nn.functional as F

from .base_networks import torch_mean_dist


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

    return mse_log(input_center, target_center)

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

class MSE_plaid():
    def __init__(self, log=False):
        if log:
            self.loss_fn = mse_log
        else:
            self.loss_fn = F.mse_loss

    def __call__(self, input, target):
        meanDist_input = torch_mean_dist(input)
        input_diag = torch.tensor(scipy.linalg.toeplitz(meanDist_input),
                                    dtype=float)

        input_plaid = input - input_diag
        input_plaid -= torch.mean(input_plaid)

        meanDist_target = torch_mean_dist(target)
        target_diag = torch.tensor(scipy.linalg.toeplitz(meanDist_input),
                                    dtype=float)
        if input.is_cuda:
            target_diag = target_diag.to(input.get_device())
        target_plaid = input - target_diag
        target_plaid -= torch.mean(target_plaid)

        return self.loss_fn(input_plaid, target_plaid)

class MSE_diag():
    def __init__(self, log=False):
        if log:
            self.loss_fn = mse_log
        else:
            self.loss_fn = F.mse_loss

    def __call__(self, input, target):
        input_diag = torch_mean_dist(input)
        input_diag -= torch.mean(input_diag)

        target_diag = torch_mean_dist(target)
        target_diag -= torch.mean(target_diag)

        return self.loss_fn(input_diag, target_diag)

class MSE_log_scc():
    def __init__(self, m):
        self.weights = np.zeros(m)
        for d in np.arange(0, m-1):
            n = m - d

            weight = n * np.var(np.arange(1, n+1)/n, ddof = 1)
            assert weight > 0, d
            self.weights[d] = weight

        self.weights /= np.max(self.weights)

        weights_toep = scipy.linalg.toeplitz(self.weights)
        self.weights_toep = torch.tensor(weights_toep, dtype=torch.float32)

    def __call__(self, input, target):
        input_log = torch.sign(input) * torch.log(torch.abs(input) + 1)
        target_log = torch.sign(target) * torch.log(torch.abs(target) + 1)
        diff = input_log - target_log
        if diff.is_cuda and not self.weights_toep.is_cuda:
            self.weights_toep = self.weights_toep.to(diff.get_device())
        error = torch.multiply(diff, self.weights_toep)
        return torch.mean(torch.square(error))

class MSE_plaid_eig():
    def __init__(self, log=False):
        if log:
            self.loss_fn = mse_log
        else:
            self.loss_fn = F.mse_loss

    def __call__(self, eigenvectors, input, target):
        if self.log:
            input_log = torch.sign(input) * torch.log(torch.abs(input) + 1)
            target_log = torch.sign(target) * torch.log(torch.abs(target) + 1)



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
    def __init__(self, lambda1, lambda2, k):
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
    def __init__(self, lambda1, lambda2, k):
        self.k = k
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def __call__(self, input, target, split_loss=False):
        mse1 = self.lambda1 * mse_log(input, target)
        mse2 = self.lambda2 * mse_top_k_diagonals(input, target, self.k)
        if split_loss:
            return mse1, mse2
        return mse1 + mse2

class MSE_log_and_MSE_log_scc():
    def __init__(self, lambda1, lambda2, m):
        self.m = m
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.mse_log_scc = MSE_log_scc(m)

    def __call__(self, input, target, split_loss=False):
        mse1 = self.lambda1 * mse_log(input, target)
        mse2 = self.lambda2 * self.mse_log_scc(input, target)
        if split_loss:
            return mse1, mse2
        return mse1 + mse2

class Combined_Loss():
    def __init__(self, criterions, lambdas, args):
        self.criterions = criterions
        self.lambdas = lambdas
        self.args = args

    def __call__(self, input, target, split_loss=False):
        loss_list = []
        tot_loss = 0
        for criterion, loss_lambda, arg in zip(self.criterions, self.lambdas, self.args):
            if arg is None:
                loss = loss_lambda * criterion(input, target)
            elif isinstance(arg, list):
                loss = loss_lambda * criterion(input, target, *arg)
            else:
                loss = loss_lambda * criterion(input, target, arg)
            loss_list.append(loss)
            tot_loss += loss

        if split_loss:
            return loss_list
        else:
            return tot_loss

def test():
    GNN_ID = 496
    dir = '/home/erschultz/sequences_to_contact_maps/results/ContactGNNEnergy/'
    gnn_dir = osp.join(dir, f'{GNN_ID}')
    assert osp.exists(gnn_dir)
    dataset = 'dataset_09_28_23'
    sample = 2452
    s_dir = osp.join(gnn_dir, f'{dataset}_sample{sample}/sample{sample}')
    e = np.loadtxt(osp.join(s_dir, 'energy.txt'))
    # e = scipy.linalg.toeplitz(np.arange(0, 512))
    ehat = np.loadtxt(osp.join(s_dir, 'energy_hat.txt'))
    # m=512
    # e = np.random.normal(size=(m,m))
    # ehat = np.random.normal(size=(m,m))
    # e = e[:10, :10]
    # ehat = ehat[:10,:10]
    # print(e)
    # print(e.shape)
    e = torch.tensor(e)
    ehat = torch.tensor(ehat)
    # mse1, mse2 = mse_and_mse_center(ehat, e, split_loss = True)
    # print(mse1, mse2)
    #
    # mse3 = mse_log(e, ehat)
    # print(mse3)

    # mse4 = mse_kth_diagonal(e, ehat, 2)

    metric = mse_top_k_diagonals
    # mse5 = metric(e, ehat, 2)
    # print(mse5)
    #
    # ind = torch.triu_indices(10, 10, -3)
    # print(e)
    # print(ind)
    # print(e[ind[0], ind[1]])

    metric2 = MSE_plaid(len(e))
    mse6 = metric2(e, ehat)
    print(mse6)

    metric3 = Combined_Loss([metric, metric2], [1, 0.1], [2, None])
    # mse7 = metric3(e, ehat)
    # print(mse7)

if __name__ == '__main__':
    test()
