import numpy as np
import torch
import torch.nn.functional as F


def mse_standardize(input, target):
    input_center = input - torch.mean(input)
    target_center = target - torch.mean(target)
    input_norm = input_center / torch.max(torch.abs(input_center))
    target_norm = target_centerr / torch.max(torch.abs(target_center))
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

def test():
    e = np.loadtxt('/home/erschultz/sequences_to_contact_maps/results/test/76/dataset_08_25_23_sample8/sample8/energy.txt')
    ehat = np.loadtxt('/home/erschultz/sequences_to_contact_maps/results/test/76/dataset_08_25_23_sample8/sample8/energy_hat.txt')
    print(np.mean(e))
    e = torch.tensor(e)
    ehat = torch.tensor(ehat)
    mse1, mse2 = mse_and_mse_center(ehat, e, split_loss = True)
    print(mse1, mse2)

if __name__ == '__main__':
    test()
