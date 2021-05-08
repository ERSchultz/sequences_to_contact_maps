import numpy as np
import torch
from neural_net_utils.base_networks import *
from neural_net_utils.networks import *

net = DeepC(1024, 2, [5, 5, 5], [32, 64, 128], [2, 4, 8, 16, 32, 64, 128, 256, 512], 128)

input = torch.tensor(np.arange(0, 1024*2, 1).reshape((1, 2, 1024))).type(torch.float32)

print(input)
print('---')


def v1():
    return 1, 2

def v2():
    return [1,2]

x, y = v1()
print(x,y)
x, y = v2()
print('here2', x, y)
