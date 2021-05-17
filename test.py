import numpy as np
import torch
import torch.nn as nn
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

out_act = nn.Sigmoid()
type_out_act = type(out_act)
print(type_out_act)
print(type(nn.Module()))
print(issubclass(type_out_act, nn.Module))
