import torch
import torch.nn as nn
import math
import numpy as np
import time

def actToModule(act, none_mode = False):
    '''
    Converts input activation, act, to nn.Module activation.

    Inputs:
        act: activation function - string, nn.Module, or None
        none_mode: True to return None instead of nn.Identity for no activation

    act can be None, a string, or nn.Module already.
    '''
    if act is None:
        act = nn.Identity()
        if none_mode:
            return None
    elif issubclass(type(act), nn.Module):
        pass
    elif isinstance(act, str):
        if act.lower() == 'none':
            act = nn.Identity()
            if none_mode:
                return None
        elif act.lower() == 'sigmoid':
            act = nn.Sigmoid()
        elif act.lower() == 'relu':
            act = nn.ReLU(True)
        elif act.lower() == 'prelu':
            act =  nn.PReLU()
        elif act.lower() == 'leaky':
            act = nn.LeakyReLU(0.2, True)
        elif act.lower() == 'tanh':
            act = nn.Tanh()
        else:
            raise Exception("Unsupported activation {}".format(act))
    else:
        raise Exception("Unsupported activation {}".format(act))

    return act

class UnetBlock(nn.Module):
    '''U Net Block adapted from https://github.com/phillipi/pix2pix.'''
    def __init__(self, input_size, inner_size, output_size = None, subBlock = None,
                 outermost = False, innermost = False,
                 kernel_size = 4, stride = 2, padding = 2, bias = True,
                 activation1 = 'leaky', activation2 = 'relu', norm = None,
                 dropout = False, dropout_p = 0.5, out_act = nn.Tanh()):
        super(UnetBlock, self).__init__()
        self.outermost = outermost
        if output_size is None:
            output_size = input_size

        if norm == 'batch':
            norm1 = nn.BatchNorm2d(inner_size)
            norm2 = nn.BatchNorm2d(output_size)
        elif norm == 'instance':
            norm1 = nn.InstanceNorm2d(inner_size)
            norm2 = nn.InstanceNorm2d(output_size)
        elif norm is None:
            norm1 = nn.Identity()
            norm2 = nn.Identity()

        conv1 = nn.Conv2d(input_size, inner_size, kernel_size, stride, padding, bias)

        act1 = actToModule(activation1)
        act2 = actToModule(activation2)

        if outermost:
            conv2 = nn.ConvTranspose2d(inner_size * 2, output_size, kernel_size, stride, padding)
            down = [conv1]
            if out_act is not None:
                out_act = actToModule(out_act)
                up = [act2, conv2, out_act]
            else:
                up = [act2, conv2]
            model = down + [subBlock] + up
        elif innermost:
            conv2 = nn.ConvTranspose2d(inner_size, output_size, kernel_size, stride, padding, bias)
            down = [act1, conv1]
            up = [act2, conv2, norm2]
            model = down + up
        else:
            conv2 = nn.ConvTranspose2d(inner_size * 2, output_size, kernel_size, stride, padding, bias)
            down = [act1, conv1, norm1]
            up = [act2, conv2, norm2]
            if dropout:
                model = down + [subBlock] + up + [nn.Dropout(dropout_p)]
            else:
                model = down + [subBlock] + up
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            # skip connection
            return torch.cat([x, self.model(x)], 1)

class ResnetBlock(nn.Module):
    def __init__(self, num_channels, kernel_size = 3, stride = 1, padding = 1, bias = True,
                 activation = 'prelu', norm = None, dropout = None, dropout_p = 0.5):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size,
                                     stride, padding, bias = bias)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size,
                                     stride, padding, bias = bias)

        self.norm = norm
        if self.norm == 'batch':
            self.norm = nn.BatchNorm2d(num_channels)
        elif self.norm == 'instance':
            self.norm = nn.InstanceNorm2d(num_channels)

        self.drop = dropout
        if self.drop == 'drop2d':
            self.drop = nn.Dropout2d(dropout_p)
        elif self.drop == 'drop':
            self.drop = nn.Dropout(dropout_p)

        self.act = actToModule(activation)

    def forward(self, x):
        if self.norm is not None:
            out = self.norm(self.conv1(x))
        else:
            out = self.conv1(x)

        if self.act is not None:
            out = self.act(out)

        if self.norm is not None:
            out = self.norm(self.conv2(out))
        else:
            out = self.conv2(out)

        out = torch.add(out, x)

        if self.drop is not None:
            out = self.drop(out)

        if self.act is not None:
            out = self.act(out)
        return out

class ConvBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size = 3, stride = 1, padding = 1, bias = True,
                 activation = 'prelu', norm = None, pool = None, pool_kernel_size = 2, dropout = None,
                 dropout_p = 0.5, dilation = 1, residual = False, conv1d = False):
        super(ConvBlock, self).__init__()

        self.residual = residual
        if isinstance(activation, str) and activation.lower() == 'gated':
            self.gated = True
        else:
            self.gated = False

        if conv1d:
            conv_fn = nn.Conv1d
        else:
            conv_fn = nn.Conv2d

        model = [conv_fn(input_size, output_size, kernel_size,
                                    stride, padding, dilation, bias = bias)]
        model2 = [conv_fn(input_size, output_size, kernel_size,
                                    stride, padding, dilation, bias = bias)] # only used if self.gated

        if norm == 'batch':
            if conv1d:
                batch_fn = nn.BatchNorm1d
            else:
                batch_fn = nn.BatchNorm2d
        elif norm == 'instance':
            if conv1d:
                batch_fn = nn.InstanceNorm1d
            else:
                batch_fn = nn.InstanceNorm2d
        else:
            batch_fn = nn.Identity
        model.append(batch_fn(output_size))
        model2.append(batch_fn(output_size))

        if pool == 'maxpool':
            if conv1d:
                maxpool_fn = nn.MaxPool1d
            else:
                maxpool_fn = nn.MaxPool2d
            model.append(maxpool_fn(pool_kernel_size))
            model2.append(maxpool_fn(pool_kernel_size))

        if dropout == 'drop':
            if conv1d:
                dropout_fn = nn.Dropout
            else:
                dropout_fn = nn.Dropout2d
            model.append(dropout_fn(dropout_p))
            model2.append(dropout_fn(dropout_p))

        if activation is None:
            pass
        elif isinstance(activation, str) and activation.lower() == 'gated':
            model.append(nn.Tanh())
            model2.append(nn.Sigmoid())
            self.model2 = nn.Sequential(*model2)
        else:
            model.append(actToModule(activation))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        if self.gated:
            out = torch.mul(out, self.model2(x))

        if self.residual:
            return torch.add(out, x)
        else:
            return out

class DeconvBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size = 3, stride = 1, padding = 1,
                 bias = True, activation = 'prelu', norm = None, dropout = None,
                 dropout_p = 0.5, output_padding = 0, conv1d = False):
        super(DeconvBlock, self).__init__()

        if conv1d:
            conv_fn = nn.ConvTranspose1d
        else:
            conv_fn = nn.ConvTranspose2d

        model = [conv_fn(input_size, output_size, kernel_size,
                                               stride, padding, bias = bias, output_padding = output_padding)]

        if norm == 'batch':
            if conv1d:
                batch_fn = nn.BatchNorm1d
            else:
                batch_fn = nn.BatchNorm2d
        elif norm == 'instance':
            if conv1d:
                batch_fn = nn.InstanceNorm1d
            else:
                batch_fn = nn.InstanceNorm2d
        else:
            batch_fn = nn.Identity
        model.append(batch_fn(output_size))

        if dropout == 'drop':
            if conv1d:
                dropout_fn = nn.Dropout
            else:
                dropout_fn = nn.Dropout2d
            model.append(dropout_fn(dropout_p))
            model2.append(dropout_fn(dropout_p))

        if activation is None:
            pass
        elif isinstance(activation, str) and activation.lower() == 'gated':
            model.append(nn.Tanh())
            model2.append(nn.Sigmoid())
            self.model2 = nn.Sequential(*model2)
        else:
            model.append(actToModule(activation))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class LinearBlock(nn.Module):
    def __init__(self, input_size, output_size, bias = True, activation = 'prelu',
                 norm = None, dropout = None, dropout_p = 0.5):
        super(LinearBlock, self).__init__()

        self.linear = nn.Linear(input_size, output_size, bias = bias)
        model =  [self.linear]

        if norm == 'batch':
            model.append(nn.BatchNorm1d(output_size))
        elif norm == 'instance':
            model.append(nn.InstanceNorm1d(output_size))

        if dropout == 'drop':
            model.append(nn.Dropout(dropout_p))

        if activation is not None:
            model.append(actToModule(activation))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Symmetrize2D(nn.Module):
    '''
    https://github.com/calico/basenji/blob/master/basenji/layers.py

    Symmetrises input by computing (x + x.T) / 2'''
    def __init__(self):
        super(Symmetrize2D, self).__init__()

    def forward(self, x):

        if len(x.shape) == 4:
            # assume x is of shape N x C x H x W
            return  (x + torch.transpose(x, 2, 3)) / 2
        elif len(x.shape) == 3:
            # assume x is of shape N x H x W
            return  (x + torch.transpose(x, 1, 2)) / 2
        else:
            raise Exception('Invalid shape: {}'.format(x.shape))

class AverageTo2d(nn.Module):
    '''https://github.com/calico/basenji/blob/master/basenji/layers.py'''
    def __init__(self, concat_d = False, n = None, mode = 'avg'):
        """
        Inputs:
            concat_d: True if positional encoding should be appended
            n: spatial dimension
            mode: 'average' for default mode, 'concat' to concat instead of average, 'outer' to use outer product
        """
        super(AverageTo2d, self).__init__()
        assert mode in {'avg', 'concat', 'outer'}, 'Invalid mode: {}'.format(mode)
        self.concat_d = concat_d
        self.mode = mode
        if concat_d:
            assert n is not None
            d = torch.zeros((n, n))
            for i in range(1, n):
                y = torch.diagonal(d, offset = i)
                y[:] = torch.ones(n-i) * i
            d = d + torch.transpose(d, 0, 1)
            d = torch.unsqueeze(d, 0)
            d = torch.unsqueeze(d, 0)
            self.d = d

    def forward(self, x):
        # assume x is of shape N x C x m
        # memory expensive
        assert len(x.shape) == 3, "shape must be 3D"
        N, C, m = x.shape

        if self.mode == 'avg':
            x1 = torch.tile(x, (1, 1, m))
            x1 = torch.reshape(x1, (-1, C, m, m))
            x2 = torch.transpose(x1, 2, 3)
            x1 = torch.unsqueeze(x1, 0)
            x2 = torch.unsqueeze(x2, 0)
            out = torch.cat((x1, x2), dim = 0)
            out = torch.mean(out, dim = 0, keepdim = False)
        elif self.mode == 'concat':
            x1 = torch.tile(x, (1, 1, m))
            x1 = torch.reshape(x1, (-1, C, m, m))
            x2 = torch.transpose(x1, 2, 3)
            out = torch.cat((x1, x2), dim = 1)
        elif self.mode == 'outer':
            # see testAverageTo2dOuter for evidence that this works
            x1 = torch.tile(x, (1, C, m))
            x1 = torch.reshape(x1, (-1, C*C, m, m))
            x2 = torch.transpose(x1, 2, 3)

            # use indices to permute x2
            indices = []
            for i in range(C):
                indices.extend(range(i, i + C * (C-1) + 1, C))
            indices = torch.tensor(indices)
            if x2.is_cuda:
                indices = indices.to(x2.get_device())
            x2 = torch.index_select(x2, dim = 1, index = indices)

            out = torch.einsum('ijkl,ijkl->ijkl', x1, x2)

        del x1, x2

        if self.concat_d:
            # append abs(i - j)
            if out.is_cuda:
                self.d = self.d.to(out.get_device())
            out = torch.cat((out, torch.tile(self.d, (N, 1, 1, 1))), dim = 1)
        return out

def testAverageTo2dOuter():
    avg = AverageTo2d(mode = 'outer')
    verbose = True
    N = 1
    m = 10
    C_list = [2]
    for C in C_list:
        t0 = time.time()
        input = np.random.randint(low = 1, high = 10, size = (N, C, m))
        input = torch.tensor(input, dtype = torch.float32)
        if verbose:
            print(input[:, :, 0], input[:, :, 1])
        out1 = avg(input)
        if verbose:
            print(out1[:, :, 0, 1])
        tf = time.time()
        deltaT = np.round(tf - t0, 3)
        print("AverageTo2d time: {}".format(deltaT))

        t0 = time.time()
        out = np.zeros((N, C*C, m, m))
        for n in range(N):
            for i in range(m):
                for j in range(m):
                    out[n, :, i,j] = torch.flatten(torch.outer(input[n, :, i], input[n, :, j]))
        tf = time.time()
        deltaT = np.round(tf - t0, 3)
        print("For loop time: {}".format(deltaT))

        assert np.array_equal(out1, out)

def main():
    sym = Symmetrize2D()
    avg = AverageTo2d(mode = 'outer')
    # x = np.random.randint(low = 1, high = 10, size = (5, 10, 500))
    x = np.array([[[5,6,7],[3,9,12],[1, 7, 8]], [[5,3,1],[4,5,12],[1, 13, 7]]])
    x = np.reshape(x, (-1, 3, 6))
    x = torch.tensor(x, dtype = torch.float32)
    print(x, x.shape)
    x = avg(x)
    # print(x, x.shape)
    print(x[:, :, 0, 1])
    print(x[:, :, 1, 0])


if __name__ == '__main__':
    # main()
    testAverageTo2dOuter()
