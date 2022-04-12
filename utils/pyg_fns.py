from typing import Union

import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.transforms import BaseTransform
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import degree
from torch_scatter import scatter_max, scatter_mean, scatter_min, scatter_std
from torch_sparse import SparseTensor, matmul


# node transforms
class WeightedLocalDegreeProfile(BaseTransform):
    '''
    Weighted version of Local Degree Profile (LDP) from https://arxiv.org/abs/1811.03508
    Appends WLDP features to feature vector.

    Reference code: https://pytorch-geometric.readthedocs.io/en/latest/_modules/
        torch_geometric/transforms/local_degree_profile.html#LocalDegreeProfile
    '''
    def __call__(self, data):
        row, col = data.edge_index
        N = data.num_nodes

        # weighted_degree must exist
        deg = data.weighted_degree
        deg_col = deg[col]

        min_deg, _ = scatter_min(deg_col, row, dim_size=N)
        min_deg[min_deg > 10000] = 0
        max_deg, _ = scatter_max(deg_col, row, dim_size=N)
        max_deg[max_deg < -10000] = 0
        mean_deg = scatter_mean(deg_col, row, dim_size=N)
        std_deg = scatter_std(deg_col, row, dim_size=N)

        x = torch.stack([deg, min_deg, max_deg, mean_deg, std_deg], dim=1)

        if data.x is not None:
            data.x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([data.x, x], dim=-1)
        else:
            data.x = x

        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}')

class Degree(BaseTransform):
    '''
    Appends degree features to feature vector.

    Reference code: https://pytorch-geometric.readthedocs.io/en/latest/_modules/
        torch_geometric/transforms/target_indegree.html#TargetIndegree
    '''
    def __init__(self, norm = True, max_val = None, weighted = False,
                split_edges = False, split_val = 0):
        self.norm = norm # bool
        self.max = max_val # float
        self.weighted = weighted # bool
        self.split_edges = split_edges # bool
        self.split_val = split_val # float

    def __call__(self, data):
        if self.weighted:
            deg = data.weighted_degree
            if self.split_edges:
                ypos = torch.clone(data.contact_map)
                ypos[ypos < self.split_val] = 0
                pos_deg = torch.sum(ypos, axis = 1)
                del ypos
                yneg = torch.clone(data.contact_map)
                yneg[yneg > self.split_val] = 0
                neg_deg = torch.sum(yneg, axis = 1)
                del yneg
        else:
            deg = degree(data.edge_index[0], data.num_nodes)
            if self.split_edges:
                pos_deg = degree((data.contact_map > self.split_val).nonzero().t()[0], data.num_nodes)
                neg_deg = degree((data.contact_map < self.split_val).nonzero().t()[0], data.num_nodes)

        if self.norm:
            deg /= (deg.max() if self.max is None else self.max)
            if self.split_edges:
                # if statement is a safety check to avoid divide by 0
                pos_deg /= (pos_deg.max() if pos_deg.max() > 0 else 1)

                neg_deg /= (neg_deg.max() if neg_deg.max() > 0 else neg_deg.min())

        if self.split_edges:
            deg = torch.stack([deg, pos_deg, neg_deg], dim=1)
        else:
            deg = torch.stack([deg], dim=1)

        if data.x is not None:
            data.x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([data.x, deg], dim=-1)
        else:
            data.x = deg

        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__} ('
                f'norm={self.norm}, max={self.max}, '
                f'weighted={self.weighted}, split_edges={self.split_edges}, '
                f'split_val={self.split_val})')

class AdjPCATransform(BaseTransform):
    '''Appends values from top k PCs of adjacency matrix to feature vector.'''
    def __init__(self, k = 5):
        self.k = k

    def __call__(self, data):
        pca = PCA(n_components = self.k)
        y_trans = pca.fit_transform(torch.clone(data.contact_map))

        y_trans = torch.tensor(y_trans, dtype = torch.float32)

        if data.x is not None:
            data.x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([data.x, y_trans], dim=-1)
        else:
            data.x = y_trans

        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}')

class AdjTransform(BaseTransform):
    '''Appends rows of adjacency matrix to feature vector.'''
    def __call__(self, data):
        if data.x is not None:
            data.x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([data.x, torch.clone(data.contact_map)], dim=-1)
        else:
            data.x = torch.clone(data.contact_map)

        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}')

class OneHotGeneticPosition(BaseTransform):
    '''Appends one hot encoded genetic position to feature vector.'''
    def __call__(self, data):
        pos = torch.arange(0, data.num_nodes, dtype=torch.long)
        pos = F.one_hot(pos, num_classes=data.num_nodes).to(torch.float)

        if data.x is not None:
            data.x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([data.x, pos], dim=-1)
        else:
            data.x = pos

        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}')

class GeneticPosition(BaseTransform):
    '''Appends genetic position to feature vector.'''
    def __init__(self, center = False, norm = False):
        self.center = center # bool
        self.norm = norm # bool

    def __call__(self, data):
        pos = torch.arange(0, data.num_nodes, dtype=torch.float32).reshape(data.num_nodes, 1)

        if self.center:
            pos -= pos.mean(dim=-2, keepdim=True)
        if self.norm:
            pos /= pos.max()

        if data.x is not None:
            data.x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([data.x, pos], dim=-1)
        else:
            data.x = pos

        return data

    def __repr__(self):
        return (f'{self.__class__.__name__}'
                f'(center={self.center}, norm={self.norm})')


# edge transforms
class GeneticDistance(BaseTransform):
    '''
    Appends genetic distance to features to edge attr vector.

    Based off of https://pytorch-geometric.readthedocs.io/en/latest/
    _modules/torch_geometric/transforms/distance.html
    Note that GeneticDistance doesn't assume data.pos exists while Distance does
    '''
    def __init__(self, norm = True, max_val = None, cat = True,
                split_edges = False, convert_to_attr = False):
        self.norm = norm
        self.max = max_val
        self.cat = cat
        self.split_edges = split_edges # bool
        self.convert_to_attr = convert_to_attr # bool, converts to 2d array


    def __call__(self, data):
        pos = torch.arange(0, data.num_nodes, dtype=torch.float32).reshape(data.num_nodes, 1)

        if self.split_edges:
            # positive
            if 'pos_edge_attr' in data._mapping:
                pseudo = data.pos_edge_attr
            else:
                pseudo = None

            row, col = data.pos_edge_index
            dist = torch.norm(pos[col] - pos[row], p=2, dim=-1)
            if self.norm and dist.numel() > 0:
                dist = dist / (dist.max() if self.max is None else self.max)
            if self.convert_to_attr:
                dist = dist.reshape(-1, 1)

            if pseudo is not None and self.cat:
                pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
                data.pos_edge_attr = torch.cat([pseudo, dist.type_as(pseudo)], dim=-1)
            else:
                data.pos_edge_attr = dist

            # negative
            if 'neg_edge_attr' in data._mapping:
                pseudo = data.neg_edge_attr
            else:
                pseudo = None
            row, col = data.neg_edge_index
            dist = torch.norm(pos[col] - pos[row], p=2, dim=-1)
            if self.norm and dist.numel() > 0:
                dist = dist / (dist.max() if self.max is None else self.max)
            if self.convert_to_attr:
                dist = dist.reshape(-1, 1)

            if pseudo is not None and self.cat:
                pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
                data.neg_edge_attr = torch.cat([pseudo, dist.type_as(pseudo)], dim=-1)
            else:
                data.neg_edge_attr = dist

        else:
            (row, col), pseudo = data.edge_index, data.edge_attr
            dist = torch.norm(pos[col] - pos[row], p=2, dim=-1).view(-1, 1)

            if self.norm and dist.numel() > 0:
                dist = dist / (dist.max() if self.max is None else self.max)

            if pseudo is not None and self.cat:
                pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
                data.edge_attr = torch.cat([pseudo, dist.type_as(pseudo)], dim=-1)
            else:
                data.edge_attr = dist

        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(norm={self.norm}, '
                f'max_value={self.max})')

class ContactDistance(BaseTransform):
    '''
    Appends contact map entries to edge attr vector.
    '''
    def __init__(self, cat = True, split_edges = False, split_val = 0,
                    convert_to_attr = False):
        self.cat = cat
        self.split_edges = split_edges # bool
        self.split_val = split_val # float
        self.convert_to_attr = convert_to_attr # bool, converts to 2d array

    def __call__(self, data):
        if self.split_edges:
            # positive
            if 'pos_edge_attr' in data._mapping:
                pseudo = data.pos_edge_attr
            else:
                pseudo = None

            row, col = data.pos_edge_index
            pos_edge_attr = data.contact_map[row, col]
            if self.convert_to_attr:
                pos_edge_attr = pos_edge_attr.reshape(-1, 1)

            if pseudo is not None and self.cat:
                pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
                data.pos_edge_attr = torch.cat([pseudo, pos_edge_attr.type_as(pseudo)], dim=-1)
            else:
                data.pos_edge_attr = pos_edge_attr

            # negative
            if 'neg_edge_attr' in data._mapping:
                pseudo = data.neg_edge_attr
            else:
                pseudo = None
            row, col = data.neg_edge_index
            neg_edge_attr = data.contact_map[row, col]
            if self.convert_to_attr:
                neg_edge_attr = neg_edge_attr.reshape(-1, 1)

            if pseudo is not None and self.cat:
                pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
                data.neg_edge_attr = torch.cat([pseudo, neg_edge_attr.type_as(pseudo)], dim=-1)
            else:
                data.neg_edge_attr = neg_edge_attr

        else:
            (row, col), pseudo = data.edge_index, data.edge_attr
            edge_attr = data.contact_map[row, col]
            if self.convert_to_attr:
                edge_attr = edge_attr.reshape(-1, 1)

            if pseudo is not None and self.cat:
                pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
                data.edge_attr = torch.cat([pseudo, edge_attr.type_as(pseudo)], dim=-1)
            else:
                data.edge_attr = edge_attr

        return data


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}, '
                f'({self.convert_to_attr})')

# message passing
class WeightedSignedConv(MessagePassing):
    '''
    Variant of SignedConv that allows for edge weights or edge features.
    Adapted from: https://pytorch-geometric.readthedocs.io/en/latest/_modules/
            torch_geometric/nn/conv/signed_conv.html#SignedConv

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        first_aggr (bool): Denotes which aggregation formula to use.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    '''
    def __init__(self, in_channels: int, out_channels: int, first_aggr: bool,
                 bias: bool = True, edge_dim: int = 0, **kwargs):

        kwargs.setdefault('aggr', 'mean')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first_aggr = first_aggr
        self.edge_dim = edge_dim

        if first_aggr:
            self.lin_pos_l = Linear(in_channels + self.edge_dim, out_channels, False)
            self.lin_pos_r = Linear(in_channels, out_channels, bias)
            self.lin_neg_l = Linear(in_channels + self.edge_dim, out_channels, False)
            self.lin_neg_r = Linear(in_channels, out_channels, bias)
        else:
            self.lin_pos_l = Linear(2 * (in_channels + self.edge_dim), out_channels, False)
            self.lin_pos_r = Linear(in_channels, out_channels, bias)
            self.lin_neg_l = Linear(2 * (in_channels + self.edge_dim), out_channels, False)
            self.lin_neg_r = Linear(in_channels, out_channels, bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_pos_l.reset_parameters()
        self.lin_pos_r.reset_parameters()
        self.lin_neg_l.reset_parameters()
        self.lin_neg_r.reset_parameters()


    def forward(self, x: Union[Tensor, PairTensor], pos_edge_index: Adj,
                neg_edge_index: Adj, pos_edge_attr: OptTensor = None,
                neg_edge_attr: OptTensor = None):
        """"""
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        # propagate_type: (x: PairTensor, edge_weight: OptTensor)
        if self.first_aggr:
            out_pos = self.propagate(pos_edge_index, x=x, size=None,
                                    edge_attr=pos_edge_attr)
            out_pos = self.lin_pos_l(out_pos)
            out_pos += self.lin_pos_r(x[1])

            out_neg = self.propagate(neg_edge_index, x=x, size=None,
                                    edge_attr=neg_edge_attr)
            out_neg = self.lin_neg_l(out_neg)
            out_neg += self.lin_neg_r(x[1])

            return torch.cat([out_pos, out_neg], dim=-1)

        else:
            F_in = self.in_channels

            out_pos1 = self.propagate(pos_edge_index, size=None,
                                    edge_attr=pos_edge_attr,
                                    x=(x[0][..., :F_in], x[1][..., :F_in]))
            out_pos2 = self.propagate(neg_edge_index, size=None,
                                    edge_attr=neg_edge_attr,
                                    x=(x[0][..., F_in:], x[1][..., F_in:]))
            out_pos = torch.cat([out_pos1, out_pos2], dim=-1)
            out_pos = self.lin_pos_l(out_pos)
            out_pos += self.lin_pos_r(x[1][..., :F_in])

            out_neg1 = self.propagate(pos_edge_index, size=None,
                                    edge_attr=pos_edge_attr,
                                    x=(x[0][..., F_in:], x[1][..., F_in:]))
            out_neg2 = self.propagate(neg_edge_index, size=None,
                                    edge_attr=neg_edge_attr,
                                    x=(x[0][..., :F_in], x[1][..., :F_in]))
            out_neg = torch.cat([out_neg1, out_neg2], dim=-1)
            out_neg = self.lin_neg_l(out_neg)
            out_neg += self.lin_neg_r(x[1][..., F_in:])

            return torch.cat([out_pos, out_neg], dim=-1)


    def message(self, x_j: Tensor, edge_attr: OptTensor) -> Tensor:
        if self.edge_dim == 0:
            # treat edge_attr as a weight if it exists
            result = x_j if edge_attr is None else edge_attr.view(-1, 1) * x_j
        else:
            result = x_j if edge_attr is None else torch.cat((x_j, edge_attr), axis = 1)
        return result


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, first_aggr={self.first_aggr},'
                f'edge_dim={self.edge_dim})')
