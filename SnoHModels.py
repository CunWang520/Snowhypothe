import utils
from SnoHLayers import SnoHGCNconv, SnoHGCN2Conv
from torch.nn import BatchNorm1d
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCN, JumpingKnowledge


class BasicSnoHGCN(nn.Module):
    def __init__(self, in_channels, out_channels, activation, withbn, withres, res=False, resplus=False):
        super().__init__()
        self.conv = SnoHGCNconv(in_channels, out_channels)
        self.activation = activation
        self.withbn = withbn
        self.bn = lambda x: x
        if withbn:
            if not resplus:
                self.bn = BatchNorm1d(out_channels)
            else:
                self.bn = BatchNorm1d(in_channels)
        self.withres = withres
        self.res = res
        self.resplus = resplus
        if res and resplus:
            raise ValueError("res and resplus can not both be True")
        self.lin_mapping = lambda x: x
        if withres and in_channels != out_channels:
            self.lin_mapping = nn.Linear(in_channels, out_channels, bias=False)
        # self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()
        if self.withbn:
            self.bn.reset_parameters()
        if isinstance(self.lin_mapping, nn.Module):
            self.lin_mapping.reset_parameters()

    def forward(self, x, edge_index):
        if not self.resplus:
            out, graph_cos_dist = self.conv(x, edge_index)
            out = self.bn(out)
            out = self.activation(out)
            if self.withres:
                out = out + self.lin_mapping(x)
        else:
            out = self.bn(x)
            out = self.activation(out)
            out, graph_cos_dist = self.conv(out, edge_index)
            if self.withres:
                out = out + self.lin_mapping(x)
        return out, graph_cos_dist


class BasicSnoHGCN2(nn.Module):
    def __init__(self, mid_channels, alpha, theta, layer, activation, withbn, withres):
        super().__init__()
        self.conv = SnoHGCN2Conv(mid_channels, alpha, theta, layer)
        self.activation = activation
        self.withbn = withbn
        self.bn = lambda x: x
        if withbn:
            self.bn = BatchNorm1d(mid_channels)
        self.withres = withres
        self.lin_mapping = lambda x: x
        if withres:
            self.lin_mapping = nn.Linear(mid_channels, mid_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()
        if self.withbn:
            self.bn.reset_parameters()
        if isinstance(self.lin_mapping, nn.Module):
            self.lin_mapping.reset_parameters()

    def forward(self, x, x_0, edge_index):
        out, graph_cos_dist = self.conv(x, x_0, edge_index)
        out = self.bn(out)
        out = self.activation(out)
        if self.withres:
            out = out + self.lin_mapping(x)
        return out, graph_cos_dist


class SnoHGCN(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, numlayer, activation, layerwithbn, block):
        super().__init__()
        self.gcnblock = nn.ModuleList()
        if numlayer <= 1:
            raise ValueError("numlayer must be greater than or equal to 2")
        else:
            res = False
            resplus = False
            if block == "gcn":
                layerwithres = False
            elif block == "resgcn":
                layerwithres = True
                res = True
                resplus = False
            elif block == "resgcn+":
                layerwithres = True
                res = False
                resplus = True
            else:
                raise NotImplementedError(f"{block} is not implemented, you can choose gcn, resgcn, resgcn+")

            for i in range(numlayer):
                if i == 0:
                    self.gcnblock.append(BasicSnoHGCN(in_channels, mid_channels, activation=activation, withbn=layerwithbn, withres=False))
                elif i == numlayer - 1:
                    # self.gcnblock.append(
                    #     BasicSnoHGCN(mid_channels, out_channels, activation=lambda x:x, withbn=layerwithbn,
                    #                  withres=layerwithres))
                    self.gcnblock.append(BasicSnoHGCN(mid_channels, out_channels, activation=lambda x: x, withbn=False, withres=False))
                else:
                    self.gcnblock.append(BasicSnoHGCN(mid_channels, mid_channels, activation=activation, withbn=layerwithbn, withres=layerwithres, res=res, resplus=resplus))
        self.edge_indexes = None
        self.numlayer = numlayer

    def get_edge_indexes(self, edge_indexes):
        self.edge_indexes = edge_indexes

    def forward(self, x):
        layers_cos_dist = []
        out, cos_dist = self.gcnblock[0](x, self.edge_indexes[0])
        out = F.dropout(out, training=self.training)

        layers_cos_dist.append(cos_dist.detach().cpu())
        for i in range(1, len(self.gcnblock)):
            out, cos_dist = self.gcnblock[i](out, self.edge_indexes[i])
            layers_cos_dist.append(cos_dist.detach().cpu())
            if i == len(self.gcnblock) - 1:
                continue
        out = F.log_softmax(out, dim=1)
        return out, layers_cos_dist
