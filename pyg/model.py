import torch
from torch import nn
from torch_geometric.nn import GCNConv
from torch.nn import functional as F


class norm(nn.Module):
    def __init__(self, in_features, mode):
        super(norm, self).__init__()
        if mode == "pre":
            self.norm = torch.nn.BatchNorm1d(in_features)
        elif mode == "post":
            self.norm = torch.nn.BatchNorm2d(in_features)

    def forward(self, x):
        out = self.norm(x)
        return out


class GCN(torch.nn.Module):
    def __init__(
        self,
        nnode,
        nfeat,
        mfeat,
        hidden1,
        linear,
        depth,
        nclass,
        dropout,
        weight,
        is_des,
    ):
        super().__init__()
        nin = nfeat  # in_features
        self.dropout = dropout
        self.mfeat = mfeat
        self.weight = weight
        if self.weight == "pre":
            ch = nnode
        elif self.weight == "post":
            ch = mfeat
        if is_des == True:
            self.hidden = [50, 50, 50, 50, 50, 20, 20, 20, 20, 20]
        else:
            self.hidden = [hidden1] * depth
        self.gcn_layers = nn.ModuleList()  # build a list for gcnn layers
        self.after_gcn = nn.ModuleList()
        for nhid in range(depth):
            self.gcn_layers.append(GCNConv(nin, self.hidden[nhid]))
            layer = nn.Sequential()
            # layer.append(norm(ch, weight)) # FIXME
            layer.append(nn.ReLU())
            self.after_gcn.append(layer)
            nin = int(self.hidden[nhid])

        nin_full = (
            self.hidden[-1] * nnode
        )  # in_features for fully connected layers, which is gcn output * number of node
        self.linear = nn.Sequential()
        # if linear != 0:
        #     self.linear.append(nn.Linear(nin_full, int(linear)))
        #     self.linear.append(nn.Linear(int(linear), nclass))
        # else:
        #     self.linear.append(nn.Linear(nin_full, nclass))
        if linear != 0:
            self.linear.append(nn.Linear(self.hidden[-1], int(linear)))
            self.linear.append(nn.Linear(int(linear), nclass))
        else:
            self.linear.append(nn.Linear(self.hidden[-1], nclass))

        self.edge_weight = nn.Linear(mfeat, 1, bias=False)

    def forward(self, data):
        x = data.x
        # data.edge_attr is in shape [E, mfeat]
        if self.weight == "pre":
            edge_attr = self.edge_weight(data.edge_attr)
            for i in range(len(self.gcn_layers)):
                x = self.gcn_layers[i](x, data.edge_index, edge_attr)
                x = self.after_gcn[i](x)
        elif self.weight == "post":
            for i in range(len(self.gcn_layers)):
                x = self.gcn_layers[i](x, data.edge_index, edge_attr)
                x = self.after_gcn[i](x)
            x = x.view(x.shape[0],x.shape[1],-1,1).expand(x.shape[0],x.shape[1],x.shape[2],self.mfeat)
            x = torch.transpose(x,-1,-3).transpose(x,-3,-2)
            x = self.edge_weight(x).view(x.shape[0],x.shape[1],-1)
        x = F.dropout(x, self.dropout, training=self.training)
        # x = x.flatten()
        x = self.linear(x)
        return x