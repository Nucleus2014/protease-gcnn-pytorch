import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from layers import *


class GCN(nn.Module):
    def __init__(self, nnode, nfeat, mfeat, hidden1, linear, depth, natt, nclass, dropout, weight, is_des,cheby):
        super(GCN, self).__init__()
#        ngcn_list = ngcn.strip('[]').split(',')
#        nfull_list = nfull.strip('[]').split(',')
#        natt_list = natt.strip('[]').split(',')
        nin = nfeat # in_features
        self.dropout = dropout
        self.mfeat = mfeat
        self.weight = weight
        if self.weight == 'pre':
            ch = nnode
        elif self.weight == 'post':
            ch = mfeat
        if is_des == True:  
            self.hidden = [50,50,50,50,50,20,20,20,20,20]
        else:
            self.hidden = [hidden1] * depth
        gcn_layers = [] # build a list for gcnn layers
        if cheby == None:
            for nhid in range(depth):
                gcn_layers.append(GraphConvolution(nin, self.hidden[nhid]))
                gcn_layers.append(norm(ch,weight))
                gcn_layers.append(ConcatReLU())
                nin = int(self.hidden[nhid])
        else:
            self.gcn1 = GraphConvolutionChebyshev(nfeat, hidden1, cheby)
            self.gcn2 = GraphConvolutionChebyshev(hidden1, hidden1, cheby)
        if natt != 0:
            self.att = SelfAttention(self.hidden[-1], natt) # dimension of value keeps the same with input size
            # single head for now
            self.is_att = True
        else:
            self.is_att = False
        self.gc = nn.Sequential(*gcn_layers)
        self.flatten = Flatten()
        full_layers = []
        nin_full = self.hidden[-1] * nnode # in_features for fully connected layers, which is gcn output * number of node
        if linear != 0:
            full_layers.append(ConcatLinear(nin_full, int(linear)))
            full_layers.append(ConcatLinear(int(linear), nclass))
        else:
            full_layers.append(ConcatLinear(nin_full, nclass))
        self.linear = nn.Sequential(*full_layers)
        # modification here: replace FloatTensor with rand to make sure values in adjacency matrix are more than zero. 
        self.edgeweight = Parameter(torch.rand(mfeat,1))

    def forward(self, x, adj):
        if self.weight == 'pre':
            adj = torch.matmul(adj, self.edgeweight).view(adj.shape[0],adj.shape[1],-1)
        elif self.weight == 'post':
            x = x.view(x.shape[0],x.shape[1],-1,1).expand(x.shape[0],x.shape[1],x.shape[2],self.mfeat)
            
            x = torch.transpose(x,-1,-3)
            x = torch.transpose(x,-1,-2)
            adj = torch.transpose(adj,-1,-3)
            adj = torch.transpose(adj,-1,-2)
        for func in self.gc:
            x = func(x,adj)
            if self.is_att == True:
                x = self.att(x,adj)
        if self.weight == 'post':
            x = torch.transpose(x,-1,-3)
            x = torch.transpose(x,-3,-2)
            x = torch.matmul(x, self.edgeweight).view(x.shape[0],x.shape[1],-1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.flatten(x,adj)
        #x = self.linear(x,adj)
        for func_full in self.linear:
            x = func_full(x,adj)
        return x #F.log_softmax(x, dim=1)
