import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from layers import *


class GCN(nn.Module):
    def __init__(self, nnode, nfeat, mfeat, hidden1, hidden2, natt, nclass, dropout,cheby):
        super(GCN, self).__init__()
#        ngcn_list = ngcn.strip('[]').split(',')
#        nfull_list = nfull.strip('[]').split(',')
#        natt_list = natt.strip('[]').split(',')
#        nin = nfeat # in_features
        self.dropout = dropout
#        gcn_layers = [] # build a list for gcnn layers
#        if cheby == None:
#            for nhid in ngcn_list:
#                gcn_layers.append(GraphConvolution(nin, int(nhid)))
#                gcn_layers.append(norm(nnode))
#                gcn_layers.append(ConcatReLU())
#                nin = int(nhid)
        self.norm1 = norm(nnode)
        self.relu1 = ConcatReLU()
        self.norm2 = norm(nnode)
        self.relu2 = ConcatReLU()
        #self.norm0 = norm(nnode)
        #self.norm3 = norm(nnode)
        #self.relu3 = ConcatReLU()
        if cheby == None:
            self.gcn1 = GraphConvolution(nfeat, hidden1)
            self.gcn2 = GraphConvolution(hidden1, hidden2)
            #self.gcn3 = GraphConvolution(hidden2, 20)
        else:
            self.gcn1 = GraphConvolutionChebyshev(nfeat, hidden1, cheby)
            self.gcn2 = GraphConvolutionChebyshev(hidden1, hidden2, cheby)
#            for nhid in ngcn_list:
#                gcn_layers.append(GraphConvolutionChebyshev(nin, int(nhid), cheby))
#                gcn_layers.append(norm(nnode))
#                gcn_layers.append(ConcatReLU())
#                nin = int(nhid)
#        natt_in = int(ngcn_list[-1])
#        if natt_list != ['']:
#            for nhid in natt_list:
#                gcn_layers.append(SelfAttention(natt_in, nhid))
#                natt_in = nhid
        if natt != 0:
            self.att = SelfAttention(hidden2, natt) # dimension of value keeps the same with input size
            # single head for now
            self.is_att = True
        else:
            self.is_att = False
#        self.gc = gcn_layers
#        self.gc1 = GraphConvolution(nfeat, nhid1)
#        self.gc2 = GraphConvolution(nhid1, nhid2)
#        full_layers = []
#        nin_full = nhid * nnode # in_features for fully connected layers, which is gcn output * number of node
#        for nlinear in nfull_list:
#            if nlinear != '':
#                full_layers.append(ConcatLinear(nin_full, int(nlinear)))
#                nin_full = int(nlinear)
#        full_layers.append(ConcatLinear(nin_full, nclass))
#        self.full = full_layers
        self.flatten = Flatten()
        #self.linear = ConcatLinear(hidden2 * nnode, nclass)
        self.linear0 = ConcatLinear(hidden2 * nnode, 1024)
        self.linear1 = ConcatLinear(1024, nclass) 
        self.edgeweight = Parameter(torch.FloatTensor(mfeat,1))

    def forward(self, x, adj):
        adj = torch.matmul(adj, self.edgeweight).view(adj.shape[0],adj.shape[1],-1)
#        for func in self.gc:
#            x = func(x,adj)
        #x = self.norm0(x,adj)
        x = self.gcn1(x,adj)
        x = self.norm1(x,adj)
        x = self.relu1(x,adj)
        x = self.gcn2(x,adj)
        x = self.norm2(x,adj)
        x = self.relu2(x,adj)
        #x = self.gcn3(x,adj)
        #x = self.norm3(x,adj)
        #x = self.relu3(x,adj)
        if self.is_att == True:
            x = self.att(x,adj)
        x = F.dropout(x, self.dropout, training=self.training)
#        x = F.relu(self.gc2(x, adj))
#        x = F.relu(self.gc3(x, adj))
        x = self.flatten(x,adj)
        #x = self.linear(x,adj)
        x = self.linear0(x,adj)
        x = self.linear1(x,adj)
        #x = self.linear2(x,adj)
#        for func_full in self.full:
#            x = func_full(x,adj)
#        x = self.linear2(x,adj)
        return x#F.log_softmax(x, dim=1)
