import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn.functional import softmax
from utils import chebyshev


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(adj, input)
        output = torch.matmul(support, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GraphConvolutionChebyshev(Module):
    def __init__(self, in_features, out_features, cheby, bias=True):
        super(GraphConvolutionChebyshev, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.K = Parameter(torch.FloatTensor(cheby,1)) # degree of chebyshev polynomial
        self.max_degree = cheby
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        stdv2 = 1. / math.sqrt(self.K.size(1))
        self.K.data.uniform_(-stdv2, stdv2)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            
    def forward(self, input, adj):
        support = chebyshev(input, adj, self.max_degree) # build the tensor form of chebyshev polynomials
        support = torch.matmul(support, self.K).view(support.shape[0],support.shape[1],-1)
        output = torch.matmul(support, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
                  + str(self.in_features) + ' -> ' \
                  + str(self.out_features) + ')'
        
class Flatten(Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x, adj):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)

class ConcatLinear(Module):
    def __init__(self, in_dim, out_dim):
        super(ConcatLinear, self).__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim)
        
    def forward(self, x, adj):
        out = self.linear(x)
        return out

class ConcatReLU(Module):
    def __init__(self):
        super(ConcatReLU, self).__init__()
        self.relu = torch.nn.ReLU()
    def forward(self, x, adj):
        out = self.relu(x)
        return out

class norm(Module):
    def __init__(self, in_features, mode):
        super(norm, self).__init__()
        if mode == 'pre':
            self.norm = torch.nn.BatchNorm1d(in_features)
        elif mode == 'post':
            self.norm = torch.nn.BatchNorm2d(in_features)
    def forward(self,x,adj):
        out = self.norm(x)
        return out

class SelfAttention(Module):
    def __init__(self, in_features, w_features):
        super(SelfAttention, self).__init__()
        self.w_key = Parameter(torch.FloatTensor(in_features, w_features))
        self.w_value = Parameter(torch.FloatTensor(in_features, in_features))
        self.w_query = Parameter(torch.FloatTensor(in_features, w_features))
    def forward(self,x,adj):
        keys = x @ self.w_key # ? x N x W
        querys = x @ self.w_query # ? x N x W
        values = x @ self.w_value # ? x N x F
        attn_scores = torch.zeros(x.shape[0],x.shape[1],x.shape[1])
        for b in range(x.shape[0]):
            attn_scores[b] = softmax(querys[b] @ keys[b].T, dim=-1) # ? x N x N
        out = torch.zeros_like(values) # ? x N x F
        for b in range(x.shape[0]):
            weighted_values = values[b][:,None] * attn_scores[b].T[:,:,None]
            out[b] = weighted_values.sum(dim=0)
        return out
            
        
        
        
