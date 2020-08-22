from __future__ import division
from __future__ import print_function

import os
import time
import logging
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import math
import scipy.sparse as sp
from torch.nn.parameter import Parameter

from utils import *
from models import *

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0,
                    help='number of gpus.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden1', type=int, default=10,
                    help='Number of hidden units for nodes.')
parser.add_argument('--hidden2', type=int, default=10,help='Number of hidden units for edge as nodes')
parser.add_argument('--att', type=int, default=30, help='the dimension of weight matrices for key and query')
parser.add_argument('--varimport', action='store_true', default = False, help = 'calculate variable importance')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--train', action='store_true', default = False, help='if need to train model or the parameters for training is already there')
parser.add_argument('--test_dataset',type=str)
parser.add_argument('--dataset',type=str, help='input dataset string')
parser.add_argument('--model', type = str, default = 'gcn',choices=['gcn','chebyshev'])
parser.add_argument('--max_degree',type=int, default = 3, help='number of supports')
parser.add_argument('--batch_size',type=int, default=8)
parser.add_argument('--save', type=str, default='./experiment1')
parser.add_argument('--save_test', action='store_true', default=False, help='If this is a optimized run! Use all data and save outputs')
parser.add_argument('--save_validation',action='store_true',default=False)
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
# test
def test():
    checkpoint = torch.load(os.path.join(args.save, 'model_for_test.pth'))
    #print("best epoch is:" + str(checkpoint['epoch']))
    model.load_state_dict(checkpoint['state_dict'])
    max_acc = 0
    with torch.no_grad():
        model.eval()
#        for j in range(100):
        logits_test = model(features[test_mask], adj_ls[test_mask])
        test_acc = accuracy(logits_test, torch.argmax(labels[test_mask],axis=1))
        #if test_acc > max_acc:
            #logits_test_fin = logits_test
            #max_acc = test_acc
        #print("Test accuracy is:" + str(max_acc))
    return test_acc
    #pkl.dump(logits_test_fin,open(os.path.join(args.save, 'logits_test'),'wb'))

def train():
    best_acc = 0
    for i in range(nepoch):
        for batch_mask in get_batch_iterator(tmp_mask, batch_size):
            optimizer.zero_grad()
            n = n + 1
    #     print('this is the {}th batch'.format(n))
            x = features[batch_mask]
            y = labels[batch_mask]
            y = torch.argmax(y,axis=1)
            adj = adj_ls[batch_mask]
            model.train()
            logits = model(x, adj)
            loss = criterion(logits,y)
            loss.backward()
            optimizer.step()
            train_acc = accuracy(logits, y)
        print("accuracy for {0}th epoch is: {1}".format(i,train_acc))
        if train_acc > best_acc:
                torch.save({'epoch': i,'state_dict': model.state_dict()}, os.path.join(args.save, 'model_for_test.pth'))
                best_acc = train_acc
                best_epo = i
is_cheby = True if args.model == 'chebyshev' else False
is_var = True if args.varimport else False
is_train = True if args.train else False
adj_ls, features, labels, sequences, proteases, labelorder, train_mask, val_mask, test_mask = load_data(args.dataset, is_test=args.test_dataset, norm_type=is_cheby)
cheby_params = args.max_degree if args.model == 'chebyshev' else None
tmp_mask = np.array([(not idx) for idx in test_mask], dtype=np.bool)

model = GCN(nnode=features.shape[1],
            nfeat=features.shape[2],
            mfeat=adj_ls.shape[3],
#            ngcn=args.ngcn,
            hidden1=args.hidden1,
            hidden2=args.hidden2,
            natt=args.att, # one layer
#            nfull=args.nfull,
#            nhid1=args.hidden1,
#            nhid2=args.hidden2,
            nclass=labels.shape[1],
            dropout=args.dropout,
            cheby=cheby_params)
batch_size = args.batch_size
n = 0

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=args.lr, weight_decay=args.weight_decay)
nepoch = args.epochs
#best_acc = 0
#for i in range(nepoch):
#    for batch_mask in get_batch_iterator(tmp_mask, batch_size):
#        optimizer.zero_grad()
#        n = n + 1
    #     print('this is the {}th batch'.format(n))
#        x = features[batch_mask]
#        y = labels[batch_mask]
#        y = torch.argmax(y,axis=1)
#        adj = adj_ls[batch_mask]
#        model.train()
#        logits = model(x, adj)
#        loss = criterion(logits,y)
#        loss.backward()
#        optimizer.step()
#        train_acc = accuracy(logits, y)
#    print("accuracy for {0}th epoch is: {1}".format(i,train_acc))
#    if train_acc > best_acc:
#            torch.save({'epoch': i,'state_dict': model.state_dict()}, os.path.join(args.save, 'model_for_test.pth'))
#            best_acc = train_acc
#            best_epo = i
if args.train == True:
    train()

if is_var == False:
    acc = test()
    print("Test accuracy is: {}".format(acc))

else:
    num_node = adj_ls.shape[1]
    var = num_node + num_node * (num_node - 1) / 2 # the number of nodes and edges
    acc_arr = np.zeros(int(var))
    print('number of candidate node/edges:{}'.format(var))
    print('number of nodes:{}'.format(num_node))
    print('number of edges:{}'.format(var - num_node))
    edge_ind = []
    for ind in range(39):
        k = ind + 1
        while k < 40:
            edge_ind.append((ind,k))
            k += 1
    for i in range(int(var)): # for each variable
        adj_ls, features, labels, sequences, proteases, labelorder, train_mask, val_mask, test_mask = load_data(args.dataset, is_test=args.test_dataset, norm_type=is_cheby)
        if i < num_node:
            for j in range(features.shape[2]): # for each node feature
                np.random.shuffle(features[:,i,j].cpu().numpy())
            print("Node Feature: {}".format(i+1))
        else:
            for j in range(adj_ls.shape[3]): # for each edge feature
                edge_node = i - num_node
                np.random.shuffle(adj_ls[:, edge_ind[edge_node][0], edge_ind[edge_node][1],j])
                after_shuffle = adj_ls[:,edge_ind[edge_node][0], edge_ind[edge_node][1], j]
                adj_ls[:,edge_ind[edge_node][0], edge_ind[edge_node][1], j] = after_shuffle
            print("Edge Feature: {}".format(edge_node + 1))

        acc = test()
        acc_arr[i] = acc
        print("Test Accuracy: {}".format(acc))
    df = pd.DataFrame(acc_arr, index = range(num_node)) # node + edge
    df.to_csv(os.path.join(args.save,"variable_importance.csv"))
