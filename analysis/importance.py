# This script is to calculate dropping accuracy for each node/edge to show each importance
# Author: Changpeng Lu
# Usage
# python importance.py --importance --dataset HCV_ternary_10_ang_aa_energy_7_energyedge_5_hbond --test_dataset HCV_ternary_10_ang_aa_energy_7_energyedge_5_hbond --hidden1 20 --depth 2 --linear 0 --att 0 --batch_size 500 --lr 0.005 --dropout 0.05 --weight_decay 5e-4 --save 'outputs/tt/HCV_ternary_10_ang_aa_energy_7_energyedge_5_hbond/bs_500/'

from __future__ import division
from __future__ import print_function

import os
import time
import logging
import argparse
import numpy as np
import pandas as pd

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
#parser.add_argument('--fastmode', action='store_true', default=False,
#                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden1', type=int, default=10,
                    help='Number of hidden units for nodes.')
parser.add_argument('--depth', type=int, default=10, help='Number of gcnn layers')
parser.add_argument('--att', type=int, default=0, help='the dimension of weight matrices for key and query')
parser.add_argument('--linear', type=int, default=0)
parser.add_argument('--dropout', type=float, default=0.1,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--no_energy', action='store_true', default=False)
parser.add_argument('--test_dataset',type=str)
parser.add_argument('--dataset',type=str, help='input dataset string')
parser.add_argument('--model', type = str, default = 'gcn',choices=['gcn','chebyshev'])
parser.add_argument('--max_degree',type=int, default = 3, help='number of supports')
parser.add_argument('--batch_size',type=int, default=8)
parser.add_argument('--weight', type=str, default='pre',choices=['pre','post'])
parser.add_argument('--dim_des',action='store_true',default=False)
parser.add_argument('--save', type=str, default='./experiment1')
parser.add_argument('--importance',action='store_true', default = False, help='Whether calculate each variable''s importance.')
args = parser.parse_args()

makedirs(args.save)
logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
logger.info(args)

# test
def test(X, graph, y, testmask, model_for_test, hidden1, linear, learning_rate, weight_decay, batch_size, dropout, path_save):
    checkpoint = torch.load(os.path.join(path_save, 'model_for_test_hidden_' + str(hidden1) + '_linear_' + str(linear) +'_lr_'+str(learning_rate)+'_wd_'+str(weight_decay)+'_bs_'+str(batch_size)+ '_dt_' + str(dropout) + '.pth'))
    logger.info("best epoch is:" + str(checkpoint['epoch']))
    model_for_test.load_state_dict(checkpoint['state_dict'])
    max_acc = 0
    with torch.no_grad():
        model_for_test.eval()
        for j in range(100):
            logits_test = model_for_test(X[testmask], graph[testmask])
            test_acc = accuracy(logits_test, torch.argmax(y[testmask],axis=1))
            if test_acc > max_acc:
                logits_test_fin = logits_test
                max_acc = test_acc
    return logits_test_fin, max_acc
    

# variable importance
def importance(all_features, all_graph, ys, full_test_mask, trained_model, hidden1, linear, learning_rate, \
              weight_decay, batch_size, dropout, path_save):
    num_node = all_graph.shape[1]
    var = int(num_node + num_node * (num_node - 1) / 2) # the number of nodes and edges
    acc_arr = np.zeros(int(var))
    logger.info('number of candidate node/edges:{}'.format(var))
    logger.info('number of nodes:{}'.format(num_node))
    logger.info('number of edges:{}'.format(var - num_node))

    edge_ind = []
    for ind in range(num_node):
        k = ind + 1
        while k < num_node:
            edge_ind.append((ind,k))
            k += 1

    for i in range(var): # for each variable
        #adj_ls, features, labels, sequences, proteases, labelorder, train_mask, val_mask, test_mask = load_data(args.dataset, is_test=args.test_dataset, norm_type=is_cheby)
        tmp_adj_ls = all_graph[full_test_mask].clone()
        tmp_features = all_features[full_test_mask].clone()
        tmp_y = ys[full_test_mask].clone()
        OOB_mask = np.asarray([1 for i in tmp_features],dtype=np.bool)
        if i < num_node:
            for j in range(tmp_features.shape[2]): # for each node feature
                np.random.shuffle(tmp_features[:,i,j].cpu().numpy())
            print("Shuffling Node Feature: {}".format(i+1))
        else:
            for j in range(tmp_adj_ls.shape[3]): # for each edge feature
                edge_node = i - num_node
                np.random.shuffle(tmp_adj_ls[:, edge_ind[edge_node][0], edge_ind[edge_node][1],j])
                after_shuffle = tmp_adj_ls[:,edge_ind[edge_node][0], edge_ind[edge_node][1], j]
                tmp_adj_ls[:,edge_ind[edge_node][1], edge_ind[edge_node][0], j] = after_shuffle
            print("Shuffling Edge Feature: {}".format(edge_node + 1))
        logit_vi, acc_vi = test(X=tmp_features, graph=tmp_adj_ls, y=tmp_y, testmask=OOB_mask, model_for_test=trained_model, \
             hidden1=hidden1, linear=linear, learning_rate=learning_rate, \
             weight_decay=weight_decay, batch_size=batch_size, dropout=dropout, path_save=path_save)
        if i < num_node:
            logger.info("Node {:04d} | Test Accuracy: {:.4f}".format(i+1, acc_vi))
        else:
            logger.info("Edge {:04d} | Test Accuracy: {:.4f}".format(i-num_node+1, acc_vi))
        acc_arr[i] = acc_vi
    return acc_arr 

is_cheby = True if args.model == 'chebyshev' else False
adj_ls, features, labels, sequences, proteases, labelorder, train_mask, test_mask = load_data(args.dataset, is_test=args.test_dataset, norm_type=is_cheby)
cheby_params = args.max_degree if args.model == 'chebyshev' else None
weight_mode = args.weight
no_energy = True if args.no_energy == True else False
dim_des = args.dim_des
tmp_mask = np.array([(not idx) for idx in test_mask], dtype=np.bool)

# Size of Different Sets
logger.info("|Training| {},|Testing| {}".format(np.sum(tmp_mask), np.sum(test_mask)))

model = GCN(nnode=features.shape[1],
            nfeat=features.shape[2],
            mfeat=adj_ls.shape[3],
#            ngcn=args.ngcn,
            hidden1=args.hidden1,
            depth=args.depth, 
#            hidden2=args.hidden2,
            natt=args.att, # one layer
            linear=args.linear,
            weight=weight_mode,
            is_des=dim_des,
            nclass=labels.shape[1],
            dropout=args.dropout,
            cheby=cheby_params)
logger.info(model)
logger.info('Number of parameters: {}'.format(count_parameters(model)))

batch_size = args.batch_size

# load trained model and test first
logit_test, acc_test = test(X=features, graph=adj_ls, y=labels, testmask=test_mask, model_for_test=model, hidden1=args.hidden1, linear=args.linear, learning_rate=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size, dropout=args.dropout, path_save=args.save)
logger.info("Original Test Accuracy is:" + str(acc_test))
#pkl.dump(logit_test,open(os.path.join(args.save, 'logits_test_vi'),'wb'))

if args.importance == True:
    acc_vi_arr = importance(all_features=features, all_graph=adj_ls, ys=labels, \
              full_test_mask=test_mask, trained_model=model, hidden1=args.hidden1, \
              linear=args.linear, learning_rate=args.lr, \
               weight_decay=args.weight_decay, batch_size=args.batch_size, \
               dropout=args.dropout, path_save=args.save)
    df = pd.DataFrame(acc_vi_arr, index = range(acc_vi_arr.shape[0])) # node + edge
    df.to_csv(os.path.join(args.save,"variable_importance.csv"))
