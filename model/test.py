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
parser.add_argument('--importance',action='store_true', default = False, help='Whether calculate each variable's importance.')
args = parser.parse_args()

makedirs(args.save)
logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
logger.info(args)
# test
def test():
    checkpoint = torch.load(os.path.join(args.save, 'model_for_test_hidden_' + str(args.hidden1) + '_linear_' + str(args.linear) +'_lr_'+str(args.lr)+'_wd_'+str(args.weight_decay)+'_bs_'+str(args.batch_size)+ '_dt_' + str(args.dropout) + '.pth'))
    print("best epoch is:" + str(checkpoint['epoch']))
    model.load_state_dict(checkpoint['state_dict'])
    max_acc = 0
    with torch.no_grad():
        model.eval()
        for j in range(100):
            logits_test = model(features[test_mask], adj_ls[test_mask])
            test_acc = accuracy(logits_test, torch.argmax(labels[test_mask],axis=1))
            if test_acc > max_acc:
                logits_test_fin = logits_test
                max_acc = test_acc
        logger.info("Test accuracy is:" + str(test_acc))
    pkl.dump(logits_test_fin,open(os.path.join(args.save, 'logits_test'),'wb'))

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

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=args.lr, weight_decay=args.weight_decay)
nepoch = args.epochs
best_acc = 0
print("Total number of forward processes:" + str(args.epochs * args.batch_size))
for i in range(nepoch):
    n = 0
    for batch_mask in get_batch_iterator(tmp_mask, batch_size):
        optimizer.zero_grad()
        n = n + 1
        x = features[batch_mask]
        y = labels[batch_mask]
        y = torch.argmax(y,axis=1)
        adj = adj_ls[batch_mask]
        logits = model(x, adj)
        loss = criterion(logits,y)
        loss.backward()
        optimizer.step()
        train_acc = accuracy(logits, y)
    print("train accuracy for {0}th epoch is: {1}".format(i+1, train_acc))
    print("train loss for {0}th epoch is : {1}".format(i+1, loss))
    if train_acc > best_acc:
        torch.save({'epoch': i+1,'state_dict': model.state_dict()}, os.path.join(args.save, 'model_for_test_hidden_' + str(args.hidden1) + '_linear_' + str(args.linear) +'_lr_'+str(args.lr)+'_wd_'+str(args.weight_decay)+'_bs_'+str(args.batch_size)+ '_dt_' + str(args.dropout) + '.pth'))
        print('save successfully')
        best_acc = train_acc
        best_epo = i
        logger.info(
             "Epoch {:04d} | "
             "Best Acc {:.4f}".format(
                 best_epo, best_acc
             ))
test()
    
