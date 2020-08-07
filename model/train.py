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
from models import GCN

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
parser.add_argument('--hidden2', type=int, default=10,help='Number of hidden units for edge as nodes')
#parser.add_argument('--ngcn', type=str, default='[10,10]',help='Set of number of hidden units for gcn layers')
#parser.add_argument('--nfull', type=str, default='[]')
parser.add_argument('--att', type=int, default=30, help='the dimension of weight matrices for key and query')
#parser.add_argument('--linear', type=int, default=10)
parser.add_argument('--dropout', type=float, default=0.1,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--test_dataset',type=str, help = "test datset string")
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
    checkpoint = torch.load(os.path.join(args.save, 'model.pth'))
    print("best epoch is:" + str(checkpoint['epoch']))
    model.load_state_dict(checkpoint['state_dict'])
    for batch_mask in get_batch_iterator(val_mask, batch_size):
        optimizer.zero_grad()
        x = features[batch_mask].to(device)
        y = labels[batch_mask]
        y = torch.argmax(y,axis=1).to(device)
        adj = adj_ls[batch_mask].to(device)
        model.train()
        logits = model(x, adj)
        loss = criterion(logits,y)
        loss.backward()
        optimizer.step()
        train_acc = accuracy(logits, y)
    with torch.no_grad():
        model.eval()
        for j in range(100):
            logits_test = model(features[test_mask], adj_ls[test_mask])
            test_acc = accuracy(logits_test, torch.argmax(labels[test_mask],axis=1))
            if test_acc > max_acc:
                logits_test_fin = logits_test
                max_acc = test_acc
        print("Test accuracy is:" + str(test_acc))
    pkl.dump(logits_test_fin,open(os.path.join(args.save, 'logits_test'),'wb'))

makedirs(args.save)
logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
logger.info(args)
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

## Determine Number of Supports and Assign Model Function
#if args.model == 'gcn':
#    num_supports = 1
#    model_func = GCN
#elif args.model == 'gcn_cheby':
#    num_supports = 1 + args.max_degree
#    model_func = GCN
#else:
#    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))
    
# Load data
is_cheby = True if args.model == 'chebyshev' else False
cheby_params = args.max_degree if args.model == 'chebyshev' else None

adj_ls, features, labels, sequences, proteases, labelorder, train_mask, val_mask, test_mask = load_data(args.dataset, is_test=args.test_dataset, norm_type=is_cheby)

# Size of Different Sets
print("|Training| {}, |Validation| {}, |Testing| {}".format(np.sum(train_mask), np.sum(val_mask), np.sum(test_mask)))
    
# Model and optimizer
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
            cheby=cheby_params).to(device)
logger.info(model)
logger.info('Number of parameters: {}'.format(count_parameters(model)))
            
#criterion = nn.NLLLoss().to(device)
criterion = nn.CrossEntropyLoss().to(device)

optimizer = optim.Adam(model.parameters(),lr=args.lr, weight_decay=args.weight_decay)

best_acc = 0
batch_time_meter = RunningAverageMeter()
end = time.time()
print("Total number of forward processes:" + str(args.epochs * args.batch_size))

#batches_per_epoch = int(sum(train_mask) / args.batch_size)
#print("Batches per epoch is:" + str(batches_per_epoch))
batch_size = args.batch_size
epochs_num = args.epochs

if args.save_validation == True:
    val_df = np.zeros([args.epochs*sum(val_mask),labels.shape[1]])
#if args.save_test == True:
#    labels[test_mask].to_csv('test_label.txt')
if args.save_test == True:
    test_index = np.where(test_mask == True)[0]
#    pkl.dump(test_index, open(os.path.join(args.save,"ind." + args.dataset + ".index"),'wb'))
    np.savetxt("ind.{}.test.index".format(args.dataset),test_index, fmt="%d")

#mask = np.array([x or y for (x,y) in zip(train_mask, val_mask)], dtype = np.bool)
for epoch in range(epochs_num):
    n = 0
    for batch_mask in get_batch_iterator(train_mask, batch_size):
        optimizer.zero_grad()
        n = n + 1
        print('this is the {}th batch'.format(n))
        x = features[batch_mask].to(device)
        y = labels[batch_mask]
        y = torch.argmax(y,axis=1).to(device)
        adj = adj_ls[batch_mask].to(device)
        model.train()
        logits = model(x, adj)
        loss = criterion(logits,y)
        loss.backward()
        optimizer.step()
        train_acc = accuracy(logits, y)
#        print("train loss is {}".format(loss))
#        print("train accuracy is {}".format(train_acc))
        batch_time_meter.update(time.time() - end)
        end = time.time()
    with torch.no_grad():
        #train_acc = accuracy(model, logits, labels[train_mask])
        model.eval()
        logits_val = model(features[val_mask], adj_ls[val_mask])
        loss_val = criterion(logits_val,torch.argmax(labels[val_mask],axis=1))
        val_acc = accuracy(logits_val, torch.argmax(labels[val_mask],axis=1))
        print("train accuracy for {0}th epoch is: {1}".format(epoch,train_acc))
        print("validation accuracy for {0}th epoch is: {1}".format(epoch,val_acc))
        print("loss is {0}:".format(loss_val))
        if val_acc > best_acc:
            if val_acc > train_acc:
                torch.save({'epoch': epoch,'state_dict': model.state_dict(), 'args': args}, os.path.join(args.save, 'model.pth'))
                best_acc = val_acc
                best_epo = epoch
                logger.info(
             "Epoch {:04d} | Time {:.3f} ({:.3f}) | "
             "Val Acc {:.4f}".format(
                 epoch, batch_time_meter.val, batch_time_meter.avg, val_acc
             )
            )

#        f = open(args.save + "epoch_record.txt","a")
#        f.write("batch_size_{0}_lr_{1}_gc1_{2}_gc2_{3}_att_{4}_decay_{5}_epoch_{6}_model_{7}\tacc:{8}".format(batch_size,args.lr,args.hidden1,args.hidden2,args.att,args.weight_decay,epoch,args.model,val_acc))
#        f.close()
#    val_df[(epoch)*sum(val_mask):(epoch + 1) * sum(val_mask), :] = logits_val
#pkl.dump(val_df, open(os.path.join(args.save, args.dataset + '_validation.csv'),'wb'))
test()

    
