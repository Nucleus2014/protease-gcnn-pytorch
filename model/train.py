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
#parser.add_argument('--hidden2', type=int, default=10, help='Number of hidden units for edge as nodes')
parser.add_argument('--depth', type=int, default=10, help='Number of gcnn layers')
parser.add_argument('--dim_des',action='store_true',default=False)
parser.add_argument('--cv', type=int, default=7, help='N-fold cross validation.')
parser.add_argument('--no_energy', action='store_true', default=False, help='check if no energy features')
#parser.add_argument('--ngcn', type=str, default='[10,10]',help='Set of number of hidden units for gcn layers')
#parser.add_argument('--nfull', type=str, default='[]')
parser.add_argument('--att', type=int, default=30, help='the dimension of weight matrices for key and query')
parser.add_argument('--linear', type=int, default=0)
parser.add_argument('--dropout', type=float, default=0.1,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--test_dataset',type=str, default=None, help = "test datset string")
parser.add_argument('--dataset',type=str, help='input dataset string')
parser.add_argument('--model', type = str, default = 'gcn',choices=['gcn','chebyshev'])
parser.add_argument('--weight', type=str, default='pre',choices=['pre','post'])
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
    
# Load data
is_cheby = True if args.model == 'chebyshev' else False
cheby_params = args.max_degree if args.model == 'chebyshev' else None
cv_fold = args.cv
no_energy = True if args.no_energy == True else False
weight_mode = args.weight # 'pre' or 'post'
dim_des = args.dim_des
if args.test_dataset != None:
    test_dataset = args.test_dataset
else:
    if args.save_test == False:
        test_dataset = args.dataset
    else:
        test_dataset = None
adj_ls, features, labels, sequences, proteases, labelorder, train_mask, test_mask = load_data(args.dataset, is_test=test_dataset, norm_type=is_cheby, cv=cv_fold, noenergy = no_energy)

# Size of Different Sets
print("|Training| {}, |Validation| {}, |Testing| {}".format(np.sum(train_mask[0:-1], axis=0, dtype=bool), np.sum(train_mask[-1]), np.sum(test_mask)))

batch_size = args.batch_size
epochs_num = args.epochs
accumulated_acc = 0

# Model and optimizer
model = GCN(nnode=features.shape[1],
            nfeat=features.shape[2],
            mfeat=adj_ls.shape[3],
            hidden1=args.hidden1,
            linear=args.linear,
            depth=args.depth,
            natt=args.att, # one layer
            nclass=labels.shape[1],
            dropout=args.dropout,
            weight=weight_mode,
            is_des=dim_des,
            cheby=cheby_params).to(device)
logger.info(model)
logger.info('Number of parameters: {}'.format(count_parameters(model)))

#criterion = nn.NLLLoss().to(device)
criterion = nn.CrossEntropyLoss().to(device)

optimizer = optim.Adam(model.parameters(),lr=args.lr, weight_decay=args.weight_decay)

print("Total number of forward processes:" + str(args.epochs * args.batch_size))

if args.save_validation == True:
    val_df = np.zeros([epochs_num * cv_fold,4]) #train_loss, val_loss, train_acc, val_acc
if args.save_test == True: # save test dataset
    test_index = np.where(test_mask == True)[0]
    pkl.dump(test_index, open(os.path.join(args.save,"ind." + args.dataset + ".index"),'wb'))
    np.savetxt("ind.{}.test.index".format(args.dataset),test_index, fmt="%d")

# save initialize parameters
torch.save({'state_dict': model.state_dict()}, os.path.join(args.save, 'temp_hidden_' + str(args.hidden1) + '_linear_' + str(args.linear) + '_lr_'+str(args.lr)+'_wd_'+str(args.weight_decay)+'_bs_'+str(args.batch_size)+'_dt_' + str(args.dropout) + '.pth'))

for fold in range(cv_fold):
    # load initialized parameters
    checkpoint = torch.load(os.path.join(args.save, 'temp_hidden_' + str(args.hidden1) + '_linear_' + str(args.linear) + '_lr_'+str(args.lr)+'_wd_'+str(args.weight_decay)+'_bs_'+str(args.batch_size)+ '_dt_' + str(args.dropout) + '.pth'))
    model.load_state_dict(checkpoint['state_dict'])
    tmp_mask = train_mask.copy()
    if args.cv == 0:
        train_mask_tmp = tmp_mask[-1]
        val_mask = tmp_mask[0]
    else:
        val_mask = tmp_mask.pop(fold)
        train_mask_tmp = np.sum(tmp_mask, axis=0, dtype=bool)

    best_acc = 0
    batch_time_meter = RunningAverageMeter()
    end = time.time()
    
    for epoch in range(epochs_num):
        n = 0
        for batch_mask in get_batch_iterator(train_mask_tmp, batch_size):
            optimizer.zero_grad()
            n = n + 1
            #print('{0}th fold:: this is the {1}th batch'.format(fold+1, n))
            x = features[batch_mask].to(device)
            y = labels[batch_mask]
            y = torch.argmax(y,axis=1).to(device)
            adj = adj_ls[batch_mask].to(device)
            logits = model(x, adj)
            loss = criterion(logits,y)
            loss.backward()
            optimizer.step()
            train_acc = accuracy(logits, y)
            batch_time_meter.update(time.time() - end)
            end = time.time()
        with torch.no_grad():
            logits_val = model(features[val_mask], adj_ls[val_mask])
            loss_val = criterion(logits_val,torch.argmax(labels[val_mask],axis=1))
            val_acc = accuracy(logits_val, torch.argmax(labels[val_mask],axis=1))
            if args.save_validation == True:
                val_df[epoch + epochs_num * fold, :] = np.array([loss, train_acc, loss_val, val_acc])
            print("{0} fold:: train accuracy for {1}th epoch is: {2}".format(fold+1, epoch, train_acc))
            print("{0} fold:: validation accuracy for {1}th epoch is: {2}".format(fold+1, epoch, val_acc))
            if val_acc > best_acc:
    #            if val_acc > train_acc:
                torch.save({'fold': fold+1, 'epoch': epoch,'state_dict': model.state_dict(), 'args': args}, os.path.join(args.save, 'current.pth'))
                best_acc = val_acc
                best_epo = epoch
                logger.info(
                 "Fold {:04d} | Epoch {:04d} | Time {:.3f} ({:.3f}) | "
                 "Val Acc {:.4f}".format(
                     fold+1, epoch, batch_time_meter.val, batch_time_meter.avg, val_acc
                 )
                )
    accumulated_acc += best_acc
logger.info("{:04d} CV fold average accuracy is: {:.3f}".format(args.cv, accumulated_acc/args.cv))
os.remove(os.path.join(args.save, 'temp_hidden_' + str(args.hidden1) + '_linear_' + str(args.linear) +'_lr_'+str(args.lr)+'_wd_'+str(args.weight_decay)+'_bs_'+str(args.batch_size)+ '_dt_' + str(args.dropout) + '.pth'))
if args.save_validation == True:
    val_df = pd.DataFrame(val_df, columns = ["train_loss","train_acc","val_loss","val_acc"])
    val_df.to_csv(os.path.join(args.save, args.dataset + '_validation.csv'))

#test()

    
