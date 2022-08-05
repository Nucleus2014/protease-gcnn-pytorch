# This script is to test GCNN with already-trained gcnn model
# Author: Changpeng Lu
# Usage:
# python test.py --dataset HCV_binary_10_ang_aa_sinusoidal_encoding_4_energy_7_energyedge_5_hbond --test_dataset HCV_binary_10_ang_aa_sinusoidal_encoding_2_energy_7_energyedge_5_hbond --epochs 500 --hidden1 20 --depth 2 --linear 1024 --att 0 --model gcn --batch_size 500 --lr $tmp_lr --dropout $tmp_dt --weight_decay $tmp_wd --save 'outputs/tt/HCV_binary_10_ang_aa_sinusoidal_encoding_4_energy_7_energyedge_5_hbond/bs_500/'
# Train and test each epoch for this new version, instead of testing only after all training are done. Changes below aren't applied to wider_deeper and more epoch trials with the base setting. Replace Adam with SGD, also add lr_scheduler. Also calculate average train accuracy and loss instead of the last batch. Also, we use earlystop to let the model train enough epochs. 1) if test accuracy always go smaller, then the model will stop; 2) if the test accuracy always the same as the former accuracy, then it means converges, then the model will stop as well.

from __future__ import division
from __future__ import print_function

import os
import time
import logging
import argparse
import numpy as np
import random

from comet_ml import Experiment
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
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
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
parser.add_argument('--energy_only', action='store_true', default=False)
parser.add_argument('--scale_type', choices = ['exp','minmax'], default='exp')
parser.add_argument('--test_dataset',type=str, default=None)
parser.add_argument('--val_dataset', type=str, default=None)
parser.add_argument('--dataset',type=str, help='input dataset string')
parser.add_argument('--model', type = str, default = 'gcn',choices=['gcn','chebyshev'])
parser.add_argument('--max_degree',type=int, default = 3, help='number of supports')
parser.add_argument('--batch_size',type=int, default=8)
parser.add_argument('--weight', type=str, default='pre',choices=['pre','post'])
parser.add_argument('--dim_des',action='store_true',default=False)
parser.add_argument('--save', type=str, default='./experiment1')
args = parser.parse_args()

makedirs(args.save)
logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
logger.info(args)
experiment = Experiment(project_name = args.dataset, api_key="yOMD8snU8WrOgVJM6jTErziMh", workspace="tevtrisplit")
hyper_params = {"seed": args.seed, "weight_decay": args.weight_decay, "learning_rate": args.lr, "dropout": args.dropout, "batch_size": args.batch_size}
experiment.log_parameters(hyper_params)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
# cheby is no longer useful
is_cheby = True if args.model == 'chebyshev' else False
no_energy = True if args.no_energy == True else False
if args.val_dataset != None:
    logger.info('TripleSplit!')
    adj_ls, features, labels, sequences, proteases, labelorder, train_mask, val_mask, test_mask = load_data(args.dataset, is_test=args.test_dataset, is_val=args.val_dataset, norm_type=True, scale_type=args.scale_type, test_format = 'index', energy_only = args.energy_only)
    logger.info("|Training| {},|Validation| {}, |Testing| {}".format(np.sum(train_mask), np.sum(val_mask), np.sum(test_mask)))
    tmp_mask = train_mask
else:
    adj_ls, features, labels, sequences, proteases, labelorder, train_mask, val_mask = load_data(args.dataset, is_test=args.test_dataset, is_val=args.val_dataset, norm_type=True, scale_type=args.scale_type, test_format = 'index', energy_only = args.energy_only) #scale_type determines node feature scale
    tmp_mask = np.array([(not idx) for idx in val_mask], dtype=np.bool)
    # Size of Different Sets
    logger.info("|Training| {},|Testing| {}".format(np.sum(tmp_mask), np.sum(val_mask)))

cheby_params = args.max_degree if args.model == 'chebyshev' else None
weight_mode = args.weight
dim_des = args.dim_des

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
#optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
nepoch = args.epochs #willbe useless if set earlystop
#scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, int(nepoch / 10)) 
#scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10000)

best_acc = 0
print("Total number of forward processes:" + str(args.epochs * args.batch_size))
#patience = 100
#last_loss = 100
#triggertimes = 0
with experiment.train():
    for i in range(nepoch):
        #i = 0 # epoch
    #converge_detect = 0
    #former_acc = 0
    #while True:
        #n = 0
        model.train()
        tmp_accs = []
        tmp_losses = []
        for batch_mask in get_batch_iterator(tmp_mask, batch_size):
            optimizer.zero_grad()
            #n = n + 1
            x = features[batch_mask]
            y = labels[batch_mask]
            y = torch.argmax(y,axis=1)
            adj = adj_ls[batch_mask]
            logits = model(x, adj)
            loss = criterion(logits,y)
            train_acc = accuracy(logits,y)
            loss.backward()
            optimizer.step()
            tmp_losses.append(loss.item())
            tmp_accs.append(train_acc.item())
#        scheduler.step()
        #train_acc = accuracy(logits, y) # only record the last batch accuracy for each epoch
        experiment.log_metric("epoch_loss", sum(tmp_losses) / len(tmp_losses), step=i+1)
        experiment.log_metric("epoch_accuracy", sum(tmp_accs) / len(tmp_accs), step=i+1)
        #print("train accuracy for {0}th epoch is: {1}".format(i+1, train_acc))
#    print("train loss for {0}th epoch is : {1}".format(i+1, loss))
        print("epoch: " + str(i+1))
        print("train_loss: " + str(sum(tmp_losses) / len(tmp_losses))) #loss.item()))
        print("train_acc: " + str(sum(tmp_accs) / len(tmp_accs))) #train_acc.item()))
        with torch.no_grad():
            with experiment.validate():
                model.eval()
                logits_test = model(features[val_mask], adj_ls[val_mask])
                val_loss = criterion(logits_test, torch.argmax(labels[val_mask],axis=1))
                val_acc = accuracy(logits_test, torch.argmax(labels[val_mask],axis=1))
            print("val_loss: " + str(val_loss.item()))
            print("val_acc: " + str(val_acc.item()))
            experiment.log_metric("val_accuracy", val_acc.item(), step=i+1)
            experiment.log_metric("val_loss", val_loss.item(), step=i+1)
        #    if test_acc > max_acc:
        #logits_test_fin = logits_test
         #       max_acc = test_acc
#        logger.info("Test accuracy is:" + str(test_acc))
        
        if val_acc > best_acc:
            torch.save({'epoch': i+1,'state_dict': model.state_dict()}, os.path.join(args.save, 'model_for_test_seed_' + str(args.seed) + '_hidden_' + str(args.hidden1) + '_linear_' + str(args.linear) +'_lr_'+str(args.lr)+'_wd_'+str(args.weight_decay)+'_bs_'+str(args.batch_size)+ '_dt_' + str(args.dropout) + '.pth'))
            pkl.dump(logits_test,open(os.path.join(args.save, 'logits_val_seed_' + str(args.seed) + '_hidden_' + str(args.hidden1) + '_linear_' + str(args.linear) +'_lr_'+str(args.lr)+'_wd_'+str(args.weight_decay)+'_bs_'+str(args.batch_size)+ '_dt_' + str(args.dropout)),'wb'))
            best_acc = val_acc
            best_epo = i
        #if abs(test_acc - former_acc) < 0.0001:
        #    converge_detect += 1
        #    if converge_detect == 100:
        #        break
        #elif test_acc < former_acc :
        #    overfit_detect += 1
        #    if overfit_detect >= 100:
        #        break
        #i += 1
logger.info("best_val_acc: " + str(best_acc.item()))
        #logger.info(
        #     "Epoch {:04d} | "
        #     "Best Acc {:.4f}".format(
        #         best_epo, best_acc
        #     ))

# test
if args.val_dataset != None:
    checkpoint = torch.load(os.path.join(args.save, 'model_for_test_seed_' + str(args.seed) + '_hidden_' + str(args.hidden1) + '_linear_' + str(args.linear) +'_lr_'+str(args.lr)+'_wd_'+str(args.weight_decay)+'_bs_'+str(args.batch_size)+ '_dt_' + str(args.dropout) + '.pth'))
    print("best epoch is:" + str(checkpoint['epoch']))
    model.load_state_dict(checkpoint['state_dict'])
    max_acc = 0
    with torch.no_grad():
        model.eval()
        logits_test = model(features[test_mask], adj_ls[test_mask])
        test_acc = accuracy(logits_test, torch.argmax(labels[test_mask],axis=1))
        #if test_acc > max_acc:
        #logits_test_fin = logits_test
         #       max_acc = test_acc
        logger.info("Test accuracy is:" + str(test_acc))
        pkl.dump(logits_test,open(os.path.join(args.save, 'logits_test_seed_' + str(args.seed) + '_hidden_' + str(args.hidden1) + '_linear_' + str(args.linear) +'_lr_'+str(args.lr)+'_wd_'+str(args.weight_decay)+'_bs_'+str(args.batch_size)+ '_dt_' + str(args.dropout)),'wb'))

    
