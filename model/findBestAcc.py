# Find best acc from logits calculation
# Author: Changpeng Lu

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
#os.chdir('/scratch/cl1205/protease-gcnn-pytorch/model')
#print(os.getcwd())
from utils import *
from models import *

def findBestAcc(dataset = 'HCV_WT_binary_10_ang_aa_energy_7_energyedge_5_hbond', 
                testset = 'HCV_WT_binary_10_ang_aa_energy_7_energyedge_5_hbond', 
                is_energy_only = True, hidden = 20, valset = None, 
               modelPath = '/scratch/cl1205/protease-gcnn-pytorch/model/outputs/tt_finalize_20220211/HCV_WT_binary_10_ang_energy_7_energyedge_5_hbond'):
    if valset == None:
        adj_ls, features, labels, sequences, proteases, labelorder, train_mask, test_mask = load_data(dataset, is_test=testset, norm_type=True, test_format = 'split', energy_only=is_energy_only, noenergy=False)
    else:
        adj_ls, features, labels, sequences, proteases, labelorder, train_mask, val_mask, test_mask = load_data(dataset, is_test=testset, is_val=valset, norm_type=True, test_format = 'split', energy_only=is_energy_only, noenergy=False)

    folder = modelPath # /projects/f_sdk94_1/PGCN/outputs/tt_finalize_20210413
    max_acc = [0,0,0]
    path_fin = ["","",""]
    for root, dirs, files in os.walk(folder):
        for name in files:
            if name.split('.')[-1] == 'pth':
                path = root + os.sep + name
                # /projects/f_sdk94_1/PGCN/outputs/tt_finalize_20210413/HCV_A171T_binary_10_ang_aa_energy_7_energyedge_5_hbond/bs_50/
                # model_for_test_seed_3_hidden_20_linear_0_lr_0.001_wd_0.005_bs_50_dt_0.3.pth
                bs = int(name.split('_')[-3])
                wd = float(name.split('_')[-5])
                lr = float(name.split('_')[-7])
                dt = float(name.split('_')[-1][0:-4])
                seed = int(name.split('_')[4])
                model = GCN(nnode=features.shape[1],
                            nfeat=features.shape[2],
                            mfeat=adj_ls.shape[3],
                #            ngcn=args.ngcn,
                            hidden1=hidden,
                            depth=2,
                #            hidden2=args.hidden2,
                            natt=0, # one layer
                            linear=0,
                            weight='pre',
                            is_des=False,
                            nclass=2, #labels.shape[1],
                            dropout=dt,
                            cheby=None)
    
                logit_test, acc_test = test(X=features, graph=adj_ls, y=labels, testmask=test_mask, model_for_test=model, 
                                            hidden1=hidden, linear=0, learning_rate=lr, weight_decay=wd, batch_size=bs, 
                                            dropout=dt, 
                                            path_save=path, 
                                            new=False)
                if acc_test > max_acc[seed-1]:
                    logit_test_fin = logit_test
                    max_acc[seed-1] = acc_test
                    path_fin[seed-1] = path
    return logit_test_fin, max_acc, path_fin

def test(X, graph, y, testmask, model_for_test, hidden1, linear, learning_rate, weight_decay, batch_size, dropout, path_save,new=False):
    #checkpoint = torch.load(os.path.join(path_save, 'model_for_test_seed_' + str(args.seed) + '_hidden_' + str(hidden1) + '_linear_' + str(linear) +'_lr_'+str(learning_rate)+'_wd_'+str(weight_decay)+'_bs_'+str(batch_size)+ '_dt_' + str(dropout) + '.pth'))
    try:
        checkpoint = torch.load(path_save)
    except:
        print(path_save)
        return None,0
    
    model_for_test.load_state_dict(checkpoint['state_dict'])
    if new == False:
        X = X[testmask]
        graph = graph[testmask]
        y = y[testmask]
        #else:
        #    print('testmask is none. bad!')
        max_acc = 0
        with torch.no_grad():
            model_for_test.eval()
            #for j in range(100):
            logits_test = model_for_test(X, graph)
            test_acc = accuracy(logits_test, torch.argmax(y,axis=1))
#                 if test_acc > max_acc:
#                     logits_test_fin = logits_test
#                     max_acc = test_acc
        return logits_test, test_acc
    else:
        with torch.no_grad():
            model_for_test.eval()
            logits_test = model_for_test(X, graph)
            return logits_test

for i in ['WT','A171T','D183A','Triple','all']:
    logit, acc, path = findBestAcc('HCV_' + i + '_binary_10_ang_aa',
                                   'HCV_' + i + '_binary_10_ang_aa',
                                   False, 20,
                                   'HCV_' + i + '_binary_10_ang_aa',
                                   '/scratch/cl1205/protease-gcnn-pytorch/model/outputs/hcv_20220708_trisplit_seqOnly/HCV_' + i + '_binary_10_ang_aa')
    print(acc, path)

#for i in ['WT','A171T','D183A','Triple', 'all']:
    #logit, acc, path = findBestAcc('TEV_' + i + '_binary_10_ang_aa_energy_7_energyedge_5_hbond', 
    #                               'TEV_' + i + '_binary_10_ang_aa_energy_7_energyedge_5_hbond',
    #                               False, '/projects/f_sdk94_1/PGCN/TEV/WT/outputs/tt_finalize_energy_only/')
    #print(acc, path)

#    logit, acc, path = findBestAcc('HCV_' + i + '_binary_10_ang_aa_energy_7_energyedge_5_hbond',
#                                   'HCV_' + i + '_binary_10_ang_aa_energy_7_energyedge_5_hbond',
#                                   False, 
#                                   '/scratch/cl1205/protease-gcnn-pytorch/model/outputs/tt_finalize_20220211/HCV_' \
#                                   + i + '_binary_10_ang_aa_energy_7_energyedge_5_hbond/') 
                                    #'/projects/f_sdk94_1/PGCN/TEV/WT/outputs/tt_finalize_aa/')
#for i in ['all']: #['WT','A171T','D183A','Triple', 'all']:
#    logit, acc, path = findBestAcc('TEV_' + 'WT' + '_binary_10_ang_aa_energy_7_energyedge_5_hbond',
#                                   'TEV_' + 'WT' + '_binary_10_ang_aa_energy_7_energyedge_5_hbond',
#                                   False, 10, 
#                                  '/scratch/cl1205/protease-gcnn-pytorch/model/outputs/tt_tev_20220403/TEV_' \
#                                   + 'WT' + '_binary_10_ang_aa_energy_7_energyedge_5_hbond_epoch_' + i + '/')
#for model in ['all']:
#    logit, acc, path = findBestAcc('TEV_' + model + '_binary_10_ang_aa_energy_7_energyedge_5_hbond',
#                                   'TEV_' + model + '_binary_10_ang_aa_energy_7_energyedge_5_hbond',
#                                   True, 20,
#                                   'TEV_' + model + '_binary_10_ang_aa_energy_7_energyedge_5_hbond',
#                                   '/scratch/cl1205/protease-gcnn-pytorch/model/outputs/tt_tev_20220629_trisplit/TEV_' + model + '_binary_10_ang_energy_7_energyedge_5_hbond/')
#    print(acc, path)

#for model in ['all']:
#    logit, acc, path = findBestAcc('TEV_' + model + '_binary_10_ang_aa',
#                                   'TEV_' + model + '_binary_10_ang_aa',
#                                   False, 20,
#                                   'TEV_' + model + '_binary_10_ang_aa',
#                                   '/scratch/cl1205/protease-gcnn-pytorch/model/outputs/tt_tev_20220629_trisplit/TEV_' + model + '_binary_10_ang_aa_/')
#    print(acc, path)
