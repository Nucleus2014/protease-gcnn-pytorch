# This script contains utility functions for gcnn
# Author: Changpeng Lu

import numpy as np
import pickle as pkl
import networkx as nx
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import scipy.sparse as sp
import os
import torch
import logging
from sklearn import preprocessing

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)
    
def get_batch_iterator(mask, batch_size):
    # Batch size iterator, returns list of masks.
    train_indices = [i for (i,boolean) in enumerate(mask) if boolean == True]
    np.random.shuffle(train_indices)
    mask_ls, i = [], 0
    while i < len(train_indices):
        m = np.zeros(shape=mask.shape, dtype=np.bool)
        if i + batch_size <= len(train_indices):
            m[train_indices[i:i+batch_size]] = True
            mask_ls.append(m)
        else:
            m[train_indices[i:]] = True
            mask_ls.append(m)
        i += batch_size
    return mask_ls

#def load_data(dataset_str, is_test=None, norm_type=False, noenergy=False, cv=7):
#    protease = dataset_str.replace("protease_","")
#    protease = protease.split("_selector")[0]
#    cwd = os.getcwd()
    #os.chdir("..")
#    print(cwd)
#    names = ['x', 'y', 'graph', 'sequences', 'labelorder']
#    objects = []
#    for i in range(len(names)):
#        with open("../data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
#            if sys.version_info > (3, 0):
#                objects.append(pkl.load(f, encoding='latin1'))
#            else:
#                objects.append(pkl.load(f))

 #   features, y_arr, adj_ls, sequences, labelorder = tuple(objects)
 #   #os.chdir(cwd)
  #  proteases = [protease for x in sequences]

    # turn edges into nodes
    # node feature matrix

    # Split all datasets into testing, training, and validation. The split of this data is fixed for each dataset
    # because the numpy seed is fixed, currently the breakdown is train: 60, validation: 10, test: 30
#    idx = [y_ind for y_ind in range(y_arr.shape[0])]
#    np.random.shuffle(idx)
#    cutoff_1 = int((cv-1)*len(idx)/10)
#    cutoff_2 = int(cv*len(idx)/10)
#    idx_train = idx[:cutoff_1]
#    idx_val = idx[cutoff_1:cutoff_2]
#    idx_test = idx[cutoff_2:]
#    idx_train, idx_val, idx_test = np.sort(idx_train), np.sort(idx_val), np.sort(idx_test)
    # make logical indices (they are the size BATCH)
#    train_mask = sample_mask(idx_train, y_arr.shape[0])
#    val_mask = sample_mask(idx_val, y_arr.shape[0])
#    test_mask = sample_mask(idx_test, y_arr.shape[0])

 #   if is_test != None:
#        test_idx_reorder = parse_index_file("../data/ind.{}.test.index".format(is_test))
#        test_idx_range = np.sort(test_idx_reorder)
#        test_mask = np.zeros(len(train_mask), dtype=np.bool)
#        test_mask[test_idx_range] = True
        # make train test split
#        tmp_mask = test_mask.copy()
#        train_mask = np.array([(not idx) for idx in test_mask],dtype=np.bool)
        
#        train_ind = set(range(len(train_mask))) - set(test_idx_range)
#        val_ind = np.random.choice(list(train_ind), int(len(train_mask)*0.1),replace=False)
        
#        val_mask = np.zeros(len(train_mask), dtype=np.bool)
#        val_mask[val_ind] = True
#        tmp_mask[val_ind] = True
#        train_mask = np.array([(not idx) for idx in tmp_mask], dtype=np.bool)

#    if cv != 0:
#        idx = np.where(train_mask == True)[0]
#        np.random.shuffle(idx)
#        cv_aux = np.array_split(idx, cv-1)
#        train_mask = [val_mask]
#        for i in cv_aux:
#            train_mask.append(sample_mask(i, y_arr.shape[0]))
        #train_mask.append(val_mask)
#    else:
#        train_mask = [val_mask, train_mask]
#    features, adj_ls = rebuild_mat(features,adj_ls)
#    if noenergy == False:
#        features = transform(features,ind='(-8,-1)') #only energy features
#        adj_ls = transform(adj_ls, ind='(0,6)') # only energy features

#    adj_ls = normalize(adj_ls)
    
#    features = torch.FloatTensor(np.array(features))
#    y_arr = torch.LongTensor(y_arr)
#    adj_ls = torch.FloatTensor(np.array(adj_ls))

#    return adj_ls, features, y_arr, sequences, proteases, labelorder, train_mask, test_mask


#def normalize(mx, norm_type=False): #norm_type = true if chebyshev  
    # normalize matrix using partition probability because weights * features should avoid features = 0
    # but distribution range is small
    # so just do exp(-x) in transformation and then normalize edge using row normalization
#    try:
#        mx = mx.numpy().copy()
#    except AttributeError:
#        pass
#    for b in range(mx.shape[0]):
#        for m in range(mx.shape[3]):
#            mx[b][:,:,m] += np.eye(mx.shape[1])
#    rowsum = np.array(mx.sum(2)) # Here starts to multiply -1/2 degree matrix on both sides of transformed adjacency matrix
#    r_inv = np.power(rowsum, -0.5)
#    r_inv[np.isinf(r_inv)] = 0.
#    for b in range(mx.shape[0]):
#        for m in range(mx.shape[3]):
#            r_mat_inv = sp.diags(r_inv[b,:,m])
#            mx[b,:,:,m] = r_mat_inv * mx[b,:,:,m] * r_mat_inv
#            if norm_type == True:
#                mx[b,:,:,m] = np.identity(mx.shape[1]) - mx[b,:,:,m] # normalized laplacian
#    return mx


def transform(mat, scale_type, ind = 'all'): # ind saves indices that needs to be transformed, like '(22,29)'
    """exp(-x) for features"""
    try:
        mat = mat.numpy().copy()
    except AttributeError:
        pass
    if scale_type == 'exp':
        if ind == 'all':
            return np.exp(-mat)
        else:
            ind_ends = [int(x) for x in ind.strip('()').split(',')]
            ind_list = np.arange(ind_ends[0],ind_ends[1])
            mat[...,ind_list] = np.exp(-mat[...,ind_list])
    elif scale_type == 'minmax':
        min_max_scaler = preprocessing.MinMaxScaler()
        tmp = np.zeros_like(mat)
        for i in range(mat.shape[2]):
            tmp[:,:,i] = min_max_scaler.fit_transform(mat[:,:,i])
        mat = tmp
    return mat

def load_input(input_dataset_str, dataname_list, input_type = 'train',path=None): #input_type: 'train' or 'test'
    objects = []
    if input_type == 'train':
        path_str = '../data/ind.{}.{}'
    elif input_type == 'test':
        path_str = path + '/ind.{}.{}' #'../data/ind.{}.test.{}'
    for i in range(len(dataname_list)):
        with open(path_str.format(input_dataset_str, dataname_list[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    return objects

def load_data(dataset_str, is_test=None, is_val=None, norm_type=True, scale_type='exp', noenergy=False, cv=0, test_format = 'individual', energy_only=False, test_path=None):
    #test_format only accepts 'index' or 'individual' or None
    cwd = os.getcwd()
    if test_format == 'individual': # this individual set will not be supposed to have y and model has been trained
        names = ['graph', 'x', 'sequences']#, 'proteases']
        test_adj, test_features, test_sequences = tuple(load_input(dataset_str, 
                                                                   names, input_type='test', path=test_path))
        # add normalize here
        #labelorder = tuple(load_input(dataset_str, ['labelorder']))[0]
        labelorder = ['CLEAVED','UNCLEAVED']
        if energy_only==True:
            test_features = test_features[:,:,20:]
        if noenergy == False:
            test_features = transform(test_features, scale_type, ind = '(-8,-1)') # accept all features in this case except the last one
            test_adj = transform(test_adj, 'exp', ind = '(0,6)')
        if norm_type == True:
            test_adj = normalize(test_adj)
        test_features = torch.FloatTensor(np.array(test_features))
        test_adj = torch.FloatTensor(np.array(test_adj))
 
        return test_adj, test_features, test_sequences, labelorder#, test_proteases, labelorder
    else:
        # loading training dataset
        names = ['x', 'y', 'graph', 'sequences', 'proteases', 'labelorder']
        features, y_arr, adj_ls, sequences, proteases, labelorder = tuple(load_input(dataset_str, names, input_type='train'))
    
        # Split all datasets into testing, training, and validation. The split of this data is fixed for each dataset
        # because the numpy seed is fixed, currently the breakdown is train: 60, validation: 10, test: 30
        # Prefer to downgrade the size of testset from 30% to 10%  //Feb 2021
        idx = np.arange(y_arr.shape[0])
        np.random.shuffle(idx)
        if is_test == None: # testset is not individual, which is 10% of the current dataset, randomly selected
            cutoff_2 = int(0.9 * len(idx)) # 10% of the benchmark set as testing data
            idx_test = idx[cutoff_2:]
            idx_train = idx[:cutoff_2]
        else:
            if is_val == None:
                test_idx_reorder = parse_index_file("../data/ind.{}.test.index".format(is_test))
                idx_test = np.sort(test_idx_reorder)
                idx_train = idx[np.array([x not in idx_test for x in idx])]
            else:
                test_idx_reorder = parse_index_file("../data/ind.{}.trisplit.test.index".format(is_test))
                idx_test = np.sort(test_idx_reorder)
                val_idx_reorder = parse_index_file("../data/ind.{}.trisplit.val.index".format(is_val))
                idx_val = np.sort(val_idx_reorder)
                val_mask = sample_mask(idx_val, y_arr.shape[0])
                idx_train = idx[np.array([x not in idx_test and x not in idx_val for x in idx])]
        train_mask = sample_mask(idx_train, y_arr.shape[0])
        test_mask = sample_mask(idx_test, y_arr.shape[0])

        #print(np.where(test_mask)[0])
        if cv != 0: # if cv != 0, means validation set exists. Then the former train indice set should be separated based on the number of folds
            cv_aux = np.array_split(idx_train, cv)
            train_mask = []
            for i in cv_aux:
                train_mask.append(sample_mask(i, y_arr.shape[0]))
        if energy_only==True:
            features = features[:,:,20:]
        if noenergy == False:
            features = transform(features, scale_type, ind = '(-8,-1)') # accept all features in this case except the last one
            adj_ls = transform(adj_ls, 'exp', ind = '(0,6)') 
        if norm_type == True:
            adj_ls = normalize(adj_ls)
    features = torch.FloatTensor(np.array(features))
    y_arr = torch.LongTensor(y_arr)
    adj_ls = torch.FloatTensor(np.array(adj_ls))
    if is_val == None:
        return adj_ls, features, y_arr, sequences, proteases, labelorder, train_mask, test_mask
    else:
        return adj_ls, features, y_arr, sequences, proteases, labelorder, train_mask, val_mask, test_mask


def normalize(mx): 
    # normalize matrix when loading data using partition probability because weights * features should avoid features = 0
    # but distribution range is small
    # so just do exp(-x) in transformation and then normalize edge using row normalization
    try:
        mx = mx.numpy().copy()
    except AttributeError:
        pass
    for b in range(mx.shape[0]):
        for m in range(mx.shape[3] - 2, mx.shape[3]):
            mx[b][:,:,m] += np.eye(mx.shape[1])
#     rowsum = np.array(mx.sum(2)) # Here starts to multiply -1/2 degree matrix on both sides of transformed adjacency matrix
#     r_inv = np.power(rowsum, -0.5)
#     r_inv[np.isinf(r_inv)] = 0.
#     for b in range(mx.shape[0]):
#         for m in range(mx.shape[3]):
#             r_mat_inv = sp.diags(r_inv[b,:,m])
#             mx[b,:,:,m] = r_mat_inv * mx[b,:,:,m] * r_mat_inv
    return mx

def accuracy(output, labels):
    preds = np.argmax(output.cpu().detach().numpy(),axis=1)
    correct = torch.from_numpy(preds).eq(labels).double()
#    correct = np.sum(preds == labels)
    correct = correct.sum()
    return correct / labels.size()[0]


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

class RunningAverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val
        
def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger

def makedirs(dirname):
    try:
        os.makedirs(dirname)
    except FileExistsError:
        pass

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def chebyshev(x, adj, max_degree):
    out = torch.zeros(x.shape[0],x.shape[1],x.shape[2],max_degree)
    for batch in range(out.shape[0]):
        eigval_max = torch.from_numpy(np.linalg.eigvals(adj[batch,:,:].detach().numpy()).astype(float))
        til_adj = 2 * adj[batch,:,:] / eigval_max - torch.eye(adj.shape[1])
        out[batch,:,:,:] = torch.from_numpy(cheby(x[batch,:,:].detach().numpy(), til_adj.detach().numpy(), max_degree))
    return out

def cheby(x, adj, mdeg):
    out = np.zeros((x.shape[0],x.shape[1],mdeg))
    for i in range(mdeg):
        if i == 0:
            recur_1 = x
            out[:,:,0] = recur_1
        elif i == 1:
            recur_2 = np.matmul(adj,x)
            out[:,:,1] = recur_2
        else:
            recur = 2 * np.matmul(adj, recur_2) - recur_1
            recur_2 = recur
            recur_1 = recur_2
    return out
    
