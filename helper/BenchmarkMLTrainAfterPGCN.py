# This script is to train and test ml models
# Author: Changpeng Lu

from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
import tensorflow as tf
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import os
import pickle as pkl
from sklearn import preprocessing
import argparse

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('-data', type=str, help='data name')
parser.add_argument('-model', type=str, help='svm or ann or classic')
parser.add_argument('-feature', type=str, choices=['complete','seq','energy'])
parser.add_argument('-save', type=str, default='./experiment1')
args = parser.parse_args()

makedirs(args.save)

def trainSeqOnly(dataset, save = '/scratch/cl1205/ml-cleavage/outputs/seqOnly_20220217', model = 'logistic_regression',
                encoding = 'energy', split=2):
    classes = 2
    if split == 2:
        X_train, y_train, X_test, y_test = load_data(dataset, classes)
    elif split == 3:
        X_train, y_train, X_val, y_val, X_test, y_test = load_data(dataset, classes, 3)

    energy_indices = []
    seq_indices = []
    if dataset.find('TEV_all') != -1:
        for i in range(X_train.shape[1]):
            if i >= 1316: #1326:
                energy_indices.append(i)
            else:
                if i % 28 >= 20: # if having identifier, need to minus 10
                    energy_indices.append(i)
                else:
                    seq_indices.append(i)
    if dataset.find('HCV') != -1:
        for i in range(X_train.shape[1]):
            if i >= 952: #1326:
                energy_indices.append(i)
            else:
                if i % 28 >= 20: # if having identifier, need to minus 10
                    energy_indices.append(i)
                else:
                    seq_indices.append(i)
    if encoding == 'energy':
        X_train = X_train.iloc[:, energy_indices].copy()
        X_test = X_test.iloc[:, energy_indices].copy()
        if split == 3:
            X_val = X_val.iloc[:, energy_indices].copy()
    elif encoding == 'seq':
        X_train = X_train.iloc[:, seq_indices].copy()
        X_test = X_test.iloc[:, seq_indices].copy()
        if split == 3:
            X_val = X_val.iloc[:, seq_indices].copy()
    
    X_train = scale(X_train)
    X_test = scale(X_test)
    if split == 3:
        X_val = scale(X_val)

    if model == 'logistic_regression':
        from sklearn import linear_model
        lg = linear_model.LogisticRegression(C = 1, max_iter = 500)
        prob, acc = train_test(lg, X_train, y_train, X_test, y_test)
        print('Test Accuracy:{:.4f}'.format(acc))
        np.savetxt(os.path.join(save, 'logits_' + model + '_' + str(dataset) + '_' + encoding), prob)
    elif model == 'random_forest':
        av_acc = 0
        for i in range(20):
            from sklearn.ensemble import RandomForestClassifier
            rf = RandomForestClassifier()
            prob, acc = train_test(rf, X_train, y_train, X_test, y_test)
            av_acc += acc
        av_acc = av_acc / 20
        print('Test Accuracy:{:.4f}'.format(av_acc))
        np.savetxt(os.path.join(save, 'logits_' + model + '_' + str(dataset) + '_' + encoding), prob)
    elif model == 'decision_tree':
        from sklearn.tree import DecisionTreeClassifier
        dt = DecisionTreeClassifier()
        prob, acc = train_test(dt, X_train, y_train, X_test, y_test)
        print('Test Accuracy:{:.4f}'.format(acc))
        np.savetxt(os.path.join(save, 'logits_' + model + '_' + str(dataset) + '_' + encoding), prob)
    elif model == 'svm':
        from sklearn import svm
        svmsvc = svm.SVC(C = 1, probability=True)
        prob, acc = train_test(svmsvc, X_train, y_train, X_test, y_test)
        print('Test Accuracy:{:.4f}'.format(acc))
        np.savetxt(os.path.join(save, 'logits_' + model + '_' + str(dataset) + '_' + encoding), prob)
    elif model == 'ann':
        import tensorflow as tf
        from tensorflow import keras
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        accs = []
        dropout_list = [0.5,0.4,0.3,0.2,0.1,0.05,0.01]
        learning_rate = [0.01,0.05,1e-3,5e-3,1e-4,5e-4]
        combinations = []
        test_accs = []
        for dropout in dropout_list:
            for lr in learning_rate:
                print('dropout: {}; learning rate: {}'.format(dropout, lr))
                combinations.append([dropout, lr])
                n_class = 2
                ann = keras.Sequential([keras.layers.Dense(1024, activation=tf.nn.relu),
                                          keras.layers.Dropout(dropout, input_shape = (1024,)),
                                          keras.layers.Dense(n_class, activation=tf.nn.softmax)])

                ann.compile(optimizer=tf.train.AdamOptimizer(learning_rate = lr),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
                if split == 2:
                    prob, acc = train_test_ann(ann, n_class, X_train, y_train, X_test, y_test)
                elif split == 3:
                    prob, acc, test_prob, test_acc = train_test_ann_split(ann, n_class, X_train, y_train,
                                                                   X_test, y_test,
                                                                    X_val, y_val)
                np.savetxt(os.path.join(save, 'logits_val_' + model + '_' + str(dataset) + '_' + encoding +
                                       '_dropout_' + str(dropout) + '_lr_' + str(lr) + '_epoch_100'), prob)
                np.savetxt(os.path.join(save, 'logits_test_' + model + '_' + str(dataset) + '_' + encoding +
                                       '_dropout_' + str(dropout) + '_lr_' + str(lr) + '_epoch_100'), test_prob)
                accs.append(acc)
                test_accs.append(test_acc)
        print('Validation Accuracy:{:.4f}'.format(max(accs)))
        i=np.argmax(np.array(accs))
        print('Test Accuracy:{:.4f}'.format(test_accs[i]))
        print('Dropout: {:f}; Learning Rate: {:f}'.format(combinations[i][0], combinations[i][1]))

enco = args.feature
data = args.data
model = args.model

print(enco)
print(data)

# trisplit
if model == 'classic':
    for m in ['logistic_regression','random_forest','decision_tree']:
        print(model)
        trainSeqOnly(data, model = m, encoding = enco, split=3,
                         save = args.save)
else:
    print(model)
    trainSeqOnly(data, model = model, encoding = enco, split=3,
                         save = args.save)

