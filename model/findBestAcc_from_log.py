import os
#import pandas as pd
import numpy as np

max_acc = 0
for f in os.listdir('./'):
    if f.split('.')[-1] == 'out' and f.split('.')[0] == 'tt':
        with open(f, 'r') as fp:
            lines = fp.readlines()
            best_epoch = int(lines[-1].split(':')[-1].strip())
            for i, line in enumerate(lines):
                if line.split(':')[-1].strip() == str(best_epoch):
                    ind = i
                    break
        val_acc = float(lines[ind+4].split(':')[-1].strip())
        if val_acc > max_acc:
            max_acc = val_acc
            good_f = f
print(good_f)
print('accuracy: {}'.format(max_acc))
                
