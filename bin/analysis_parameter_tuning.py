import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-i','--input',type=str,help='Name of input data')
parser.add_argument('-a','--array',type=int,help='Number of array jobs')
args = parser.parse_args()

data=args.input
#array=args.array
acc = 0
for file in os.listdir('./'):
    if file[-3:] == 'err':
        if file.split('.')[1] == data:
            fp=open(file,'r')
            for line in fp:
                if line[0:4] == '0007':
                    if float(line.split(' ')[-1]) > acc:
                        acc = float(line.split(' ')[-1])
                        infile = file
print(infile)
print(acc)
