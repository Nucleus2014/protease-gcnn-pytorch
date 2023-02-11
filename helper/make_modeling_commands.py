import os
import pandas as pd
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--info_file', type=str,
                        default='/projects/f_sdk94_1/protease_3C/data_label_assignment/',
                        help='Directory of the information for all structures to be generated. \
                        Each of them should consist of three columns, \
                        protease_name or protease_mutations, substrate_sequence, and label.')
    parser.add_argument('-p1', '--p1', type=int,
                        default=-2,
                        help='index of p1, can be either negative or positive indices. \
                             e.g., p1=0 means p1 is the first of the substrate; \
                             p1=-2 means p1 is the last second of the substrate sequence. \
                             If you use . as a delimiter between P1 and P1 prime position, ignore this flag.')
    parser.add_argument('-struct', '--starting_structures', type=str,
                        default='/projects/f_sdk94_1/protease_3C/final_3C_protease_peptide_structures',
                        help='Directory of starting structure(s). It can handle multiple starting structures. \
                             If multiple starting strctures, make sure names of starting structures match \
                             protease_name in the info_file.')
    return parser.parse_args()

def createCrys(p, prot, root):
    letter1 = 'ARNDBCEQZGHILKMFPSTWYV'
    letter1 = list(letter1)
    letter3 = ['ALA', 'ARG', 'ASN', 'ASP', 'ASX', 'CYS', 'GLU', 'GLN', 'GLX', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS',
               'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
    letterMap = {letter1[i]: letter3[i] for i in range(len(letter1))}

    with open(os.path.join(root, prot + '_' + p + '.pdb'), 'w') as gp:
        fp = open(os.path.join(root, prot + '.pdb'), 'r')
        p1Count = 0
        p2Count = 0
        pp = list(p)
        for line in fp:
            if p1Count != 1 or p2Count != 1:
                p1Ind = line.find('GLN')
                p2Ind = line.find('SER')
                if p1Ind != -1:
                    p1Count = 1
                    line = line[0:p1Ind] + letterMap[p[0]] + line[p1Ind + 3:]
                elif p2Ind != -1:
                    p2Count = 1
                    line = line[0:p2Ind] + letterMap[p[1]] + line[p2Ind + 3:]
            gp.write(line)

def main(args):
    info_files_path = args.info_file
    p1_ind = args.p1
    starting_structure_path = args.starting_structures
    for mutSeqLabel in os.listdir(info_files_path):

    df = pd.read_csv(mutSeqLabel, index_col=0)
    proteases = []
    p1p11s = []
    for i in range(df.shape[0]):
        protease = df.iloc[i, 0]
        seq = df.iloc[i, 1]
        dotInd = seq.find('.')
        if dotInd == -1:
            dotInd = p1_ind
            p1p11 = ''.join(seq[dotInd - 1] + seq[dotInd + 1])
        # check whether file exists or not
        if os.path.isfile(os.path.join(starting_structure_path, protease + '_' + p1p11 + '.pdb')):
            pass
        else:
            createCrys(p1p11, protease, starting_structure_path)

if __name__ == '__main__':
    args = parse_args()
    main(args)
