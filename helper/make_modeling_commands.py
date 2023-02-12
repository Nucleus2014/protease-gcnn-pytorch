import numpy as np
import pandas as pd
import os
from pathlib import Path
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--info_file', type=str,
                        default='/projects/f_sdk94_1/protease_3C/data_ngs_enrichment/2bof-ER-summarized.csv',
                        help='Directory of the information for all structures to be generated. \
                        It should consist of three columns, (currently, the program cannot support multiple proteases)\
                        protease_name or protease_mutations, substrate_sequence, and label.')
    parser.add_argument('-p1p11', '--p1p11_wt', type=str,
                        default='QS',
                        help='index of p1, can be either negative or positive indices. \
                                 e.g., p1=0 means p1 is the first of the substrate; \
                                 p1=-2 means p1 is the last second of the substrate sequence. \
                                 If you use . as a delimiter between P1 and P1 prime position, ignore this flag.')
    parser.add_argument('-p1_ind', '--p1_index_substrate', type=int,
                        default=888,
                        help='index of p1, can be either negative or positive indices. \
                             e.g., p1=0 means p1 is the first of the substrate; \
                             p1=-2 means p1 is the last second of the substrate sequence. \
                             If you use . as a delimiter between P1 and P1 prime position, ignore this flag.')
    parser.add_argument('-p1_pdb', '--p1_index_pdb', type=int,
                        default=7,
                        help='pdb index of p1.')
    parser.add_argument('-struct', '--starting_structures', type=str,
                        default='/projects/f_sdk94_1/protease_3C/final_3C_protease_peptide_structures/2b0f_wt_pep.pdb',
                        help='Directory of starting structure(s). It currently cannot handle multiple starting structures. \
                             If multiple starting strctures, make sure names of starting structures match \
                             protease_name in the info_file.')
    return parser.parse_args()

def createCrys(p_wt, p, ind, root):
    letter1 = 'ARNDBCEQZGHILKMFPSTWYV'
    letter1 = list(letter1)
    letter3 = ['ALA', 'ARG', 'ASN', 'ASP', 'ASX', 'CYS', 'GLU', 'GLN', 'GLX', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS',
               'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
    letterMap = {letter1[i]: letter3[i] for i in range(len(letter1))}

    with open(root.parent / (root.stem + '_' + p + '.pdb'), 'w') as gp:
        fp = open(root, 'r')
        # p1Count = 0
        # p2Count = 0
        pp = list(p) #['Q','S']
        p1_motif = p_wt[0] + ' ' + str(ind)
        p2_motif = p_wt[1] + ' ' + str(ind+1)
        for line in fp:
            if line.find(p1_motif) != -1:
                p1Ind = line.find(p1_motif)
                line = line[0:p1Ind] + letterMap[pp[0]] + line[p1Ind + 3:]
            if line.find(p2_motif) != -1:
                p2Ind = line.find(p2_motif)
                line = line[0:p2Ind] + letterMap[pp[1]] + line[p2Ind + 3:]
            gp.write(line)

def main(args):
    mutSeqLabel = args.info_file #info_files_path
    # protease = args.info_file.split('-')[0]
    p1_ind = args.p1_index_substrate
    p1_ind_pdb = args.p1_index_pdb
    starting_structure_path = Path(args.starting_structures)
    structure_save_path = starting_structure_path.parent
    p1p11_wt = args.p1p11_wt
    # Use intermediate output from CleavEX as the input. Need to update in the future
    df = pd.read_csv(mutSeqLabel, index_col=0)
    sequences = df.index.values

    p1p11s = []
    new_c = 0
    for seq in sequences:
        # protease = df.iloc[i, 0]
        dotInd = seq.find('.')
        if dotInd == -1:
            dotInd = p1_ind
            p1p11 = ''.join(seq[dotInd - 1] + seq[dotInd + 1])
        # check whether file exists or not
        if (structure_save_path / (starting_structure_path.stem + '_' + p1p11 + '.pdb')).is_file(): #, protease + '_' + p1p11 + '.pdb'
            # print('starting structure for {} exists! Skip it....'.format(p1p11))
            continue
        else:
            createCrys(p1p11_wt, p1p11, p1_ind_pdb, starting_structure_path)
            new_c += 1
    print('Swapping {} number of P1P11 combinations'.format(new_c))

if __name__ == '__main__':
    args = parse_args()
    main(args)
