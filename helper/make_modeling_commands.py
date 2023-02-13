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
    parser.add_argument('-script_path', '--script_path', type=str,
                        default = '/projects/f_sdk94_1/EnzymeModelling/Protease-Substrate-Design',
                        help='Where to save output file for all commands')
    parser.add_argument('-o', '--output_name', type=str,
                        default='new.command.txt',
                        help='output command file name')
    parser.add_argument('-f', '--output_format', choices=['sequence','silent'],
                        default='sequence',
                        help='two options of output format, either sequence, or silent files. \
                             Silent file mode will concatenate sequences which have same patterns into one same file.')
    parser.add_argument('-os', '--output_structure_directory', type=str,
                        default='/projects/f_sdk94_1/EnzymeModelling/Protease3C/2bof',
                        help='where to put generated Rosetta structures')
    parser.add_argument('-constraint', '--constraint_suffix', type=str,
                        default="-site 215 -cons tev.cst -cr 39 74 144 -dprot 0 -dpep 0",
                        help='Specify all flags for design_protease.py, e.g., -site 215 -cons tev.cst -cr 39 74 144 -dprot 0 -dpep 0 \
                                -site specifies the starting pose index of threading, -cr specifies three catalytic residues.')
    parser.add_argument('-jn', '--job_name', type=str,
                        default=None,
                        help='job name for Rosetta modeling')
    parser.add_argument('-bs', '--batch_size', type=int,
                        default=5,
                        help='')
    parser.add_argument('-cd', '--command_directory', type=str,
                        default='/projects/f_sdk94_1/EnzymeModelling/Commands_OYDV')
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
        p1_motif = letterMap[p_wt[0]] + ' ' + str(ind)
        p2_motif = letterMap[p_wt[1]] + ' ' + str(ind+1)
        for line in fp:
            if line.find('REMARK') != 1 and line.find(p1_motif) != -1:
                p1Ind = line.find(p1_motif)
                line = line[0:p1Ind] + letterMap[pp[0]] + line[p1Ind + 3:]
            if line.find('REMARK') !=1 and line.find(p2_motif) != -1:
                p2Ind = line.find(p2_motif)
                line = line[0:p2Ind] + letterMap[pp[1]] + line[p2Ind + 3:]
            gp.write(line)

def toCommands(args, info_set, constraint, mode = 'silent'):
    output_name = args.output_name
    script_path = args.script_path
    p1_ind = args.p1_index_substrate
    root = Path(args.starting_structures)
    outStructFolder = args.output_structure_directory

    # if mode == 'silent':
    #     with open(os.path.join(out_path, output_name), 'w') as fp:
    #         for silent in tmpSilent:
    #             tmp = list(silent)
    #             dotInd = silent.find('.')
    #             p1p11 = ''.join(silent[dotInd-1] + silent[dotInd+1])
    #             fp.write('python design_protease.py -s ' + os.path.join(crysPath, crysPath.split('/')[-1] + '_' + p1p11 + '.pdb') +
    #             ' -od ' + silentPath + ' -st ' + os.path.join(out, 'new.sequence.txt') +
    #             ' -sf ' + silent + " " + constraint + '\n')
    # elif mode == 'sequence':
    sequences = info_set[0]
    mutant_list = info_set[1]
    with open(os.path.join(script_path, output_name), 'w') as fp:
        for i in range(len(sequences)):
            mutant = mutant_list[i]
            seq = sequences[i]
            p1p11, newSeq = locate_p1p11(seq, p1_ind)
            newStructPath = root.parent / (root.stem + '_' + p1p11 + '.pdb')
            name = mutant + '_' + newSeq
            if mutant == '':
                name = newSeq
            fp.write('python design_protease.py -s ' + str(newStructPath) +
                    ' -od ' + outStructFolder + ' -seq ' + newSeq + ' -name ' + name +
                    " " + constraint + '\n')

def locate_p1p11(seq, p1_ind=None):
    dotInd = seq.find('.')
    p1p11 = ''.join(seq[dotInd - 1] + seq[dotInd + 1])
    oriSeq = ''.join(seq[0:dotInd] + seq[dotInd + 1:])
    if dotInd == -1:
        dotInd = p1_ind
        assert p1_ind != -1
        p1p11 = seq[dotInd] + seq[dotInd+1]
        oriSeq = seq
    return p1p11, oriSeq

def printToBatchCommand(args):
    jobName = Path(args.info_file).stem
    if args.job_name != None:
        jobName = args.job_name
    commandPath = args.command_directory
    nBatch = args.batch_size
    scriptPath = args.script_path
    output_name = args.output_name

    splitCommand = "python " + scriptPath + "/text_to_slurm.py -txt " + os.path.join(scriptPath, output_name) + " -job_name " + \
          jobName + " -mem 12000 -path_operation " + scriptPath + " -path_sh " + \
          commandPath + " -batch " + str(nBatch) + " -time 3-00:00:00"
    os.system(splitCommand)
    # print("python text_to_slurm.py -txt " + os.path.join(scriptPath, 'new.command.txt') + " -job_name " +
    #       jobName + " -mem 12000 -path_operation " + scriptPath + " -path_sh " +
    #       commandPath + " -batch " + str(nBatch) + " -time 3-00:00:00")

def mkdir(path):
    if not path.exists():
        path.mkdir(parents=True)

def main(args):
    mutSeqLabel = Path(args.info_file) #info_files_path
    p1_ind = args.p1_index_substrate
    p1_ind_pdb = args.p1_index_pdb
    starting_structure_path = Path(args.starting_structures)
    structure_save_path = starting_structure_path.parent
    p1p11_wt = args.p1p11_wt
    format = args.output_format
    constraintSuffix = args.constraint_suffix
    commandPath = Path(args.command_directory)
    mkdir(commandPath)

    # Use intermediate output from CleavEX as the input. Need to update in the future
    df = pd.read_csv(mutSeqLabel, index_col=0)
    mutant_list = [''] * df.shape[0]
    for column_name in df.columns:
        if column_name.lower().find('mutant') != -1:
            mutant_list = df[column_name]
    sequences = df.index.values
    p1p11s = []
    new_c = 0
    for seq in sequences:
        # protease = df.iloc[i, 0]
        p1p11,_ = locate_p1p11(seq, p1_ind)
        # check whether file exists or not
        if (structure_save_path / (starting_structure_path.stem + '_' + p1p11 + '.pdb')).is_file(): #, protease + '_' + p1p11 + '.pdb'
            # print('starting structure for {} exists! Skip it....'.format(p1p11))
            continue
        else:
            createCrys(p1p11_wt, p1p11, p1_ind_pdb, starting_structure_path)
            new_c += 1
    print('Swapping {} number of P1P11 combinations'.format(new_c))
    # if format == 'silent':
    toCommands(args, (sequences, mutant_list), constraintSuffix, mode=format)
    printToBatchCommand(args)

if __name__ == '__main__':
    args = parse_args()
    main(args)
