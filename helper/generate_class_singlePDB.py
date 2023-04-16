'''
This script is to generate classification file if generated structures are in single PDB format.
python generate_class_singlePDB.py -s /projects/f_sdk94_1/PGCN/protease-gcnn-pytorch/graph/crystal_structures/2yol_class_generated_structures -class /projects/f_sdk94_1/PGCN/protease-gcnn-pytorch/graph/classifications/3cProt_class/2yol-ER-summarized_label.txt 
Changpeng Lu 2023-04-16
Vidur Sarma 2023-04-15
'''
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import json
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--score_folder_path', type=str,
                        default='/projects/f_sdk94_1/PGCN/protease-gcnn-pytorch/graph/crystal_structures/2yol_class_generated_structures',
                        help='Directory of generated structures.')
    parser.add_argument('-class', '--classification_file', type=str,
                        default='/projects/f_sdk94_1/PGCN/protease-gcnn-pytorch/graph/classifications/3cProt_class/2yol-ER-summarized_label.txt',
                        help='Directory of generated structures.')
    return parser.parse_args()

def main(args):
    score_path = Path(args.score_folder_path)
    class_file = Path(args.classification_file)
    df_class = pd.read_csv(class_file, delimiter='\t')
    # edit based on Vidur's code
    new_sequences = []
    for seq in df_class['Sequence']: #df_class['Sequence']
        fasc = score_path / (seq + '.fasc')
        with open(fasc, 'r') as fp:
            for i, line in enumerate(fp):
                js = json.loads(line)
                if i == 0:
                    dic_scores = defaultdict(list, { k:[v] for k,v in js.items()})
                else:
                    for k in js.keys():
                        dic_scores[k].append(js[k])
        df = pd.DataFrame(dic_scores)
        pdb = df.loc[df['total_score'].idxmin(),['filename']].values[0].split('/')[-1]
        new_sequences.append(pdb)
    df = pd.DataFrame({'Sequence': new_sequences, 'Result': df_class['Result']}) #df_class['Result']
    df.to_csv(class_file.parent / (class_file.stem.split('.')[0] + '_singlePDB.txt'), sep='\t', index=None)

if __name__ == '__main__':
    args = parse_args()
    main(args)
