# This lets you read a list of commands from a text file given in a flag and does all the slurming for you.
# By default they are run at /scratch/ss3410/GCNN. Additionally, you can specify where to put the .sh output file.
# By default they go down on file directory ex) /scratch/ss3410/GCNN/

"""
python text_to_slurm.py -txt /projects/f_sdk94_1/EnzymeModelling/Protease-Substrate-Design/HCV_D183A_commands.txt -job_name HCV_D183A -mem 12000 -path_operation /projects/f_sdk94_1/EnzymeModelling/Protease-Substrate-Design -path_sh /projects/f_sdk94_1/EnzymeModelling/Commands -batch 20 -time 2-00:00:00
"""

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-txt", type=str)
parser.add_argument("-job_name", type=str)
parser.add_argument("-path_operation", type=str)
parser.add_argument("-path_sh", type=str)
parser.add_argument("-mem", type=str)
parser.add_argument("-delay", type=int)
parser.add_argument("-batch", type=int)
parser.add_argument("-np",type=int, help="ratio in each batch that should be parallel")
parser.add_argument("-time", type=str)

args = parser.parse_args()

filename = args.txt
job_name = args.job_name
path = args.path_operation
sh = args.path_sh
delay = args.delay
mem = args.mem
batch = args.batch
np = args.np
time = args.time

if np == None:
   np = 1

if batch == None:
    batch == 1

if delay == None:
    delay = ""

if mem == None:
    mem = 2000
    
if path == None:
    path = "/projects/f_SDK94_1/EnymeModelling/Commands"

if job_name == None:
    raise ValueError("no name given")

if time == None:
    time = "3-00:00:00"

if not os.path.exists(filename) and not os.path.exists(os.path.join(os.getcwd(), filename)):
    raise ValueError("file specified not found")

with open(filename) as f:
    lineList = f.readlines()
    
header ="""#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name {0}.{1}
#SBATCH --partition main
#SBATCH --ntasks {2}
#SBATCH --cpus-per-task 1
#SBATCH --mem {3}
#SBATCH --output {0}.{1}.log
#SBATCH --error {0}.err
#SBATCH --time {5}
#SBATCH --begin now

cd {4}

"""

lineList = [x.strip() for x in lineList]

if sh == None:
    sh = "../Commands/"
else:
    sh += "/"

i = 0 
counter = 1

while i < len(lineList) + batch:
    command = r"{}{}_{}.sh".format(sh, job_name, counter)
    header_specific = header.format(job_name, counter, np, mem, path, time)
    if os.path.isfile(command):
        os.remove(command)
    f = open(command, "w")
    f.write(header_specific)
    for j in range(batch):
        if i + j < len(lineList):
            if (i + j) % np == 0:
                line = lineList[i+j]
                file_as_string = "\nsrun {}\n".format(line)
                f.write(file_as_string)
            else:
                line = lineList[i+j]
                file_as_string = "\nsrun {} &\n".format(line)
                f.write(file_as_string)
    f.write("printf done\n")
    f.close()
    i += batch
    counter += 1



