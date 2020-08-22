#!/bin/bash

#SBATCH --partition=main             # Partition (job queue)
#SBATCH --requeue                    # Return job to the queue if preempted
#SBATCH --job-name=graph_bin         # Assign an short name to your job
#SBATCH --nodes=1                    # Number of nodes you require
#SBATCH --ntasks=1                  # Total # of tasks across all nodes
#SBATCH --cpus-per-task=1            # Cores per task (>1 if multithread tasks)
#SBATCH --mem=32000                 # Real memory (RAM) required (MB)
#SBATCH --time=24:00:00              # Total run time limit (HH:MM:SS)
#SBATCH --output=gg.10_ang_aa.%N.%j.out     # STDOUT output file
#SBATCH --error=gg.10_ang_aa.%N.%j.err      # STDERR output file (optional)
#SBATCH --export=ALL                 # Export you current env to the job env

cd /scratch/cl1205/protease-gcnn-pytorch/graph/

srun python protein_graph.py -o HCV_binary_10_ang -pr_path /scratch/cl1205/test_gcnn/silent_files -class HCV.txt -prot HCV.pdb -d 10

