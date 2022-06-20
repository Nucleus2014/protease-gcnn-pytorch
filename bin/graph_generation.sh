#!/bin/bash

#SBATCH --partition=main             # Partition (job queue)
#SBATCH --requeue                    # Return job to the queue if preempted
#SBATCH --job-name=TEV_all         # Assign an short name to your job
#SBATCH --nodes=1                    # Number of nodes you require
#SBATCH --ntasks=1                  # Total # of tasks across all nodes
#SBATCH --cpus-per-task=1            # Cores per task (>1 if multithread tasks)
#SBATCH --mem=32000                 # Real memory (RAM) required (MB)
#SBATCH --time=3-00:00:00              # Total run time limit (HH:MM:SS)
#SBATCH --output=gg.tev_all.10_ang_aa_energy_7_energyedge_5_hbond.%N.%j.out     # STDOUT output file
#SBATCH --error=gg.tev_all.10_ang_aa_energy_7_energyedge_5_hbond.%N.%j.err      # STDERR output file (optional)
#SBATCH --export=ALL                 # Export you current env to the job env

cd /scratch/cl1205/protease-gcnn-pytorch/graph/

srun python protein_graph.py -o TEV_all_binary_10_ang_aa_energy_7_energyedge_5_hbond -pr_path /projects/f_sdk94_1/EnzymeModelling/TEVFinalStructures -class TEV_final_all_var_noDup.txt -prot TEV_QS.pdb -d 10

