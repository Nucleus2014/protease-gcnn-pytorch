#!/bin/bash

#SBATCH --partition=main             # Partition (job queue)
#SBATCH --requeue                    # Return job to the queue if preempted
#SBATCH --job-name=ml         # Assign an short name to your job
#SBATCH --ntasks=1                  # Total # of tasks across all nodes
#SBATCH --cpus-per-task=1            # Cores per task (>1 if multithread tasks)
#SBATCH --mem=32000                 # Real memory (RAM) required (MB)
#SBATCH --time=3-00:00:00              # Total run time limit (HH:MM:SS)
#SBATCH --output=tt.HCV_flatten.%a.%N.%j.out     # STDOUT output file
#SBATCH --error=tt.HCV_flatten.%a.%N.%j.err      # STDERR output file (optional)
#SBATCH --export=ALL                 # Export you current env to the job env
data=$1
feature=$2
model=$3

cd /scratch/cl1205/ml-cleavage/scripts
python BenchmarkMLTrainAfterPGCN.py -data $data -feature $feature -model $model -save "/scratch/cl1205/ml-cleavage/outputs/hcv_noProtID_trisplit_20220705"

