#!/bin/bash

#SBATCH --partition=p_sdk94_1             # Partition (job queue)
#SBATCH --requeue                    # Return job to the queue if preempted
#SBATCH --job-name=ms1         # Assign an short name to your job
#SBATCH --array=0-239
#SBATCH --ntasks=1                  # Total # of tasks across all nodes
#SBATCH --cpus-per-task=1            # Cores per task (>1 if multithread tasks)
#SBATCH --mem=3200                 # Real memory (RAM) required (MB)
#SBATCH --time=24:00:00              # Total run time limit (HH:MM:SS)
#SBATCH --output=ms.preweight.%a.%N.%j.out     # STDOUT output file
#SBATCH --error=ms.preweight.%a.%N.%j.err      # STDERR output file (optional)
#SBATCH --export=ALL                 # Export you current env to the job env

cd /scratch/cl1205/protease-gcnn-pytorch/model/
depth=(2 5 10)
learning_rate=(1e-2 5e-2 1e-3 5e-3 1e-4 5e-4)
dropout=(0.01 0.05 0.1 0.2 0.3 0.4 0.5)
weight_decay=(1e-3 5e-3 1e-4 5e-4)
hidden=(16 32 64 128 8)
linear=(0 1024)
hid=()
lin=()
wd=()
lr=()
dt=()
dep=()
for i in {0..2}
do
    for j in {0..5}
    do
        for k in {0..6}
        do
            for l in {0..3}
            do
                for m in {0..4}
                do
                    for n in {0..1}
                    do
                        hid+=(${hidden[$m]})
                        lin+=(${linear[$n]})
                        wd+=(${weight_decay[$l]})
                        lr+=(${learning_rate[$j]})
                        dt+=(${dropout[$k]})
                        dep+=(${depth[$i]})
                    done
                done
            done
        done
    done
done
echo $SLRUM_ARRAY_TASK_ID
echo ${hid[$SLURM_ARRAY_TASK_ID]}
tmp_hid=${hid[$SLURM_ARRAY_TASK_ID]}
tmp_lin=${lin[$SLURM_ARRAY_TASK_ID]}
tmp_wd=${wd[$SLRUM_ARRAY_TASK_ID]}
tmp_lr=${lr[$SLRUM_ARRAY_TASK_ID]}
tmp_dt=${dt[$SLRUM_ARRAY_TASK_ID]}
tmp_dep=${dep[$SLRUM_ARRAY_TASK_ID]}
srun python train.py --save_validation --dataset HCV_binary_10_ang_aa_sinusoidal_encoding_4_energy_7_energyedge_5_hbond --test_dataset HCV_binary_10_ang_aa_sinusoidal_encoding_2_energy_7_energyedge_5_hbond --epochs 500 --hidden1 $tmp_hid --linear $tmp_lin --depth $tmp_dep --att 0 --model gcn --batch_size 500 --lr $tmp_lr --dropout $tmp_dt --weight_decay $tmp_wd --save 'outputs/ms/preweight/' #&> tt.log 




