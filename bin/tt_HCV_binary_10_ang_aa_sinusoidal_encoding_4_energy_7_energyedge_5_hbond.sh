#!/bin/bash

#SBATCH --partition=p_sdk94_1             # Partition (job queue)
#SBATCH --requeue                    # Return job to the queue if preempted
#SBATCH --job-name=tt1         # Assign an short name to your job
#SBATCH --array=0-167
#SBATCH --ntasks=1                  # Total # of tasks across all nodes
#SBATCH --cpus-per-task=1            # Cores per task (>1 if multithread tasks)
#SBATCH --mem=32000                 # Real memory (RAM) required (MB)
#SBATCH --time=24:00:00              # Total run time limit (HH:MM:SS)
#SBATCH --output=tt.HCV_binary_10_ang_aa_sinusoidal_encoding_4_energy_7_energyedge_5_hbond.1024.%a.%N.%j.out     # STDOUT output file
#SBATCH --error=tt.HCV_binary_10_ang_aa_sinusoidal_encoding_4_energy_7_energyedge_5_hbond.1024.%a.%N.%j.err      # STDERR output file (optional)
#SBATCH --export=ALL                 # Export you current env to the job env

cd /scratch/cl1205/protease-gcnn-pytorch/model/
weight_decay=(1e-3 5e-3 1e-4 5e-4)
learning_rate=(1e-2 5e-2 1e-3 5e-3 1e-4 5e-4)
dropout=(0.01 0.05 0.1 0.2 0.3 0.4 0.5)
wd=()
lr=()
dt=()
for i in {0..3}
do
    for j in {0..5}
    do
        for k in {0..6}
        do
            wd+=(${weight_decay[$i]})
            lr+=(${learning_rate[$j]})
            dt+=(${dropout[$k]})
        done
    done
done
echo ${wd[@]}
echo ${lr[@]}
echo ${dt[@]}
echo $SLRUM_ARRAY_TASK_ID
echo ${wd[$SLURM_ARRAY_TASK_ID]}
echo ${lr[$SLURM_ARRAY_TASK_ID]}
echo ${dt[$SLURM_ARRAY_TASK_ID]}
tmp_wd=${wd[$SLURM_ARRAY_TASK_ID]}
tmp_lr=${lr[$SLURM_ARRAY_TASK_ID]}
tmp_dt=${dt[$SLURM_ARRAY_TASK_ID]}
srun python test.py --dataset HCV_binary_10_ang_aa_sinusoidal_encoding_4_energy_7_energyedge_5_hbond --test_dataset HCV_binary_10_ang_aa_sinusoidal_encoding_2_energy_7_energyedge_5_hbond --epochs 500 --hidden1 20 --depth 2 --linear 1024 --att 0 --model gcn --batch_size 500 --lr $tmp_lr --dropout $tmp_dt --weight_decay $tmp_wd --save 'outputs/tt/HCV_binary_10_ang_aa_sinusoidal_encoding_4_energy_7_energyedge_5_hbond/bs_500/' #&> tt.log 

srun python test.py --dataset HCV_binary_10_ang_aa_sinusoidal_encoding_4_energy_7_energyedge_5_hbond --test_dataset HCV_binary_10_ang_aa_sinusoidal_encoding_2_energy_7_energyedge_5_hbond --epochs 500 --hidden1 20 --depth 2 --linear 1024 --att 0 --model gcn --batch_size 100 --lr $tmp_lr --dropout $tmp_dt --weight_decay $tmp_wd --save 'outputs/tt/HCV_binary_10_ang_aa_sinusoidal_encoding_4_energy_7_energyedge_5_hbond/bs_100/'

srun python test.py --dataset HCV_binary_10_ang_aa_sinusoidal_encoding_4_energy_7_energyedge_5_hbond --test_dataset HCV_binary_10_ang_aa_sinusoidal_encoding_2_energy_7_energyedge_5_hbond --epochs 500 --hidden1 20 --depth 2 --linear 1024 --att 0 --model gcn --batch_size 1000 --lr $tmp_lr --dropout $tmp_dt --weight_decay $tmp_wd --save 'outputs/tt/HCV_binary_10_ang_aa_sinusoidal_encoding_4_energy_7_energyedge_5_hbond/bs_1000/'

srun python test.py --dataset HCV_binary_10_ang_aa_sinusoidal_encoding_4_energy_7_energyedge_5_hbond --test_dataset HCV_binary_10_ang_aa_sinusoidal_encoding_2_energy_7_energyedge_5_hbond --epochs 500 --hidden1 20 --depth 2 --linear 1024 --att 0 --model gcn --batch_size 50 --lr $tmp_lr --dropout $tmp_dt --weight_decay $tmp_wd --save 'outputs/tt/HCV_binary_10_ang_aa_sinusoidal_encoding_4_energy_7_energyedge_5_hbond/bs_50/'

srun python test.py --dataset HCV_binary_10_ang_aa_sinusoidal_encoding_4_energy_7_energyedge_5_hbond --test_dataset HCV_binary_10_ang_aa_sinusoidal_encoding_2_energy_7_energyedge_5_hbond --epochs 500 --hidden1 20 --depth 2 --linear 1024 --att 0 --model gcn --batch_size 10 --lr $tmp_lr --dropout $tmp_dt --weight_decay $tmp_wd --save 'outputs/tt/HCV_binary_10_ang_aa_sinusoidal_encoding_4_energy_7_energyedge_5_hbond/bs_10/'


