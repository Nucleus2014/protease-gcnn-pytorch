#!/bin/bash

#SBATCH --partition=main             # Partition (job queue)
#SBATCH --requeue                    # Return job to the queue if preempted
#SBATCH --job-name=ms1         # Assign an short name to your job
#SBATCH --array=0-399
#SBATCH --ntasks=1                  # Total # of tasks across all nodes
#SBATCH --cpus-per-task=1            # Cores per task (>1 if multithread tasks)
#SBATCH --mem=32000                 # Real memory (RAM) required (MB)
#SBATCH --time=24:00:00              # Total run time limit (HH:MM:SS)
#SBATCH --output=ms.weight_node_feature_matrix.%a.%N.%j.out     # STDOUT output file
#SBATCH --error=ms.weight_node_feature_matrix.%a.%N.%j.err      # STDERR output file (optional)
#SBATCH --export=ALL                 # Export you current env to the job env

cd /scratch/cl1205/protease-gcnn-pytorch/model/
cd /scratch/cl1205/protease-gcnn-pytorch/model/
weight_decay=(1e-3 5e-3 1e-4 5e-4)
learning_rate=(1e-2 1e-3 5e-3 1e-4 5e-4)
batch_size=(50 100 500 1000)
hidden=(10 15 20 25 30)
wd=()
lr=()
bs=()
hid=()
for i in {0..3}
do
    for j in {0..4}
    do
        for k in {0..3}
        do
            for l in {0..4}
            do
                wd+=(${weight_decay[$i]})
                lr+=(${learning_rate[$j]})
                bs+=(${batch_size[$k]})
                hid+=(${hidden[$l]})
            done
        done
    done
done
echo ${wd[@]}
echo ${lr[@]}
echo ${bs[@]}
echo ${hid[@]}
echo $SLRUM_ARRAY_TASK_ID
echo ${wd[$SLURM_ARRAY_TASK_ID]}
echo ${lr[$SLURM_ARRAY_TASK_ID]}
echo ${bs[$SLURM_ARRAY_TASK_ID]}
echo ${hid[$SLURM_ARRAY_TASK_ID]}
tmp_wd=${wd[$SLURM_ARRAY_TASK_ID]}
tmp_lr=${lr[$SLURM_ARRAY_TASK_ID]}
tmp_bs=${bs[$SLURM_ARRAY_TASK_ID]}
tmp_hid=${hid[$SLURM_ARRAY_TASK_ID]}
srun python train.py --save_validation --dataset HCV_binary_10_ang_aa_sinusoidal_encoding_6_energy_7_energyedge_5_hbond --test_dataset HCV_binary_10_ang_aa_sinusoidal_encoding_2_energy_7_energyedge_5_hbond --epochs 1000 --hidden1 $tmp_hid --weight post --depth 2 --att 0 --model gcn --batch_size $tmp_bs --lr $tmp_lr --dropout 0.01 --weight_decay $tmp_wd --save 'outputs/ms/weight_node_feature_matrix/' #&> tt.log 




