#!/bin/bash

#SBATCH --partition=main             # Partition (job queue)
#SBATCH --requeue                    # Return job to the queue if preempted
#SBATCH --job-name=tt1         # Assign an short name to your job
#SBATCH --array=0-167
#SBATCH --ntasks=1                  # Total # of tasks across all nodes
#SBATCH --cpus-per-task=1            # Cores per task (>1 if multithread tasks)
#SBATCH --mem=32000                 # Real memory (RAM) required (MB)
#SBATCH --time=24:00:00              # Total run time limit (HH:MM:SS)
#SBATCH --output=tt.HCV_binary_10_ang_aa_energy_7_energyedge_5_hbond.%a.%N.%j.out     # STDOUT output file
#SBATCH --error=tt.HCV_binary_10_ang_aa_energy_7_energyedge_5_hbond.%a.%N.%j.err      # STDERR output file (optional)
#SBATCH --export=ALL                 # Export you current env to the job env

data=$1
seed=$2
feature=$3

echo "data: $data"
echo "seed: $seed"
echo "feature: $feature"
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
echo "array id: $SLRUM_ARRAY_TASK_ID"
echo "weight decay: ${wd[$SLURM_ARRAY_TASK_ID]}"
echo "learning rate: ${lr[$SLURM_ARRAY_TASK_ID]}"
echo "dropout rate: ${dt[$SLURM_ARRAY_TASK_ID]}"
tmp_wd=${wd[$SLURM_ARRAY_TASK_ID]}
tmp_lr=${lr[$SLURM_ARRAY_TASK_ID]}
tmp_dt=${dt[$SLURM_ARRAY_TASK_ID]}

if [ ${feature} == _ ]
then
    flag=--energy_only
else
    flag=
fi

echo "batch_size: 500"
python test.py --dataset HCV_${data}_binary_10_ang_aa_energy_7_energyedge_5_hbond --test_dataset HCV_aa_binary_10_ang_aa_energy_7_energyedge_5_hbond --seed ${seed} --epochs 500 --hidden1 20 --depth 2 --linear 0 --att 0 --model gcn --batch_size 500 --lr $tmp_lr --dropout $tmp_dt --weight_decay $tmp_wd --save "outputs/tt_finalize_20220211/HCV_${data}_binary_10_ang${feature}energy_7_energyedge_5_hbond/bs_500/" ${flag} #&> tt.log 

echo "batch_size: 100"
python test.py --dataset HCV_${data}_binary_10_ang_aa_energy_7_energyedge_5_hbond --test_dataset HCV_${data}_binary_10_ang_aa_energy_7_energyedge_5_hbond --seed ${seed} --epochs 500 --hidden1 20 --depth 2 --linear 0 --att 0 --model gcn --batch_size 100 --lr $tmp_lr --dropout $tmp_dt --weight_decay $tmp_wd --save "outputs/tt_finalize_20220211/HCV_${data}_binary_10_ang${feature}energy_7_energyedge_5_hbond/bs_100/" ${flag}

echo "batch_size: 1000"
python test.py --dataset HCV_${data}_binary_10_ang_aa_energy_7_energyedge_5_hbond --test_dataset HCV_${data}_binary_10_ang_aa_energy_7_energyedge_5_hbond --seed ${seed} --epochs 500 --hidden1 20 --depth 2 --linear 0 --att 0 --model gcn --batch_size 1000 --lr $tmp_lr --dropout $tmp_dt --weight_decay $tmp_wd --save "outputs/tt_finalize_20220211/HCV_${data}_binary_10_ang${feature}energy_7_energyedge_5_hbond/bs_1000/" ${flag}

echo "batch_size: 50"
python test.py --dataset HCV_${data}_binary_10_ang_aa_energy_7_energyedge_5_hbond --test_dataset HCV_${data}_binary_10_ang_aa_energy_7_energyedge_5_hbond --seed ${seed} --epochs 500 --hidden1 20 --depth 2 --linear 0 --att 0 --model gcn --batch_size 50 --lr $tmp_lr --dropout $tmp_dt --weight_decay $tmp_wd --save "outputs/tt_finalize_20220211/HCV_${data}_binary_10_ang${feature}energy_7_energyedge_5_hbond/bs_50/" ${flag}

echo "batch_size: 10"
python test.py --dataset HCV_${data}_binary_10_ang_aa_energy_7_energyedge_5_hbond --test_dataset HCV_${data}_binary_10_ang_aa_energy_7_energyedge_5_hbond --seed ${seed} --epochs 500 --hidden1 20 --depth 2 --linear 0 --att 0 --model gcn --batch_size 10 --lr $tmp_lr --dropout $tmp_dt --weight_decay $tmp_wd --save "outputs/tt_finalize_20220211/HCV_${data}_binary_10_ang${feature}energy_7_energyedge_5_hbond/bs_10/" ${flag}


