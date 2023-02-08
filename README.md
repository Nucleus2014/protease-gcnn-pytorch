# protease-gcnn-pytorch  
This project is to present a graph-based convolutional neural network, called protein convolutional neural network (PGCN) to predict protease specificity. We propose a new creation of feature set that holds natural energy information for proteins, which could best represent protein activities.  

![](https://github.com/Nucleus2014/protease-gcnn-pytorch/blob/master/pipeline.png)

To use our method, first download this repository by using the following command:  
```git clone https://github.com/Nucleus2014/protease-gcnn-pytorch```  
## Step 1: Generation of graphs  
Go to *graph* folder and excecute *protein_graph.py*:  
```
cd graph  
srun python protein_graph.py -o TEV_all_binary_10_ang_aa_energy_7_coord_energyedge_5_hbond -pr_path /projects/f_sdk94_1/EnzymeModelling/TEVFinalStructures -class TEV_final_all_var_noDup.txt -prot TEV_QS.pdb -d 10  
```
## Step 2: Train, validate and test
Go to *model* folder and excecute *train.py*:
```
cd model  
python train.py --dataset TEV_all_binary_10_ang_aa_energy_7_energyedge_5_hbond 
--test_dataset TEV_all_binary_10_ang_aa_energy_7_energyedge_5_hbond 
--val_dataset TEV_all_binary_10_ang_aa_energy_7_energyedge_5_hbond 
--seed 1 --epochs 500 --hidden1 20 --depth 2 --linear 0 --att 0 
--model gcn --batch_size 100 --lr 0.005 --dropout 0.2 --weight_decay 0.0005 
--save "outputs/tev/TEV_all_binary_10_ang_aa_energy_7_energyedge_5_hbond/bs_100/'  
```
Options of hyperparameter tuning:
```
weight_decay=(1e-3 5e-3 1e-4 5e-4)
learning_rate=(1e-2 5e-2 1e-3 5e-3 1e-4 5e-4)
dropout=(0.01 0.05 0.1 0.2 0.3 0.4 0.5)
batch_size=(500 100 1000 50 10)
```
## Test with trained model (Alternative)
If you would like to test with already-trained gcnn model, you could use *test.py* in *model* folder. It will load existed pytorch model file and test data that you specify.  
```
cd model
python test.py --dataset HCV_binary_10_ang_aa_sinusoidal_encoding_4_energy_7_energyedge_5_hbond --test_dataset HCV_binary_10_ang_aa_sinusoidal_encoding_2_energy_7_energyedge_5_hbond --epochs 500 --hidden1 20 --depth 2 --linear 1024 --att 0 --model gcn --batch_size 500 --lr $tmp_lr --dropout $tmp_dt --weight_decay $tmp_wd --save 'outputs/tt/HCV_binary_10_ang_aa_sinusoidal_encoding_4_energy_7_energyedge_5_hbond/bs_500/'  
```
## Hyperparameter Tuning on cluster (Alternative)
There are several examples in bash sciprt format for hyperparameter tuning on clusters. To use these scripts, clusters should support Slurm.  
Go to *bin* folder, *tmp_tuning.sh* is the slurm bash file that could submit to the cluster by using the following command:  
```
sbatch tmp_tuning.sh
```
When model has been trained, *train.py* saves the model to the corresponding preset directory (using the flag *--save*). 
I wrote a script to find the model with best accuracy, named as *analysis_parameter_tuning.py*. The way to use it is shown below:  
```
python analysis_parameter_tuning.py -i weight_node_feature_matrix  
```
where the flag '-i' records the middle name you gave to STDERR output file. For example, in *tmp_tuning.sh* the STDERR output fille is named as "ms.weight_node_feature_matrix.%a.%N.%j.err", then you should give "-i" flag "weight_node_feature_matrix".  

## Variable Importance Analysis (Alternative)
Here we propose a method to represent importance of nodes and edges. We refer variable importance method in random forest and you could use it by using following command:  
```
cd analysis
python importance.py --importance --dataset HCV_ternary_10_ang_aa_energy_7_energyedge_5_hbond --test_dataset HCV_ternary_10_ang_aa_energy_7_energyedge_5_hbond --hidden1 20 --depth 2 --linear 0 --att 0 --batch_size 500 --lr 0.005 --dropout 0.05 --weight_decay 5e-4 --save 'outputs/tt/HCV_ternary_10_ang_aa_energy_7_energyedge_5_hbond/bs_500/'  
```
## Comparison with other machine learning methods
In the paper, we compare GCNN + new generated feature set with five machine learning models + traditional feature set. For those results (parameter tuning + train and test) using machine learning models, see [ml-cleavage repository](https://github.com/Nucleus2014/ml-cleavage) in details. 
