# protease-gcnn-pytorch  
This project is to present a graph-based convolutional neural network, called protein convolutional neural network (PGCN) to predict protease specificity. We propose a new creation of feature set that holds natural energy information for proteins, which could best represent protein activities. 

![](https://github.com/Nucleus2014/protease-gcnn-pytorch/blob/master/pipeline.png)

To use our method, first download this repository by using the following command:  
```git clone https://github.com/Nucleus2014/protease-gcnn-pytorch```  

Rosetta models were generated using *design_protease.py* script. If you need source structures for pre-trained HCV/TEV models, please contact us.  
Pre-trained models for HCV/TEV are in [model/outputs](https://github.com/Nucleus2014/protease-gcnn-pytorch/tree/master/model/outputs), and cleavage information for HCV/TEV/TEV_design are in [graph/classifications folder](https://github.com/Nucleus2014/protease-gcnn-pytorch/tree/master/graph/classifications).   

## Step 1: Generation of graphs  
Go to *graph* folder and excecute *protein_graph.py*:  
```
cd graph  
srun python protein_graph.py -o <name of data> -pr_path /projects/f_sdk94_1/EnzymeModelling/TEVFinalStructures -class TEV.txt -prot TEV_QS.pdb -d 10  
```
### Description of generated data  
If the suffix is one of the below,  
`<graph>`: the edge feature tensor in the dimension of (K,N,N,M)  
`<x>`: the node feature matrix in the dimension of (K,N,F)  
`<y>`: labels in the dimension of $(K,2)$; CLEAVED if $[1,0]$, UNCLEAVED if $[0,1]$  
`<labelorder>`: the indicator of which class is for the columns in `<y>`   
`<sequences>`: the list of sample names   
Where K is the number of samples (graphs), N is the number of nodes, M is the number of edge features, F is the number of node features.
  
Slit data and save their original indices in:  
<trisplit.test.index> and <trisplit.val.index>: indices of samples if triple splitting data into training, validation and test sets. Indices starts from 0.

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

When model has been trained, *train.py* saves the model to the corresponding preset directory (using the flag *--save*).  
I wrote a script to find the model with best accuracy, named as *find_best_acc.py*.  

## Test with the pre-trained model
If you would like to test with already-trained pgcn model, you could use *importance.py* in *model* folder. It will load existed pytorch model file and test data that you specify. Currently, we offer pre-trained models located at *model/outputs* for HCV wild type, HCV A171T, HCV D183A, HCV R170K_A171T_D183A, HCV Combined and TEV Combined.    
```
cd model
python importance.py --dataset ${data} --hidden1 20 --depth 2 --linear 0 --att 0 
--batch_size ${bs} --lr ${lr} --dropout ${dt} --weight_decay ${wd} --seed ${seed} 
--save <pre-trained-model-path> --data_path <where designs are located> --new 
--test_logits_path <where to save outputs>
```

## Variable Importance Analysis (Alternative)
Here we propose a method to represent importance of nodes and edges. We refer variable importance method in random forest and you could use it by using following command:  
```
cd analysis
python importance.py --importance --dataset HCV_ternary_10_ang_aa_energy_7_energyedge_5_hbond --test_dataset HCV_ternary_10_ang_aa_energy_7_energyedge_5_hbond --hidden1 20 --depth 2 --linear 0 --att 0 --batch_size 500 --lr 0.005 --dropout 0.05 --weight_decay 5e-4 --save <model-path>
```
## Comparison with other machine learning methods
In the paper, we compare GCNN + new generated feature set with five machine learning models + traditional feature set. For those results (parameter tuning + train and test) using machine learning models, see [ml-cleavage repository](https://github.com/Nucleus2014/ml-cleavage) in details. 
