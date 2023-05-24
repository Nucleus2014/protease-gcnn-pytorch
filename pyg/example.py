from model import GCN
from torch_geometric.loader import DataLoader
from torch.utils.data.dataset import Dataset
from torch_geometric.utils import scatter
import torch
from torch_geometric.data import Batch, Data


# Example Dataset containing 3 graphs
class MyOwnDataset(Dataset):
    
    def __init__(self) -> None:
        super().__init__()
        self.graphs = [
            Data(x=torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]), edge_index=torch.tensor([[0, 1], [1, 0]]), edge_attr=torch.tensor([[1.0, 2.0], [3.0, 4.0]])),
            Data(x=torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]), edge_index=torch.tensor([[0], [1]]), edge_attr=torch.tensor([[1.0, 2.0]])),
            Data(x=torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]), edge_index=torch.tensor([[0, 1], [1, 0]]), edge_attr=torch.tensor([[1.0, 2.0], [3.0, 4.0]])),
        ]
    def __getitem__(self, index: int) -> Data:
        return self.graphs[index]
    
    def __len__(self) -> int:
        return len(self.graphs)

dataset = MyOwnDataset()
dataLoader = DataLoader(dataset, batch_size=3, shuffle=False) # The dataloader will pack the 3 graphs into a batch

mfeat = 2
nfeat = 3
hidden1 = 10
depth = 10
linear = 0
weight = 'pre'
is_des = False
nclass = 2
dropout = 0.5
model = GCN(2,nfeat,mfeat,hidden1,linear,depth,nclass,dropout,weight,is_des)
for batch in dataLoader:
    print(batch) # (6,2): 6 nodes in all 3 graphs, 2 features
    y = model(batch)
    print(y.shape) # (6,2): 6 nodes in all 3 graphs, 2 classes
    y = scatter(y, batch.batch, dim=0, reduce='mean') # graph level node mean
    print(y.shape) # (3,2): 3 graphs, 2 classes