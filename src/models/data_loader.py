from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class ASLDataset(Dataset):
    X:np.ndarray
    y:np.ndarray
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx:int):
        return {
            "features" : torch.Tensor(self.X[idx]),
            "label" : torch.FloatTensor([self.y[idx]])
        }
        

X,y = np.load("data/splits/X_train.npy"), np.load("data/splits/y_train.npy")
dataset = ASLDataset(X,y)  
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
import ipdb;ipdb.set_trace()