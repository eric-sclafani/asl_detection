from dataclasses import dataclass
import numpy as np
from glob import glob
from typing import Tuple
import torch
from torch.utils.data import Dataset, DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class ASLDataset(Dataset):
    X:np.ndarray
    y:np.ndarray
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx:int):
        return {
            "features" : torch.tensor(self.X[idx], device=DEVICE),
            "label" : torch.tensor([self.y[idx]], device=DEVICE)
        }
        
def get_splits(data_dir:str) -> Tuple[np.ndarray]:
    
    try:
        X_train_path = glob(f"{data_dir}/X_train.npy")[0]
        X_test_path = glob(f"{data_dir}/X_test.npy")[0]
        y_train_path = glob(f"{data_dir}/y_train.npy")[0]
        y_test_path = glob(f"{data_dir}/y_test.npy")[0]
    except IndexError:
        raise FileNotFoundError(f"One or more splits could not be found in provided directory: '{data_dir}'")
    
    return np.load(X_train_path), np.load(X_test_path), np.load(y_train_path), np.load(y_test_path)
