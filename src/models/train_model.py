#!/usr/bin/env python3

import argparse
import torch
from torch import nn

from data_loader import ASLDataset, DataLoader, get_splits

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ASLCLassifierBaseline(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units), 
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
            # nn.ReLU(),
        )
    
    def forward(self, x:torch.Tensor):
        return self.layer_stack(x)

# class ASLClassifierCNN(nn.Module):
    
#     def __init__(self, input_shape:int, hidden_units:int, output_shape:int):
#         super().__init__()
        
#         self.block_1 = nn.Sequential(
            
#         )




def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-d",
                        "--splits_dir",
                        help="Directory with train and test splits (named accordingly)",
                        required=True)
    parser.add_argument("-c",
                        "--num_classes",
                        help="Number of classes for predicting",
                        required=True) 

    args = parser.parse_args()
    
    X_train, X_test, y_train, y_test = get_splits(args.splits_dir)
    
    train_dataloader = DataLoader(dataset=ASLDataset(X_train, y_train),
                                  batch_size=32,
                                  shuffle=True)
    test_dataloader = DataLoader(dataset=ASLDataset(X_test, y_test),
                                  batch_size=32,
                                  shuffle=True)
    
    model_baseline = ASLCLassifierBaseline(input_shape=1024, 
                                           hidden_units=10,
                                           output_shape=int(args.num_classes)).to(DEVICE)
    
    print(model_baseline)


if __name__ == "__main__":
    main()