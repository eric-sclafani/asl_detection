#!/usr/bin/env python3

import argparse
import torch
from torch import nn

from data_loader import ASLDataset, DataLoader, get_splits

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"





def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-d",
                        "--splits_dir",
                        help="Directory with train and test splits (named accordingly)")
    args = parser.parse_args()
    
    X_train, X_test, y_train, y_test = get_splits(args.splits_dir)
    
    train_dataloader = DataLoader(dataset=ASLDataset(X_train, y_train),
                                  batch_size=32,
                                  shuffle=True)
    test_dataloader = DataLoader(dataset=ASLDataset(X_test, y_test),
                                  batch_size=32,
                                  shuffle=True)
    

    
    


if __name__ == "__main__":
    main()