#!/usr/bin/env python3

import argparse
from tqdm import tqdm
import torch
from torch import nn
from torch.optim import SGD, Adam
from torchmetrics import Accuracy

from data_loader import ASLDataset, DataLoader, get_splits, measure_time
from models import ASLCLassifierBaseline, ASLClassifierCNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@measure_time
def train_model(model: nn.Module,
                data_loader: DataLoader,
                loss_fn: nn.Module,
                optimizer: torch.optim.Optimizer,
                accuracy:Accuracy,
                device: torch.device = DEVICE):
    train_loss, train_acc = 0, 0
    model.to(device)
    model.train()
    for X, y in data_loader:
        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy(y_true=y, y_pred=y_pred.argmax(dim=1)).item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")
    
@measure_time
def test_model(data_loader: DataLoader,
                model: nn.Module,
                loss_fn: nn.Module,
                accuracy:Accuracy,
                device: torch.device = DEVICE):
    
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval() 
    with torch.inference_mode(): 
        for X, y in data_loader:
            test_pred = model(X)
            
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy(y_true=y, y_pred=test_pred.argmax(dim=1)).item()
        
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")

def run_pipeline(epochs:int,
                 model:nn.Module,
                 train_dataloader:DataLoader,
                 test_dataloader:DataLoader,
                 loss_fn:nn.Module,
                 optimizer:torch.optim.Optimizer,
                 accuracy:Accuracy):
    
    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch}\n---------")
        train_model(data_loader=train_dataloader, 
                    model=model, 
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    accuracy=accuracy
                    )
        test_model(data_loader=test_dataloader,
                   model=model,
                   loss_fn=loss_fn,
                   accuracy=accuracy
                   )

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-d",
                        "--splits_dir",
                        help="Directory with train and test splits (named accordingly)",
                        required=True)
    parser.add_argument("-bs",
                        "--batch_size",
                        help="Batch size for data loader",
                        required=True) 
    parser.add_argument("-c",
                        "--num_classes",
                        help="Number of classes to predict",
                        required=True) 
    parser.add_argument("-e",
                        "--epochs",
                        help="Number of epochs",
                        required=True) 
    parser.add_argument("-lr",
                        "--lr",
                        help="Learning rate",
                        required=True) 

    args = parser.parse_args()
    bs = int(args.batch_size)
    epochs = int(args.epochs)
    lr = float(args.lr)
    num_classes = int(args.num_classes)
    
    X_train, X_test, y_train, y_test = get_splits(args.splits_dir)
    
    train_dataloader = DataLoader(dataset=ASLDataset(X_train, y_train),
                                  batch_size=bs,
                                  shuffle=True)
    test_dataloader = DataLoader(dataset=ASLDataset(X_test, y_test),
                                  batch_size=bs,
                                  shuffle=True)
    
    model_baseline = ASLCLassifierBaseline(input_shape=1024, 
                                           hidden_units=10,
                                           output_shape=num_classes)
    
    loss = nn.CrossEntropyLoss()
    optimizer = SGD(params=model_baseline.parameters(),
                    lr=lr)
    accuracy = Accuracy(task="multiclass", num_classes=num_classes)
    
    run_pipeline(epochs,
                 model_baseline,
                 train_dataloader,
                 test_dataloader,
                 loss,
                 optimizer,
                 accuracy
                 )


if __name__ == "__main__":
    main()