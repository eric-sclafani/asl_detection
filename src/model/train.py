#!/usr/bin/env python3

import argparse
from tqdm.auto import tqdm
import time
import torch
from torch import nn
from torch.optim import SGD, Adam
from torchmetrics import Accuracy
from torch.utils.data import DataLoader

from data_loader import ASLDataset, get_splits
from torch_models import ASLClassifierBaseline, ASLClassifierCNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

def measure_time(func):
    """Debugging function for measuring function execution time"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {execution_time:.6f} seconds.")
        return result
    return wrapper

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
        train_acc += accuracy(y_pred, y).item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")
    
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
            test_acc += accuracy(test_pred, y).item()
        
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")

@measure_time
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
                        type=int,
                        required=True) 
    parser.add_argument("-c",
                        "--num_classes",
                        help="Number of classes to predict",
                        type=int,
                        required=True) 
    parser.add_argument("-e",
                        "--epochs",
                        help="Number of epochs",
                        type=int,
                        required=True) 
    parser.add_argument("-lr",
                        "--lr",
                        type=float,
                        help="Learning rate",
                        required=True) 

    args = parser.parse_args()
    bs = args.batch_size
    epochs = args.epochs
    lr = args.lr
    num_classes = args.num_classes
    
    X_train, X_test, y_train, y_test = get_splits(args.splits_dir)
    
    train_dataloader = DataLoader(dataset=ASLDataset(X_train, y_train),
                                  batch_size=bs,
                                  shuffle=True)
    test_dataloader = DataLoader(dataset=ASLDataset(X_test, y_test),
                                  batch_size=bs,
                                  shuffle=True)
    
    model_baseline = ASLClassifierBaseline(input_shape=3072, 
                                           hidden_units=15,
                                           output_shape=num_classes)
    
    loss = nn.CrossEntropyLoss()
    optimizer = Adam(params=model_baseline.parameters(),
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
    print("Saving torch model...")
    scripted_model = torch.jit.script(model_baseline)
    scripted_model.save("src/model/asl_baseline.pt")
    


if __name__ == "__main__":
    main()