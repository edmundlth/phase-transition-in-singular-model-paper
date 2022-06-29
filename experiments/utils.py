import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        super(MyDataset, self).__init__()
        
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        index = torch.randperm(len(self.X))[index]
        return self.X[index], self.y[index]



def regression_fn(x, w1, w2):
    actual = (
        w2 * 
        torch.tanh((w1 -1) * x) * 
        torch.tanh((w1 + 1) * x) * 
        torch.tanh((w2 - (w1 + 1)**2) * x) 
    )
    local_minima_term = 0.0
    return actual + local_minima_term


class Model(nn.Module):
    def __init__(self, a=0.5, b=0.5, regression_fn=regression_fn):
        super(Model, self).__init__()
        self.regression_fn = regression_fn
        self.w1 = torch.nn.Parameter(torch.tensor(float(a)))
        self.w2 = torch.nn.Parameter(torch.tensor(float(b)))
        
        self.w_history = [(self.w1.item(), self.w2.item())]
        self.loss_history = []
        
        
    def forward(self, x):
        return self.regression_fn(x, self.w1, self.w2)


class FeedForwardNetwork(nn.Module):
    def __init__(self, a=0.5, b=0.5):
        super(FeedForwardNetwork, self).__init__()
        self.w1 = torch.nn.Parameter(torch.tensor(float(a)))
        self.w2 = torch.nn.Parameter(torch.tensor(float(b)))
        
        self.w_history = [(self.w1.item(), self.w2.item())]
        self.loss_history = []
        
        
    def forward(self, x):
        return self.w2 * torch.tanh((self.w1 -1) * x) * torch.tanh((self.w1 + 1) * x) * torch.tanh((self.w2 - (self.w1 + 1)**2) * x)


def train(dataloader, model, loss_fn, optimiser, device="cpu"):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        pred = model(X)
        loss = loss_fn(pred, y)
        
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        
        model.w_history.append((model.w1.item(), model.w2.item()))
        model.loss_history.append(loss.item())
    return

            
            
def test(dataloader, model, loss_fn):
    return 
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader: 
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    return

