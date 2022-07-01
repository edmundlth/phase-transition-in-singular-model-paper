import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal

######################################################
# Pytorch Dataset
######################################################


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


######################################################
# Helper functions
######################################################
def make_dataset(model, n, sigma=0.1):
    uniform_dist = Uniform(torch.tensor([-1.0]), torch.tensor([1.0]))
    X = uniform_dist.sample([n])
    y = model(X)
    normal_dist = Normal(0.0, sigma)
    noise = normal_dist.sample(y.shape)
    y += noise
    return X, y


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


def test(dataloader, model, loss_fn, device):
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
