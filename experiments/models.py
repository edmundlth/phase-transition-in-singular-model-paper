import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


######################################################
# Models
######################################################

def default_regression_fn(x, w1, w2):
    actual = (
        w2
        * torch.tanh((w1 - 1) * x)
        * torch.tanh((w1 + 1) * x)
        * torch.tanh((w2 - (w1 + 1) ** 2) * x)
    )
    local_minima_term = 0.0
    return actual + local_minima_term



class Model(nn.Module):
    def __init__(self, a=0.5, b=0.5, regression_fn=default_regression_fn):
        super(Model, self).__init__()
        self.regression_fn = regression_fn
        self.w1 = torch.nn.Parameter(torch.tensor(float(a)))
        self.w2 = torch.nn.Parameter(torch.tensor(float(b)))

        self.w_history = [(self.w1.item(), self.w2.item())]
        self.loss_history = []

    def forward(self, x):
        return self.regression_fn(x, self.w1, self.w2)


class OneHiddenLayerTanhNetwork(nn.Module):
    """
    1-hidden layer tanh network with no bias terms. 
    """
    def __init__(self, input_dim=1, output_dim=1, H=1, init_param=None):
        super(OneHiddenLayerTanhNetwork, self).__init__()
        self.fully_connected_1 = nn.Linear(input_dim, H, bias=False)
        self.fully_connected_2 = nn.Linear(H, output_dim, bias=False)

        if init_param is not None:
            self.set_parameters(init_param)

    def forward(self, x):
        x = torch.tanh(self.fully_connected_1(x))
        x = self.fully_connected_2(x)
        return x

    def set_parameters(self, parameters):
        state_dict = self.state_dict()
        for i in [0, 1]:
            name = f"fully_connected_{i + 1}.weight"
            shape = state_dict[name].shape
            state_dict[name] = torch.tensor(parameters[i], requires_grad=True).reshape(shape)
        self.load_state_dict(state_dict)
        return 
