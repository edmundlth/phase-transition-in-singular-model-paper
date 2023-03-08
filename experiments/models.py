import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.optim.optimizer import Optimizer



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


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        super(MyDataset, self).__init__()
        
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]

    

class MLP_Network(nn.Module):
    def __init__(
        self, 
        dim_input=3, 
        dim_output=23, 
        hidden_layer_sizes=[64, 128, 128, 64], 
        activation=None,
        require_bias=False,
        device=None,
    ):
        super(MLP_Network, self).__init__()
        
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation if activation is not None else nn.GELU()
        
        self.layers = nn.ModuleList()
        input_size = dim_input
        for size in hidden_layer_sizes:
            self.layers.append(nn.Linear(input_size, size, bias=require_bias))
            input_size = size
            self.layers.append(self.activation)
        self.layers.append(nn.Linear(input_size, dim_output, bias=require_bias))
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        self.to(self.device)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

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

###########################################################################
###########################################################################
class OneHiddenLayerTanhNetwork(nn.Module):
    """
    1-hidden layer tanh network with no bias terms. 
    """
    def __init__(self, input_dim=1, output_dim=1, H=1, init_param=None, activation=None):
        super(OneHiddenLayerTanhNetwork, self).__init__()
        self.fc_1 = nn.Linear(input_dim, H, bias=False)
        self.fc_2 = nn.Linear(H, output_dim, bias=False)
        
        if activation is None:
            self.activation = torch.tanh
        else:
            self.activation = activation

        if init_param is not None:
            self.set_parameters(init_param)

    def forward(self, x):
        x = self.activation(self.fc_1(x))
        x = self.fc_2(x)
        return x
    
    def loglik(self, data, sigma=1.0, inverse_temp=1.0):
        # Evaluates the log of the likelihood given a set of X and Y
        X, y = data[:]
        yhat = self.forward(X)
        logprob = torch.distributions.Normal(y, sigma).log_prob(yhat)
        return logprob.sum() * inverse_temp

    def logprior(self):
        # Return a scalar of 0 since we have uniform prior
        return torch.tensor(0.0)
    
    def logpost_unnormalised(self, data):
        return self.loglik(data) + self.logprior()

    def set_parameters(self, parameters):
        state_dict = self.state_dict()
        for i in [0, 1]:
            name = f"fc_{i + 1}.weight"
            shape = state_dict[name].shape
            state_dict[name] = torch.tensor(parameters[i], requires_grad=True).reshape(shape)
        self.load_state_dict(state_dict)
        return 

    def flatten_parameter(self):
        return torch.vstack([
            self.get_parameter(f"fc_{i + 1}.weight").flatten() for i in [0, 1]
        ]).T.ravel()
    

class LearningMachine(object):
    def __init__(
        self, 
        model, 
        train_dataset, 
        test_dataset,
        loss_fn, 
        optimiser, 
        batch_size=None,
        batch_fraction=None,
        device="cpu", 
    ):
        self.device = device
        # model
        self.model = model
        self.model.to(device)
        
        # dataset
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        # set batch size
        if batch_size is not None:
            self.batch_size = batch_size
        elif batch_fraction is not None:
            self.batch_size = max(int(len(self.train_dataset) * batch_fraction), 1)
        else:
            self.batch_size = max(len(self.train_dataset) // 5)
            
        # dataloaders
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size, 
            shuffle=True
        )
        self.test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        # loss function
        self.loss_fn = loss_fn
        
        # training config
        self.optimiser = optimiser
        self.epoch = 0
        self.num_gradient_steps = 0
        self.train_loss = self.compute_loss(self.train_dataset)
        self.history = {}
        self.snapshot()
        
    def training_loop(self, epoch, stopping_condition=None):
        while self.epoch < epoch:
            train_loss = self.train(self.train_dataloader)
            self.epoch += 1
            
            if stopping_condition is not None:
                if stopping_condition(self):
                    break
        return 
    
    def train(self, dataloader):
        self.model.train()
        for batch, (X, y) in enumerate(dataloader):
#             X, y = X.to(self.device), y.to(self.device)
            # Compute prediction error
            pred = self.model(X)
            self.train_loss = self.loss_fn(pred, y)
            
            # Backpropagation
            self.optimiser.zero_grad()
            # TODO: Find out what went wrong here.. retain_graph=True shouldn't be needed.
            self.train_loss.backward(retain_graph=True) 
            self.optimiser.step()
            self.num_gradient_steps += 1
            # Might be a bit too much to snapshot every gradient step... 
            self.snapshot()
        return self.train_loss.item()
    
    def snapshot(self, include_test_loss=False, include_train_loss=True):        
        info_bundle = {
            "parameter": self.model.flatten_parameter(),
            "epoch": self.epoch, 
        }
        if include_test_loss: 
            info_bundle["test_loss"] = self.compute_loss(self.test_dataset)
        if include_train_loss:
            info_bundle["train_loss"] = self.train_loss
#             info_bundle["train_loss"] = self.compute_loss(self.train_dataset)
        self.history[self.num_gradient_steps] = info_bundle
        return info_bundle
        
    def compute_loss(self, dataset):
        X, y = dataset[:]
        pred = self.model(X)
        return self.loss_fn(pred, y).item()
    
    def test(self):
        return self.compute_loss(self.test_dataset)
    
    def loglikelihood(self, new_state_dict, ):
        return 
        

OPTIMISER_SWITCH = {
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD, 
}
class EnsembleModel(object):
    
    def __init__(
        self, 
        model_list, 
        loss_fn, 
        learning_rate, 
        train_dataset, 
        test_dataset, 
        optimiser_type="adam", 
        batch_size=None,
        batch_fraction=None,
    ):
        self.model_list = model_list
        self.learning_rate = learning_rate, 
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.loss_fn = loss_fn
        
        self.optimiser_class = OPTIMISER_SWITCH[optimiser_type.lower()]
        
        self.learning_machines = []
        for model in self.model_list:
            machine = LearningMachine(
                model, 
                self.train_dataset,
                self.test_dataset, 
                loss_fn, 
                self.optimiser_class(model.parameters(), lr=learning_rate), 
                batch_size=batch_size,
                batch_fraction=batch_fraction,
            )
            self.learning_machines.append(machine)
        
        self.history = {}
    
    def __getitem__(self, index):
        return self.learning_machines[index]
    
    def predict(self, x):
        return torch.mean([model(x) for model in self.model_list])

    def get_all_parameters(self):
        return torch.vstack([model.flatten_parameter() for model in self.model_list])
    
    def train_ensemble(self, num_epoch):
        for machine in self.learning_machines:
            machine.training_loop(num_epoch)
        return
    
    
    

class TrainOptimizerSGLD(Optimizer):
    def __init__(self, net, alpha=1e-4):
        super(TrainOptimizerSGLD, self).__init__(net.parameters(), {})
        self.net = net
        self.alpha = alpha
        
        self.original_params = dict(self.net.named_parameters())
    
    def step(self, batch, batch_size=1, num_data=1, inverse_temp=1.0, restoring_force=0):
        self.zero_grad()
        weight = num_data / batch_size
  
        loss = -self.net.loglik(batch, inverse_temp=inverse_temp) * weight - self.net.logprior()
        # Should introduce some sort of restoring force to prevent trajectory wandering far away
        # Force should be zero near the origin, so a bump function of sort... 
        if restoring_force > 0:
            sqdist = 0
            for name, param in self.net.named_parameters():
                sqdist += torch.sum((param - self.original_params[name])**2)
            loss -= restoring_force * sqdist
        loss.backward(retain_graph=True)
        
        with torch.no_grad():
            for name, param in self.net.named_parameters():
                newparam = (
                    param - 0.5 * self.alpha * param.grad 
                    + torch.normal(torch.zeros_like(param), std=self.alpha)
                )
                param.copy_(newparam)
        return loss

    def fit(self, data=None, num_steps=1000, inverse_temp=1.0, restoring_force=0):
        # We create one tensor per parameter so that we can keep track of the parameter values over time:
        self.parameter_trace = {
            key : torch.zeros( (num_steps,) + par.size()) 
            for key, par in self.net.named_parameters()
        }

        for s in range(num_steps):
            loss = self.step(data, inverse_temp=inverse_temp, restoring_force=restoring_force)
            for key, val in self.net.named_parameters():
                self.parameter_trace[key][s,] = val.data
    
                
    