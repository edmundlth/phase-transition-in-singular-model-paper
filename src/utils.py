import numpy as np

import torch
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
from torch.optim.optimizer import Optimizer

class EarlyStopping(object):
    def __init__(self, patience=5, epsilon=1e-4):
        self.patience = patience
        self.epsilon = epsilon
        self.history = []
        
        
    def early_stopping_by_train_loss_improvement(self, machine):
        with torch.no_grad():
            current_step_count = max(machine.history.keys())
            machine_history = machine.history[current_step_count]
            current_epoch = machine_history["epoch"]
            train_loss = machine_history["train_loss"].item()
            self.history.append(train_loss)
            if len(self.history) > self.patience and np.mean(np.abs(np.diff(self.history[-self.patience:]))) < self.epsilon:
                print(f"Stopping at: {current_step_count} (epoch: {current_epoch}).")
                return True
            return False


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


# def train(dataloader, model, loss_fn, optimiser, device="cpu"):
#     size = len(dataloader.dataset)
#     model.train()
#     for batch, (X, y) in enumerate(dataloader):
#         X, y = X.to(device), y.to(device)

#         pred = model(X)
#         loss = loss_fn(pred, y)

#         optimiser.zero_grad()
#         loss.backward()
#         optimiser.step()

#         model.w_history.append((model.w1.item(), model.w2.item()))
#         model.loss_history.append(loss.item())
#     return


# def test(dataloader, model, loss_fn, device):
#     size = len(dataloader.dataset)
#     num_batches = len(dataloader)
#     model.eval()
#     test_loss = 0
#     with torch.no_grad():
#         for X, y in dataloader:
#             X, y = X.to(device), y.to(device)
#             pred = model(X)
#             test_loss += loss_fn(pred, y).item()
#     test_loss /= num_batches
#     return

# def default_regression_fn(x, w1, w2):
#     actual = (
#         w2
#         * torch.tanh((w1 - 1) * x)
#         * torch.tanh((w1 + 1) * x)
#         * torch.tanh((w2 - (w1 + 1) ** 2) * x)
#     )
#     local_minima_term = 0.0
#     return actual + local_minima_term