import torch
from torch import nn


######################################################
# Models
######################################################

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

class Custom2DRegressionModel(nn.Module):
    def __init__(self, a=0.5, b=0.5, regression_fn=None):
        super(Custom2DRegressionModel, self).__init__()
        self.regression_fn = regression_fn
        self.w1 = torch.nn.Parameter(torch.tensor(float(a)))
        self.w2 = torch.nn.Parameter(torch.tensor(float(b)))

        self.w_history = [(self.w1.item(), self.w2.item())]
        self.loss_history = []

    def forward(self, x):
        return self.regression_fn(x, self.w1, self.w2)

class OneHiddenLayerFeedForwardNetwork(nn.Module):
    """
    1-hidden layer tanh network with no bias terms. 
    """
    def __init__(self, input_dim=1, output_dim=1, H=1, init_param=None, activation=None):
        super(OneHiddenLayerFeedForwardNetwork, self).__init__()
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
    