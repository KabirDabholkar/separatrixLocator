from torch import nn
import torch
import numpy as np
from functools import wraps

def discrete_to_continuous(discrete_dynamics):
    @wraps(discrete_dynamics)
    def continuous_dynamics(x):
        return discrete_dynamics(x) - x
    return continuous_dynamics
def multidimbatch(func):
    def new_func(inp):
        new_inp = inp.reshape(-1,inp.shape[-1])
        new_out = func(new_inp)
        out = new_out.reshape(*inp.shape[:-1],new_out.shape[-1])
        return out
    return new_func

def set_model_with_checkpoint(model,checkpoint):
    model.load_state_dict(checkpoint) #['model_state_dict'])
    return model
def get_autonomous_dynamics_from_model(model,device='cpu'):
    @multidimbatch
    def dynamics(hx):
        hx = hx[None]
        inp = torch.zeros_like(hx)[..., :model.rnn.input_size]
        model.to(device)
        output = model.rnn(inp,hx)[0][0]
        model.to('cpu')
        return output
    return dynamics

def hidden_distribution_from_model(model,dataset, alpha = 1e-4):
    inputs,targets = dataset()
    _,hidden = model(torch.tensor(inputs),return_hidden=True)
    hidden = hidden.reshape(-1,hidden.shape[-1])
    print(hidden.shape)
    mean = hidden.mean(0)
    cov = torch.cov((hidden-mean[None]).T)
    cov += torch.eye(cov.shape[0]) * alpha
    return torch.distributions.MultivariateNormal(mean,cov)

class GRU_RNN(nn.Module):
    def __init__(self, num_h, ob_size, act_size):
        super(GRU_RNN, self).__init__()
        self.rnn = nn.GRU(ob_size, num_h)
        self.linear = nn.Linear(num_h, act_size)

    def forward(self, x, return_hidden = False):
        out, hidden = self.rnn(x)
        x = self.linear(out)
        if return_hidden:
            return x, out
        return x

class RNN(nn.Module):
    def __init__(self, num_h, ob_size, act_size, RNN_class='RNN'):
        super().__init__()
        self.rnn = getattr(nn,RNN_class)(ob_size, num_h)
        self.linear = nn.Linear(num_h, act_size)

    def forward(self, x, return_hidden = False):
        out, hidden = self.rnn(x)
        x = self.linear(out)
        if return_hidden:
            return x, out
        return x


def convert_to_perturbableRNN(old_model):
    # Extract the original RNN and its parameters.
    old_rnn = old_model.rnn
    # RNN_class is the string name like 'RNN' or 'LSTM' used to create the module.
    RNN_class = type(old_rnn).__name__
    old_input_size = old_rnn.input_size
    hidden_size = old_rnn.hidden_size
    act_size = old_model.linear.out_features

    # New input size is the old input size plus the hidden size.
    new_input_size = old_input_size + hidden_size

    # Create a new RNN module with the updated input size.
    new_rnn = getattr(nn, RNN_class)(new_input_size, hidden_size)

    # Copy the hidden-to-hidden weights (and bias, if they exist)
    new_rnn.weight_hh_l0.data.copy_(old_rnn.weight_hh_l0.data)
    if hasattr(old_rnn, 'bias_hh_l0') and old_rnn.bias_hh_l0 is not None:
        new_rnn.bias_hh_l0.data.copy_(old_rnn.bias_hh_l0.data)

    # Create the new weight_ih by concatenating the old weight with an identity matrix.
    # old_rnn.weight_ih_l0 has shape (hidden_size, old_input_size)
    # torch.eye(hidden_size) has shape (hidden_size, hidden_size)
    # The new weight_ih will have shape (hidden_size, old_input_size + hidden_size)
    identity = torch.eye(hidden_size, device=old_rnn.weight_ih_l0.data.device)
    new_weight_ih = torch.cat([old_rnn.weight_ih_l0.data, identity], dim=1)
    new_rnn.weight_ih_l0.data.copy_(new_weight_ih)

    # Copy bias_ih if available.
    if hasattr(old_rnn, 'bias_ih_l0') and old_rnn.bias_ih_l0 is not None:
        new_rnn.bias_ih_l0.data.copy_(old_rnn.bias_ih_l0.data)

    # Now create a new instance of the overall model with the new input size.
    # Note: The first argument is the hidden size (num_h), and the second is the input size.
    new_model = type(old_model)(hidden_size, new_input_size, act_size, RNN_class=RNN_class)
    new_model.rnn = new_rnn
    new_model.linear = old_model.linear  # keeping the same output layer

    return new_model

if __name__ == '__main__':
    # nn.RNN
    model = RNN(10,3,3, RNN_class='RNN')

    inp = torch.zeros((1, 5, 3))
    hx = torch.zeros((1, 5, 10))
    out,last = model.rnn(inp, hx)
    print(out.shape)

    hx = torch.zeros((5, 10))
    dynamics = get_autonomous_dynamics_from_model(model)
    print(dynamics(hx).shape)

    inp = torch.ones((10, 5, 3))
    def dataset():
        return (np.array(inp),None)

    print(
        hidden_distribution_from_model(model, dataset)
    )

    new_model = convert_rnn(model)
    print(
        new_model
    )

