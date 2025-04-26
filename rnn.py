from torch import nn
import torch
import numpy as np
from functools import wraps

def discrete_to_continuous(discrete_dynamics, delta_t=1):
    @wraps(discrete_dynamics)
    def continuous_dynamics(*args):
        return (discrete_dynamics(*args) - args[0]) / delta_t
    return continuous_dynamics

def multidimbatch(func):
    def new_func(*args):
        # new_inp = inp.reshape(-1,inp.shape[-1])
        new_args = [arg.reshape(-1,arg.shape[-1]) for arg in args]

        new_out = func(*new_args)
        out = new_out.reshape(*args[0].shape[:-1],new_out.shape[-1])
        return out
    return new_func

def set_model_with_checkpoint(model,checkpoint):
    model.load_state_dict(checkpoint) #['model_state_dict'])
    return model

def get_autonomous_dynamics_from_model(model,device='cpu',rnn_submodule_name='rnn',kwargs={},output_id=0):
    @multidimbatch
    def dynamics(hx,inp=None):
        submodule = model
        if rnn_submodule_name is not None:
            submodule = getattr(model,rnn_submodule_name)
        hx = hx[None]
        inp = torch.zeros_like(hx)[..., :submodule.input_size] if inp is None else inp[None]
        submodule.to(hx.device)
        output = submodule(inp,hx,**kwargs)[output_id][0]
        return output
    return dynamics

def extract_hidden_from_model(model, dataset):
    with torch.no_grad():
        inputs, _ = dataset()
        _, hidden = model(torch.tensor(inputs, dtype=torch.float32), return_hidden=True)
    return hidden

def extract_after(data, after):
    return data[:, after:, :]


def reshape_hidden(hidden):
    return hidden.reshape(-1, hidden.shape[-1])

def hidden_distribution(hidden, alpha=1e-4):
    hidden = reshape_hidden(hidden)
    mean = hidden.mean(0)
    cov = torch.cov((hidden - mean[None]).T)
    cov += torch.eye(cov.shape[0]) * alpha
    return torch.distributions.MultivariateNormal(mean, cov)

def hidden_distribution_with_spectral_norm(hidden, scale=1):
    if len(hidden.shape)>2:
        hidden = reshape_hidden(hidden)
    mean = hidden.mean(0)
    cov = torch.cov((hidden - mean[None]).T)
    spectral_norm = torch.linalg.norm(cov, ord=2)
    cov = torch.eye(cov.shape[0]) * spectral_norm * scale
    return torch.distributions.MultivariateNormal(mean, cov)

def get_spectral_norm(hidden):
    """
    Calculate the spectral norm of the covariance matrix of hidden states.
    
    Args:
        hidden: Tensor of shape (batch_size, seq_len, hidden_dim) or (batch_size, hidden_dim)
    
    Returns:
        spectral_norm: Scalar value representing the spectral norm of the covariance matrix
    """
    if len(hidden.shape)>2:
        hidden = reshape_hidden(hidden)
    mean = hidden.mean(0)
    cov = torch.cov((hidden - mean[None]).T)
    spectral_norm = torch.linalg.norm(cov, ord=2)
    return float(spectral_norm)


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
        # Initialize weights with normal distribution scaled by 1/sqrt(num_h)
        # nn.init.normal_(self.linear.weight, mean=0, std=1/num_h)

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


    hx = torch.zeros((1, 2, 10))
    inp = torch.ones((1, 2, 3))
    new_model = convert_to_perturbableRNN(model)
    print(
        model.rnn(inp, hx)
    )

    print(model.rnn.batch_first)