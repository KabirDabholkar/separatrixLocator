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

if __name__ == '__main__':
    model = GRU_RNN(10,3,3)

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
    print(hidden_distribution_from_model(model, dataset))

