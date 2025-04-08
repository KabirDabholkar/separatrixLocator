from phase_limit_cycle_RNNs.rnn_scripts.bifurcations import bifurcation, to_torch
from phase_limit_cycle_RNNs.rnn_scripts.utils import weight_scalers_to_1,extract_loadings,orthogonolise_Im,load_rnn,set_dt
import os
import numpy as np
import torch


model_name = "N512_T0217-151523"
device = "cpu"
model_dir = os.getcwd()+"/phase_limit_cycle_RNNs/models/"
out_dir = os.getcwd()+"/phase_limit_cycle_RNNs/data/"


rnn, params, task_params, training_params = load_rnn(
    model_dir + model_name, device=device
)
dt = 0.5
set_dt(task_params, rnn, dt)
#make_deterministic(task_params, rnn)

# Orthogonalise singular vectors and scale input weights
rnn.rnn.svd_orth()
weight_scalers_to_1(rnn)

# Extract connectivity
I, n, m, W = extract_loadings(rnn, orth_I=False, split=True)
alpha, I_orth = orthogonolise_Im(I, m)
alpha, I_orth, m, n = to_torch(alpha, I_orth, m, n, device=device)
freq = 8 #frequency
amp = 1 #

def dynamical_function(x,w=freq):
    k_t = x[...,:2]
    phase_cartsian = x[...,2:] # phase at which you evaluate
    phase = torch.arctan2(phase_cartsian[...,1],phase_cartsian[...,0])

    bifur = bifurcation(alpha, I_orth, m, n, rnn.rnn.tau / 1000, config={})
    x, u = bifur.calc_x_u(amp, w, phase, k_t, stim=torch.zeros(1))
    dK = bifur.dKdt(k_t, x, u)

    dph =  np.pi * 2 * freq

    # Use chain rule to convert dph (scalar) into the derivative in Cartesian coordinates
    dphase_cartesian = torch.stack([-torch.sin(phase), torch.cos(phase)], axis=-1) * dph

    # Return the combined derivative: dK for k_t and dphase_cartesian for the phase
    return torch.concatenate([dK, dphase_cartesian], axis=-1)


if __name__ == "__main__":
    x_init = torch.zeros(4)[None]
    dynamical_function(x_init)
