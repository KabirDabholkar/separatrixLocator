import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RNNModel(nn.Module):
    def __init__(self, dt, N, input_size, ramp_train, tau, f0, beta0, theta0,
                 M, eff_dt, h, sigma_noise, init_sigma=0.05, x_init=None, batch_first=False):
        """
        Args:
            dt (float): Simulation time step.
            N (int): Number of neurons.
            input_size (int): Dimension of external input (should be 3).
            ramp_train (float): Scaling factor for ramping input.
            tau (float): Time constant (used in noise scaling).
            f0 (float): Baseline firing rate factor.
            beta0 (float): Gain parameter in the nonlinearity.
            theta0 (float): Threshold parameter in the nonlinearity.
            M (Tensor or array-like): Recurrent weight matrix of shape (N+input_size, N+input_size).
            eff_dt (float): Effective integration time step.
            h (Tensor or array-like): Bias vector of shape (N+input_size,).
            sigma_noise (float): Noise standard deviation (for neurons).
            init_sigma (float): Standard deviation for the initial condition noise.
            x_init (Tensor or array-like): Default initial condition for the neurons (shape (N,)).
            batch_first (bool): If True, inputs/outputs are expected with shape (batch, seq, feature). Default: False.
        """
        super(RNNModel, self).__init__()
        self.dt = dt
        self.N = N
        self.input_size = input_size
        self.ramp_train = ramp_train
        self.tau = tau
        self.f0 = f0
        self.beta0 = beta0
        self.theta0 = theta0
        self.batch_first = batch_first

        # Ensure M is a tensor of shape (N+input_size, N+input_size)
        if not torch.is_tensor(M):
            M = torch.tensor(M, dtype=torch.float32)
        # Make M a learnable parameter.
        self.M = nn.Parameter(M)

        # Ensure h is a tensor of shape (N+input_size,)
        if not torch.is_tensor(h):
            h = torch.tensor(h, dtype=torch.float32)
        self.register_buffer('h', h)

        self.eff_dt = eff_dt
        self.sigma_noise = sigma_noise
        self.init_sigma = init_sigma

        if x_init is None:
            # If no x_init is provided, use zeros (this should normally be overwritten by init_network).
            self.x_init = torch.zeros(self.N, dtype=torch.float32)
        else:
            if not torch.is_tensor(x_init):
                x_init = torch.tensor(x_init, dtype=torch.float32)
            self.x_init = x_init

        # Precompute effective noise standard deviation (applied to the first N entries only)
        self.noise_sigma_eff = (self.dt ** 0.5) * self.sigma_noise / self.tau

    def forward(self, r_in, x_init=None, deterministic=True, batch_first=None, return_hidden=True):
        """
        Simulate the RNN dynamics given an external input sequence.

        Args:
            r_in (Tensor): External input tensor. Expected shapes:
                - If batch_first=False: (seq_len, batch, input_size)
                - If batch_first=True: (batch, seq_len, input_size)
            x_init (Tensor, optional): Initial hidden state of shape (1, batch, N). If provided,
                the neuron initial conditions will be taken as x_init[0]. Defaults to None,
                in which case the internal default (self.x_init) is used.
            deterministic (bool, optional): If True, run dynamics without adding noise.
            batch_first (bool, optional): If provided, overrides the module attribute. If None,
                uses self.batch_first.

        Returns:
            output (Tensor): Firing rates for each time step. Shapes:
                - (seq_len, batch, N) if batch_first is False,
                - (batch, seq_len, N) if batch_first is True.
            h_final (Tensor): Final hidden state (firing rates) of shape (batch, N).
        """
        # Use provided batch_first override if not None.
        if batch_first is None:
            batch_first = self.batch_first

        # Convert input to (seq_len, batch, input_size) if needed.
        if batch_first:
            # Input provided as (batch, seq_len, input_size) -> transpose to (seq_len, batch, input_size)
            r_in = r_in.transpose(0, 1)
            seq_len, batch_size, _ = r_in.size()
        else:
            seq_len, batch_size, _ = r_in.size()

        # Process provided initial condition.
        if x_init is not None:
            # Expect x_init to have shape (1, batch, N); extract the neuron initial state.
            if x_init.dim() == 3 and x_init.size(0) == 1:
                x_neurons = x_init[0]
            else:
                raise ValueError("x_init must have shape (1, batch, N)")
            if not deterministic:
                x_neurons = x_neurons * (1 + self.init_sigma * torch.randn(batch_size, self.N, device=r_in.device))
        else:
            # Expand self.x_init to batch dimension.
            x_neurons = self.x_init.unsqueeze(0).expand(batch_size, self.N)
            if not deterministic:
                x_neurons = x_neurons * (1 + self.init_sigma * torch.randn(batch_size, self.N, device=r_in.device))

        # Initialize extra dimensions (for external inputs) as zeros.
        x_extra = torch.zeros(batch_size, self.input_size, device=r_in.device)
        # Combined state: first N entries for neurons, last input_size for external input (which remain zero)
        x = torch.cat([x_neurons, x_extra], dim=1)  # shape: (batch, N+input_size)

        # Compute initial firing rate from the neuron state.
        r = self.f0 / (1.0 + torch.exp(-self.beta0 * (x[:, :self.N] - self.theta0)))

        outputs = []  # to store firing rates at each time step
        Xs = []
        # Iterate over time steps.
        for t in range(seq_len):
            # r_in[t] is shape (batch, input_size)
            r_in_t = r_in[t]
            # Concatenate current firing rates with external input.
            combined = torch.cat([r, r_in_t], dim=1)  # shape: (batch, N+input_size)
            dx = (-x + torch.matmul(combined, self.M.t()) + self.h) * self.eff_dt

            if deterministic:
                x = x + dx
            else:
                # Add noise only to the neurons.
                noise_neurons = self.noise_sigma_eff * torch.randn(batch_size, self.N, device=r_in.device)
                noise_extra = torch.zeros(batch_size, self.input_size, device=r_in.device)
                noise = torch.cat([noise_neurons, noise_extra], dim=1)
                x = x + dx + noise

            # Update firing rate from the updated neuron state.
            r = self.f0 / (1.0 + torch.exp(-self.beta0 * (x[:, :self.N] - self.theta0)))
            outputs.append(r)

            Xs.append(x[:, :self.N])

        # Stack outputs to form a tensor of shape (seq_len, batch, N)
        outputs = torch.stack(outputs, dim=0)
        Xs = torch.stack(Xs, dim=0)

        if batch_first:
            outputs = outputs.transpose(0, 1)

        if return_hidden:
            return outputs, Xs
        return outputs, r

def init_network(params_dict):
    """
    Initializes the RNN network using parameters extracted from MATLAB files.

    The initial condition x_init is computed as the mean over the first 10 columns of both
    des_out_left and des_out_right:

        x_init = mean([mean(des_out_left(:,1:10),2), mean(des_out_right(:,1:10),2)],2)

    Args:
        params_dict (dict): Dictionary of parameters.

    Returns:
        model (RNNModel): An initialized instance of RNNModel.
    """
    dt = float(params_dict["dt"])
    N = int(params_dict["N"])
    tau = float(params_dict["tau"])
    f0 = float(params_dict["f0"])
    beta0 = float(params_dict["beta0"])
    theta0 = float(params_dict["theta0"])
    ramp_train = float(params_dict["ramp_train"]) if np.isscalar(params_dict["ramp_train"]) else float(
        params_dict["ramp_train"].item())
    eff_dt = float(params_dict["eff_dt"])
    sigma_noise = float(params_dict["sigma_noise_cd"])
    input_size = 3  # external input dimension

    # Extract recurrent weight matrix (for neurons) and bias.
    M_neurons = params_dict["M"]  # shape: (N, N)
    h_neurons = params_dict["h"]  # shape: (N,)

    # M_aug = np.zeros((N + input_size, N + input_size), dtype=np.float32)
    # M_aug[:N, :N] = M_neurons
    M_aug = M_neurons

    # h_aug = np.zeros(N + input_size, dtype=np.float32)
    h_aug = h_neurons

    M_tensor = torch.tensor(M_aug, dtype=torch.float32)
    h_tensor = torch.tensor(h_aug, dtype=torch.float32)

    # Compute x_init from des_out_left and des_out_right.
    trg_left = params_dict["des_out_left"]
    trg_right = params_dict["des_out_right"]
    # Compute the mean of the first 10 columns (MATLAB indices 1:10 correspond to Python 0:10).
    mean_left = np.mean(trg_left[:, :10], axis=1)  # shape (N,)
    mean_right = np.mean(trg_right[:, :10], axis=1)  # shape (N,)
    x_init = np.mean(np.stack([mean_left, mean_right], axis=0), axis=0)  # shape (N,)
    x_init_tensor = torch.from_numpy(x_init).to(torch.float32)
    print(x_init_tensor[:5])

    # Instantiate the model with the computed x_init.
    model = RNNModel(dt=dt, N=N, input_size=input_size, ramp_train=ramp_train, tau=tau,
                     f0=f0, beta0=beta0, theta0=theta0, M=M_tensor,
                     eff_dt=eff_dt, h=h_tensor, sigma_noise=sigma_noise,
                     x_init=x_init_tensor, batch_first=False)
    return model


if "__main__" == __name__:
    torch.manual_seed(2)
    #
    # # Example hyperparameters (you should replace these with your actual values)
    # dt = 0.001
    # N = 100  # number of neurons
    input_dim = 3  # external input dimension
    # ramp_train = 1.0
    # tau = 0.1
    # f0 = 1.0
    # beta0 = 1.0
    # theta0 = 0.5
    # eff_dt = 0.001
    # sigma_noise = 0.01
    # init_sigma = 0.05
    #
    # # M and h should be of shape (N+input_dim, N+input_dim) and (N+input_dim,)
    # M = torch.randn(N + input_dim, N + input_dim)
    # h = torch.randn(N + input_dim)
    #
    # # Optionally set an initial condition for the neurons:
    # x_init = torch.randn(N)
    #
    # # Instantiate the model
    # model = RNNModel(dt, N, input_dim, ramp_train, tau, f0, beta0, theta0,
    #                  M, eff_dt, h, sigma_noise, init_sigma, x_init)
    #
    # # Suppose r_in_cd is your external input tensor of shape (batch_size, 3, T)
    # # For illustration, we create a dummy input:
    # batch_size = 10
    # T = 2000
    # r_in_cd = torch.zeros(batch_size, input_dim, T)
    # # (Fill r_in_cd with your chirp/stimulus/ramp values as needed)
    #
    # # Run the simulation
    # rp_vec_nd = model(r_in_cd)  # rp_vec_nd will have shape (batch_size, N, T)
    #
    # print(rp_vec_nd.shape)


    from load_RNN_ALM_gating import get_params_dict

    model = init_network(get_params_dict())
    model.eval()
    model.batch_first = True
    # Suppose r_in_cd is your external input tensor of shape (batch_size, 3, T)
    # For illustration, we create a dummy input:
    batch_size = 10
    T = 10000
    r_in_cd = torch.zeros(batch_size, T, input_dim)
    x_init = torch.zeros(batch_size, model.N)
    # (Fill r_in_cd with your chirp/stimulus/ramp values as needed)

    # Run the simulation
    rp_vec_nd,_ = model(r_in_cd,deterministic=True)  # rp_vec_nd will have shape (batch_size, N, T)

    print(rp_vec_nd.shape)



    # model(r_in_cd[:,0],x_init=x_init)
    # print()
    # import matplotlib.pyplot as plt
    # plt.plot(rp_vec_nd[0,:2,:].T)
    # plt.show()

    from rnn import get_autonomous_dynamics_from_model

    dynamics = get_autonomous_dynamics_from_model(
        model,rnn_submodule_name=None,kwargs={'deterministic':True,'batch_first':False}
    )
    inp = r_in_cd.swapaxes(0,1)[:1]
    print(
        model(
            inp,
            # x_init=torch.zeros(1, batch_size, model.N),
        ),
        dynamics(x_init).shape
    )