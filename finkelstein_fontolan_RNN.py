import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from compose import compose

def convert_to_perturbable_RNNModel(old_model):
    """
    Convert an RNNModel to accept perturbations by expanding its input size.
    The new model will accept inputs of size (old_input_size + N).
    
    Args:
        old_model (RNNModel): Original RNNModel instance
        
    Returns:
        RNNModel: New model with expanded input size
    """
    # Extract parameters from old model
    old_N = old_model.N
    old_input_size = old_model.input_size
    new_input_size = old_input_size + old_N
    
    # Get the old weight matrix M
    old_M = old_model.M.data
    
    # Create new M matrix with expanded input section
    new_M = torch.zeros(old_N + new_input_size, old_N + new_input_size, 
                       device=old_M.device)
    
    # Copy the neuron-to-neuron weights (top-left block)
    new_M[:old_N, :old_N] = old_M[:old_N, :old_N]
    
    # Copy the old input-to-neuron weights (top-middle block)
    new_M[:old_N, old_N:old_N+old_input_size] = old_M[:old_N, old_N:old_N+old_input_size]
    
    # Add identity mapping for perturbation inputs (top-right block)
    new_M[:old_N, old_N+old_input_size:] = torch.eye(old_N, device=old_M.device)
    
    # Copy the old input feedback weights if they exist (middle blocks)
    new_M[old_N:old_N+old_input_size, :old_N] = old_M[old_N:, :old_N]
    new_M[old_N:old_N+old_input_size, old_N:old_N+old_input_size] = old_M[old_N:, old_N:]
    
    # Create new bias vector h with expanded size
    new_h = torch.zeros(old_N + new_input_size, device=old_model.h.device)
    new_h[:len(old_model.h)] = old_model.h
    
    # Create new model with expanded input size
    new_model = RNNModel(
        dt=old_model.dt,
        N=old_N,
        input_size=new_input_size,
        ramp_train=old_model.ramp_train,
        tau=old_model.tau,
        f0=old_model.f0,
        beta0=old_model.beta0,
        theta0=old_model.theta0,
        M=new_M,
        eff_dt=old_model.eff_dt,
        h=new_h,
        sigma_noise=old_model.sigma_noise,
        init_sigma=old_model.init_sigma,
        batch_first=old_model.batch_first
    )
    
    return new_model


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
        super().__init__()
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
            M = torch.tensor(M, dtype=torch.float32, requires_grad=True)
        # Make M a learnable parameter.
        self.M = nn.Parameter(M)

        # Ensure h is a tensor of shape (N+input_size,)
        if not torch.is_tensor(h):
            h = torch.tensor(h, dtype=torch.float32)
            h = nn.Parameter(h, requires_grad=True)
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

def init_network(params_dict,device=torch.device('cpu')):
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
    # print(x_init_tensor[:5])

    # Instantiate the model with the computed x_init.
    model = RNNModel(dt=dt, N=N, input_size=input_size, ramp_train=ramp_train, tau=tau,
                     f0=f0, beta0=beta0, theta0=theta0, M=M_tensor,
                     eff_dt=eff_dt, h=h_tensor, sigma_noise=sigma_noise,
                     x_init=x_init_tensor, batch_first=False).to(device)
    return model


def get_points_on_opposite_attractors(inputs,hidden,input_range = (0.9,0.92), return_separately=False):
    """
    Get points on the attractors of the RNN model.

    Args:
        inputs (Tensor): External input tensor of shape (batch_size, T, input_dim).
        hidden (Tensor): Hidden state tensor of shape (batch_size, T, N).
    """

    top_indices = np.where(
        (input_range[0] <= inputs[:, :, 2]) & (inputs[:, :, 2] <= input_range[1])
    )

    # Extract corresponding hidden states for those indices
    top_hidden = hidden[top_indices[0], top_indices[1], :].detach().cpu().numpy()

    # Compute pairwise euclidean distances between all points in top_hidden using torch
    distances = torch.cdist(
        torch.from_numpy(top_hidden), 
        torch.from_numpy(top_hidden)
    )
    
    # Get indices of points with maximum distance
    max_dist_idx = np.unravel_index(torch.argmax(distances), distances.shape)
    
    # Get the actual points with maximum distance
    point1 = top_hidden[max_dist_idx[0]]
    point2 = top_hidden[max_dist_idx[1]]
    max_distance = distances[max_dist_idx[0], max_dist_idx[1]].item()
    
    # print(f"Maximum distance: {max_distance}")

    average_inputs = (inputs[top_indices[0][max_dist_idx[0]], top_indices[1][max_dist_idx[0]]] + 
                            inputs[top_indices[0][max_dist_idx[1]], top_indices[1][max_dist_idx[1]]]) / 2
    if return_separately:
        return point1, point2, average_inputs
    return torch.from_numpy(np.stack([point1,point2]))


def extract_opposite_attractors_from_model(model,dataset,input_range=(0.9,0.92)):
    model.eval()
    inputs,_ = dataset()
    inputs = torch.from_numpy(inputs).type(torch.float32)

    # Run the simulation
    _,hidden = model(inputs,deterministic=False)

    attractors = get_points_on_opposite_attractors(inputs,hidden,return_separately=False,input_range=input_range)
    return attractors


def extract_submodel(model, indices):
    """
    Extract a sub-model from an RNNModel by selecting specific neurons.
    
    Args:
        model: RNNModel instance
        indices: List or tensor of indices to extract
        
    Returns:
        submodel: New RNNModel with only the selected neurons
    """
    # Get the original parameters
    N = len(indices)  # New number of neurons
    input_size = model.input_size
    
    # Extract the relevant parts of M
    # M has shape (N+input_size, N+input_size)
    # We need to select both the rows and columns corresponding to the selected neurons
    # and keep the input dimensions intact
    M_sub = torch.zeros(N + input_size, N + input_size)
    
    # Copy the neuron-to-neuron connections
    M_sub[:N, :N] = model.M[indices, :][:, indices]
    
    # Copy the input-to-neuron connections
    M_sub[:N, N:] = model.M[indices, -input_size:]
    
    # Copy the neuron-to-input connections
    M_sub[N:, :N] = model.M[-input_size:, indices]
    
    # Copy the input-to-input connections
    M_sub[N:, N:] = model.M[-input_size:, -input_size:]
    
    # Extract the relevant parts of h
    h_sub = torch.zeros(N + input_size)
    h_sub[:N] = model.h[indices]
    h_sub[N:] = model.h[-input_size:]
    
    # Extract the initial condition
    x_init_sub = model.x_init[indices]
    
    # Create new model with the same parameters but new N, M, h, and x_init
    submodel = RNNModel(
        dt=model.dt,
        N=N,
        input_size=input_size,
        ramp_train=model.ramp_train,
        tau=model.tau,
        f0=model.f0,
        beta0=model.beta0,
        theta0=model.theta0,
        M=M_sub,
        eff_dt=model.eff_dt,
        h=h_sub,
        sigma_noise=model.sigma_noise,
        init_sigma=model.init_sigma,
        x_init=x_init_sub
    )
    
    return submodel

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

    from finkelstein_fontolan_task import initialize_task
    dataset = initialize_task(N_trials_cd=30)
    inputs,_ = dataset()
    inputs = torch.from_numpy(inputs).type(torch.float32)

    # Run the simulation
    _,hidden = model(inputs,deterministic=True)


    point1, point2, average_input = get_points_on_opposite_attractors(inputs, hidden, input_range=(0.9, 0.92), return_separately=True)
    point1_additional, point2_additional, average_input_additional = get_points_on_opposite_attractors(inputs, hidden, input_range=(0.7, 0.72), return_separately=True)

    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    # Detach and convert hidden to numpy
    hidden_np = hidden.detach().cpu().numpy()

    # Fit PCA on the hidden states
    pca = PCA(n_components=2)
    pca.fit(hidden_np.reshape(-1, hidden_np.shape[-1]))
    hidden_pca = pca.transform(hidden_np.reshape(-1, hidden_np.shape[-1]))
    hidden_pca = hidden_pca.reshape(hidden_np.shape[0], hidden_np.shape[1], -1)  

    # Transform point1 and point2
    point1_transformed = pca.transform(point1.reshape(-1, point1.shape[-1]))
    point2_transformed = pca.transform(point2.reshape(-1, point2.shape[-1]))
    point1_additional_transformed = pca.transform(point1_additional.reshape(-1, point1_additional.shape[-1]))
    point2_additional_transformed = pca.transform(point2_additional.reshape(-1, point2_additional.shape[-1]))

    # Plot PC1 and PC2 of hidden as well as the two sets of points
    # plt.figure(figsize=(8, 6))
    # plt.plot(hidden_pca[:,:, 0], hidden_pca[:,:, 1], alpha=0.5)
    # plt.scatter(point1_transformed[:, 0], point1_transformed[:, 1], color='red', label='Point 1 (0.9, 0.92)', marker='x', s=100)
    # plt.scatter(point2_transformed[:, 0], point2_transformed[:, 1], color='blue', label='Point 2 (0.9, 0.92)', marker='x', s=100)
    # plt.scatter(point1_additional_transformed[:, 0], point1_additional_transformed[:, 1], color='green', label='Point 1 (0.7, 0.72)', marker='x', s=100)
    # plt.scatter(point2_additional_transformed[:, 0], point2_additional_transformed[:, 1], color='orange', label='Point 2 (0.7, 0.72)', marker='x', s=100)
    # plt.xlabel('PC1')
    # plt.ylabel('PC2')
    # plt.title('PCA of Hidden States and Points')
    # plt.legend()
    # plt.grid()
    # plt.show()


    # Extract a submodel from the original model
    # submodel = extract_submodel(model, indices=[0, 1, 2])  # Example: extract first 3 neurons
    #
    # # Test the submodel with some input
    # test_input = torch.randn(1, 1, model.input_size)  # batch_size=1, seq_len=1
    # output,_ = submodel(test_input)
    # print("Submodel output shape:", output.shape)
    #
    # # Verify the submodel has the correct number of neurons
    # print("Original model neurons:", model.N)
    # print("Submodel neurons:", submodel.N)
    # assert submodel.N == 3  # Should match the number of indices we extracted

    # submodel = extract_submodel(
    #     model,
    #     # indices=np.random.permutation(np.arange(model.N))
    #     # indices=np.random.permutation(np.arange(model.N-5))
    #     # indices = np.random.choice(668,size=(668,),replace=True)
    #     indices=sort_idx[10:]
    # )


    # # Run the simulation
    # _,hidden = submodel(inputs,deterministic=True)
    #
    # # Detach and convert hidden to numpy
    # hidden_np = hidden.detach().cpu().numpy()
    #
    # # Fit PCA on the hidden states
    # pca = PCA(n_components=2)
    # pca.fit(hidden_np.reshape(-1, hidden_np.shape[-1]))
    # hidden_pca = pca.transform(hidden_np.reshape(-1, hidden_np.shape[-1]))
    # hidden_pca = hidden_pca.reshape(hidden_np.shape[0], hidden_np.shape[1], -1)

    #
    # # Plot PC1 and PC2 of hidden as well as the two sets of points
    # plt.figure(figsize=(8, 6))
    # plt.plot(hidden_pca[:,:, 0], hidden_pca[:,:, 1], alpha=0.5)
    # plt.scatter(point1_transformed[:, 0], point1_transformed[:, 1], color='red', label='Point 1 (0.9, 0.92)', marker='x', s=100)
    # plt.scatter(point2_transformed[:, 0], point2_transformed[:, 1], color='blue', label='Point 2 (0.9, 0.92)', marker='x', s=100)
    # plt.scatter(point1_additional_transformed[:, 0], point1_additional_transformed[:, 1], color='green', label='Point 1 (0.7, 0.72)', marker='x', s=100)
    # plt.scatter(point2_additional_transformed[:, 0], point2_additional_transformed[:, 1], color='orange', label='Point 2 (0.7, 0.72)', marker='x', s=100)
    # plt.xlabel('PC1')
    # plt.ylabel('PC2')
    # plt.title('PCA of Hidden States and Points')
    # plt.legend()
    # plt.grid()
    # plt.show()

    # # Test that the submodel's weights are properly extracted
    # for i, idx in enumerate([0, 1, 2]):
    #     assert torch.allclose(submodel.M[i], model.M[idx])

    

    # model(r_in_cd[:,0],x_init=x_init)
    # print()
    # import matplotlib.pyplot as plt
    # plt.plot(rp_vec_nd[0,:2,:].T)
    # plt.show()
    #
    # from rnn import get_autonomous_dynamics_from_model
    #
    # dynamics = get_autonomous_dynamics_from_model(
    #     model,rnn_submodule_name=None,kwargs={'deterministic':True,'batch_first':False}
    # )
    # inp = inputs[:1]
    # print(
    #     model(
    #         inp,
    #         # x_init=torch.zeros(1, batch_size, model.N),
    #     ),
    #     dynamics(x_init).shape
    # )
    # new_model = convert_to_perturbable_RNNModel(model)
    # print(new_model.M.shape)
    #
    # assert np.allclose(new_model.M[:model.N, :model.N].detach().numpy(), model.M[:model.N, :model.N].detach().numpy())
    #
    # assert np.allclose(new_model.M[:model.N,model.N+model.input_size:].detach().numpy(),np.eye(model.N))

    # Create a random submodel with N=5 neurons
    random_indices = torch.randperm(model.N)[:5]
    submodel = extract_submodel(model, random_indices)

    # Get the top 5 PCs of the original model's activity
    pca = PCA(n_components=5)
    hidden_reshaped = hidden_np.reshape(-1, hidden_np.shape[-1])
    pca.fit(hidden_reshaped)
    top_5_pcs = pca.components_  # Shape: (5, N)

    # Training loop
    optimizer = torch.optim.Adam(submodel.parameters(), lr=0.1)
    criterion = nn.MSELoss()
    num_epochs = 100

    for epoch in range(num_epochs):
        inputs, _ = dataset()
        inputs = torch.from_numpy(inputs).type(torch.float32)

        optimizer.zero_grad()
        
        # Sample from original model
        with torch.no_grad():
            _, original_hidden = model(inputs, deterministic=True)
        original_hidden_np = original_hidden.detach().cpu().numpy()
        original_hidden_reshaped = original_hidden_np.reshape(-1, original_hidden_np.shape[-1])
        
        # Get target PCs
        target_activity = pca.transform(original_hidden_reshaped).reshape(*original_hidden.shape[0:2], -1)
        target_activity = torch.from_numpy(target_activity).float()
        
        # Get submodel output
        _, submodel_hidden = submodel(inputs, deterministic=True)
        # print(submodel_hidden.shape, target_activity.shape)
        # Calculate loss and update
        loss = criterion(submodel_hidden, target_activity)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Evaluate the trained submodel
    submodel.eval()
    with torch.no_grad():
        _, trained_hidden = model(inputs, deterministic=True)
        trained_hidden_np = trained_hidden.detach().cpu().numpy()
        
        # Compare with target PCs
        trained_pca = PCA(n_components=5)
        trained_pca.fit(trained_hidden_np.reshape(-1, trained_hidden_np.shape[-1]))
        
        # Calculate alignment between original and trained PCs
        alignment = np.abs(trained_pca.components_ @ top_5_pcs.T)
        print("\nAlignment between original and trained PCs:")
        print(alignment)


    # # Run the simulation
    _,hidden = submodel(inputs,deterministic=True)

    # Detach and convert hidden to numpy
    hidden_np = hidden.detach().cpu().numpy()

    # Fit PCA on the hidden states
    pca = PCA(n_components=2)
    pca.fit(hidden_np.reshape(-1, hidden_np.shape[-1]))
    hidden_pca = pca.transform(hidden_np.reshape(-1, hidden_np.shape[-1]))
    hidden_pca = hidden_pca.reshape(hidden_np.shape[0], hidden_np.shape[1], -1)
