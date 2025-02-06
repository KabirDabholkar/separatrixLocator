import torch
import torch.nn as nn
import torch.optim as optim
from functools import partial
import torch.nn.functional as F
from torchdiffeq import odeint
import numpy as np
from PytorchRBFLayer.rbf_layer.rbf_layer import RBFLayer, AnisotropicRBFLayer

import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import torch
from functools import partial


def process_initial_conditions(
        func,
        init_cond_dist,
        initial_conditions,
        input_dim,
        dist_needs_dim,
        batch_size,
        threshold,
        resample_above_threshold
):
    """
    Processes initial conditions for optimization.

    If initial_conditions is None and resample_above_threshold is True, samples candidates from
    `init_cond_dist` until obtaining a full batch of points with func(point) > threshold.
    Otherwise, if initial_conditions is None, samples a full batch normally.
    If initial_conditions is provided, it is used as is (with gradients enabled).

    Args:
        func: Callable, a differentiable scalar-valued function.
        init_cond_dist: A PyTorch distribution for sampling initial conditions.
        initial_conditions: Optional tensor of initial conditions.
        input_dim: Dimension of the input space.
        dist_needs_dim: Boolean indicating whether to add an extra dimension to the sample.
        batch_size: Number of initial points to optimize.
        threshold: Threshold value used to filter points.
        resample_above_threshold: If True, only accept points where func(point) > threshold.

    Returns:
        initial_conditions: A tensor of initial conditions with gradients enabled.
        batch_size: The effective batch size.
    """
    if initial_conditions is None:
        if resample_above_threshold:
            if threshold is None:
                raise ValueError("When resample_above_threshold is True, threshold must be provided.")
            accepted_points = []
            # Continue sampling until we have enough valid points.
            while sum(pt.shape[0] for pt in accepted_points) < batch_size:
                sample_shape = [batch_size] + ([input_dim] if dist_needs_dim else [])
                candidates = init_cond_dist.sample(sample_shape=sample_shape)
                # Evaluate candidates without tracking gradients.
                with torch.no_grad():
                    candidate_losses = func(candidates)
                # Create a boolean mask: only keep candidates above the threshold.
                mask = candidate_losses[...,0] > threshold
                valid = candidates[mask]
                if valid.numel() > 0:
                    accepted_points.append(valid)
            # Concatenate and take only the first batch_size samples.
            accepted_points = torch.cat(accepted_points, dim=0)[:batch_size]
            initial_conditions = accepted_points.requires_grad_()
        else:
            sample_shape = [batch_size] + ([input_dim] if dist_needs_dim else [])
            initial_conditions = init_cond_dist.sample(sample_shape=sample_shape).requires_grad_()
    else:
        # Use provided initial conditions.
        batch_size = initial_conditions.shape[0]
        initial_conditions = initial_conditions.requires_grad_()

    return initial_conditions, batch_size


def runGD(
        func,
        init_cond_dist,
        initial_conditions=None,
        input_dim=1,
        dist_needs_dim=True,
        num_steps=100,
        partial_optim=partial(torch.optim.Adam, lr=1e-2),
        batch_size=64,
        threshold=5e-2,
        lr_scheduler=None,
        resample_above_threshold=False
):
    """
    Optimizes a scalar-valued function using full-batch Adam and records trajectories.

    Args:
        func: Callable, a differentiable scalar-valued function.
        init_cond_dist: A PyTorch distribution for sampling initial conditions.
        initial_conditions: Optional tensor of initial conditions. If None, conditions will be sampled.
        input_dim: Dimension of the input space.
        dist_needs_dim: Boolean indicating whether to add an extra dimension to the sample.
        num_steps: Number of optimization steps.
        partial_optim: Partial function for creating the optimizer.
        batch_size: Number of initial points to optimize.
        threshold: Threshold value. During optimization, points with func(value) below
                   threshold are recorded separately. When resampling is enabled, only
                   initial conditions with func(value) above threshold are used.
        lr_scheduler: Optional learning rate scheduler.
        resample_above_threshold: If True, only initial conditions with func(initial_conditions) > threshold
                                  are used (others are dropped and re-sampled).

    Returns:
        trajectories: A tensor of shape (num_steps, batch_size, input_dim),
                      recording the optimization trajectories.
        below_threshold_points: A tensor containing points that dropped below the threshold.
    """
    # Process the initial conditions using the helper function.
    initial_conditions, batch_size = process_initial_conditions(
        func, init_cond_dist, initial_conditions, input_dim, dist_needs_dim, batch_size, threshold,
        resample_above_threshold
    )

    # Create the optimizer.
    optimizer = partial_optim([initial_conditions])

    # Apply learning rate scheduler if provided.
    scheduler = lr_scheduler(optimizer) if lr_scheduler else None

    # Record trajectories.
    trajectories = torch.zeros((num_steps, batch_size, input_dim), dtype=torch.float32)

    # Track points that go below the threshold.
    below_threshold_mask = torch.zeros(batch_size, dtype=torch.bool)
    below_threshold_points = []

    for step in range(num_steps):
        # Record current positions.
        trajectories[step] = initial_conditions.detach()

        # Zero gradients.
        optimizer.zero_grad()

        # Compute loss (scalar value).
        losses = func(initial_conditions)
        # if threshold is not None:
        #     # Only optimize points with loss above threshold.
        #     losses_to_optimize = losses * (losses > threshold)
        # else:
        #     losses_to_optimize = losses
        losses_to_optimize = losses

        loss = losses_to_optimize.sum()

        # Backward pass.
        loss.backward()

        # Update parameters.
        optimizer.step()

        # Step the learning rate scheduler if provided.
        if scheduler:
            scheduler.step()

        # Check for values going below threshold and store them.
        newly_below_threshold = (losses[...,0] < threshold) & ~below_threshold_mask
        if newly_below_threshold.any():
            indices = newly_below_threshold.nonzero(as_tuple=True)[0]
            below_threshold_points.append(initial_conditions[indices].detach().clone())
            below_threshold_mask[indices] = True

    if below_threshold_points:
        below_threshold_points = torch.cat(below_threshold_points, dim=0)
    else:
        below_threshold_points = torch.empty((0, input_dim))

    return trajectories, below_threshold_points


def log_metrics(logger, metrics, epoch):
    """
    Log metrics to the given logger(s).

    Args:
        logger (None, callable, list of callables): Logger(s) to log metrics. Each logger should have a `.log()` or `.add_scalar()` method.
        metrics (dict): Dictionary of metric names and their values.
        epoch (int): Current epoch number.
    """
    if logger is None:
        return

    if not isinstance(logger, list):
        logger = [logger]

    for log in logger:
        if hasattr(log, "add_scalar"):  # TensorBoard-like logger
            for key, value in metrics.items():
                log.add_scalar(key, value, epoch)
        elif hasattr(log, "log"):  # WandB-like logger
            log.log({**metrics, "epoch": epoch})

class DecayModule:
    def __init__(self, initial_decay=1.0, decay_rate=0.95, start_epoch=1000):
        """
        Optional module to compute a decaying weight factor with a delay.

        Args:
            initial_decay (float): Initial weight for the decayed term.
            decay_rate (float): Exponential decay rate per epoch.
            start_epoch (int): The epoch after which decay begins.
        """
        self.initial_decay = initial_decay
        self.decay_rate = decay_rate
        self.start_epoch = start_epoch

    def get_decay_factor(self, epoch):
        """
        Calculate the decay factor for the given epoch.

        Args:
            epoch (int): Current epoch number.

        Returns:
            float: Decay factor for the epoch.
        """
        if epoch < self.start_epoch:
            # No decay applied before the start epoch
            return 1.0
        # Apply decay after the start epoch
        adjusted_epoch = epoch - self.start_epoch
        return self.initial_decay * (self.decay_rate ** adjusted_epoch)


##### Neural ODE ###

class ODEBlock(nn.Module):

    def __init__(self, odefunc, odefunc_dim,input_dim=1,output_dim = 1):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()
        self.output_dim = output_dim
        self.input_layer = nn.Linear(input_dim,odefunc_dim)
        self.readout = nn.Linear(odefunc_dim,output_dim)
        self.atol = None
        self.rtol = None

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        x = self.input_layer(x)
        out = odeint(lambda t,y: self.odefunc(y), x, self.integration_time)#, rtol=self.rtol, atol=self.atol)
        return self.readout(out[1])

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value

class ScaledLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, scale=0.1):
        super().__init__(in_features, out_features, bias)
        self.scale = scale
        self._scale_weights()

    def _scale_weights(self):
        self.weight.data *= self.scale  # Apply scaling during initialization

    def forward(self, x):
        return nn.functional.linear(x, self.weight * self.scale, self.bias)

# Define the neural network as phi(x) using nn.Sequential
def create_phi_network(input_dim=1, hidden_dim=200, output_dim=1, nonlin = nn.Tanh):
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nonlin(),
        nn.Linear(hidden_dim, hidden_dim),
        nonlin(),
        nn.Linear(hidden_dim, hidden_dim),
        nonlin(),
        # nn.Linear(hidden_dim, hidden_dim),
        # nonlin(),
        # nn.Dropout(p=0.02),
        nn.Linear(hidden_dim, output_dim)

    )
    return model

class AttentionSelectorDNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()

        # Fully connected layers
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim

        # Last hidden layer produces `output_dim` features
        self.feature_extractor = nn.Sequential(*layers)
        self.last_layer = nn.Linear(prev_dim, output_dim)

        # Attention mechanism (produces weights for output selection)
        self.attention = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        # Extract deep features
        features = self.feature_extractor(x)  # Shape: (batch_size, hidden_dim)
        outputs = self.last_layer(features)   # Shape: (batch_size, output_dim)

        # Compute attention weights
        attention_scores = self.attention(outputs)  # Shape: (batch_size, output_dim)
        attention_weights = F.softmax(attention_scores, dim=1)  # Normalize

        # Weighted sum (soft attention) or select max (hard attention)
        selected_output = (attention_weights * outputs).sum(dim=1, keepdim=True)

        return selected_output, attention_weights


class ParallelModels(nn.Module):
    def __init__(self, base_model, num_models, prod_output = True, select_max = False):
        """
        base_model: An instance of torch.nn.Module representing the base architecture.
        num_models: Number of independent models to train in parallel.
        """
        super().__init__()
        self.num_models = num_models
        self.models = nn.ModuleList([base_model() for _ in range(num_models)])
        # self.model = base_model() #()
        self.prod_output = prod_output
        self.select_max = select_max

    def forward(self, x):
        """
        x: Input tensor of shape (num_models, batch_size, input_dim)
        Returns stacked output of shape (num_models, batch_size, output_dim)
        """
        outputs = [model(x) for i, model in enumerate(self.models)]
        if self.prod_output:
            stack_outputs = torch.stack(outputs, dim=-1)
            mean_outputs = torch.mean(torch.abs(stack_outputs),dim=-2)
            ids = torch.argmax(mean_outputs, dim=-1)
            # print(ids)
            select_outputs = torch.zeros_like(mean_outputs)
            for i in range(select_outputs.shape[0]):
                select_outputs[i] = stack_outputs[i, ..., ids[i]]
            outputs = select_outputs
        else:
            outputs = torch.concatenate(outputs, dim=-1)
        # if self.prod_output:
        #     outputs = torch.prod(torch.abs(outputs),axis=-1,keepdim=True)
        # return self.model(x) #
        # return self.models[0](x)
        return outputs

class ExpOutput(nn.Module):
    def __init__(self, base_model):
        super(ExpOutput, self).__init__()
        self.base_model = base_model

    def forward(self, x):
        # Forward pass through the base model
        output = self.base_model(x)
        # Apply torch.abs and torch.exp
        output = torch.exp(torch.abs(output))-1
        return output

class LogOutput(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        # Forward pass through the base model
        output = self.base_model(x)
        # Apply torch.abs and torch.exp
        # output = torch.exp(torch.abs(output))-1
        output = - torch.log(torch.abs(output))
        return output

class AttentionNN(nn.Module):
    def __init__(self, input_dim, output_dim, temperature=1.0):
        """
        Initialize the neural network with attention.
        Args:
            input_dim (int): Dimensionality of the input space (R^N).
            output_dim (int): Dimensionality of the output space (R^M).
            temperature (float): Temperature for softmax (controls smoothness of attention weights).
        """
        super(AttentionNN, self).__init__()
        self.temperature = temperature
        # Define layers
        self.fc1 = nn.Linear(input_dim, 128)  # Feature extraction
        self.fc2 = nn.Linear(128, 64)
        self.query = nn.Linear(64, 64)  # Query vector
        self.key = nn.Linear(64, 64)  # Key vector
        self.value = nn.Linear(64, output_dim)  # Value vector (output space)

    def forward(self, x):
        """
        Forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        # Feature extraction
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Attention mechanism
        query = self.query(x)  # Compute queries
        key = self.key(x)  # Compute keys
        value = self.value(x)  # Compute values

        # Attention scores: scaled dot-product attention
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (key.size(-1) ** 0.5)
        attention_weights = F.softmax(attention_scores / self.temperature, dim=-1)

        # Compute the context vector as the weighted sum of values
        context = torch.matmul(attention_weights, value)

        return context

class OneHotOutputNN(nn.Module):
    def __init__(self, input_dim, output_dim, temperature=1.0):
        """
        Initialize the neural network.
        Args:
            input_dim (int): Dimensionality of the input space (R^N).
            output_dim (int): Dimensionality of the output space (R^M).
            temperature (float): Temperature for softmax (controls smoothness of output).
        """
        super(OneHotOutputNN, self).__init__()
        self.temperature = temperature
        # Define the network layers
        self.fc1 = nn.Linear(input_dim, 128)  # First fully connected layer
        self.fc2 = nn.Linear(128, 64)        # Second fully connected layer
        self.fc3 = nn.Linear(64, output_dim) # Final output layer

    def forward(self, x):
        """
        Forward pass of the network.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        # Hidden layers with non-linear activation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Output layer (scores)
        scores = self.fc3(x)
        # Apply softmax with temperature
        probabilities = F.softmax(scores / self.temperature, dim=1)
        # Compute one-hot-like output
        output = probabilities * scores  # Smooth approximation of argmax
        return output


class AttentionOneHotNN(nn.Module):
    def __init__(self, input_dim, output_dim, temperature=1.0):
        """
        Initialize the neural network with attention.
        Args:
            input_dim (int): Dimensionality of the input space (R^N).
            output_dim (int): Dimensionality of the output space (R^M).
            temperature (float): Temperature for softmax (controls smoothness of attention weights).
        """
        super(AttentionOneHotNN, self).__init__()
        self.temperature = temperature
        # Define layers
        self.fc1 = nn.Linear(input_dim, 128)  # Feature extraction
        self.fc2 = nn.Linear(128, 64)
        self.query = nn.Linear(64, 64)  # Query vector
        self.key = nn.Linear(64, 64)  # Key vector
        self.value = nn.Linear(64, output_dim)  # Value vector (output space)

    def forward(self, x):
        """
        Forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        # Feature extraction
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Attention mechanism
        query = self.query(x)  # Compute queries
        key = self.key(x)  # Compute keys
        value = self.value(x)  # Compute values

        # Attention scores: scaled dot-product attention
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (key.size(-1) ** 0.5)
        attention_weights = F.softmax(attention_scores / self.temperature, dim=-1)

        # Compute the context vector as the weighted sum of values
        context = torch.matmul(attention_weights, value)

        # Output layer: apply softmax to context vector to get a one-hot-like output
        output = F.softmax(context, dim=-1) * value
        return output

def compute_loss(model, x, F, epoch, decay_factor=1.0):
    """
    Compute the regularized loss with optional decay.

    Args:
        model (torch.nn.Module): The model being trained.
        x (torch.Tensor): Input batch.
        F (callable): Dynamical system function.
        epoch (int): Current epoch number.
        decay_factor (float): Weight for the decay term.

    Returns:
        torch.Tensor: Total loss.
    """
    phi_x = model(x)

    # Compute phi'(x) using autograd
    x.requires_grad_(True)
    phi_x_prime = torch.autograd.grad(
        outputs=model(x),
        inputs=x,
        grad_outputs=torch.ones_like(model(x)),
        create_graph=True
    )[0]

    # Main loss term
    dot_prod = (phi_x_prime * F(x)).sum(axis=-1, keepdim=True)
    main_loss = torch.mean((dot_prod - phi_x) ** 2)

    # Variance penalty
    phi_mean = torch.mean(phi_x)
    phi_variance = torch.mean((phi_x - phi_mean) ** 2)
    variance_penalty = (phi_variance - 1) ** 2

    # Decay term (-l0 with weight)
    l0 = torch.abs(phi_x).mean()

    # Combine losses
    total_loss = main_loss + variance_penalty - decay_factor * l0

    return total_loss

def variance_normaliser(x,y):
    return torch.mean((x-y)**2,axis=0)/torch.mean(y**2,axis=0)

def shuffle_normaliser(x,y,axis=0,return_terms=False):
    permutation = np.random.permutation(x.shape[0])
    numerator = torch.mean((x - y) ** 2, axis=axis)
    denominator = torch.mean((x - y[permutation]) ** 2, axis=axis)
    ratio = numerator / denominator
    if return_terms:
        return ratio, numerator, denominator
    return ratio


def eval_loss(model,F,dist,dist_requires_dim=True,batch_size=64,dynamics_dim=1,eigenvalue=1,drop_values_outside_range = None, normaliser=shuffle_normaliser):
    sample_shape = [batch_size]
    if dist_requires_dim:
        sample_shape += [dynamics_dim]
    x_batch = dist.sample(sample_shape=sample_shape)

    # Enable gradient computation for x_batch
    x_batch.requires_grad_(True)

    # Forward pass and compute phi(x)
    phi_x = model(x_batch)
    points_to_use = torch.ones_like(x_batch)[...,0:1]
    if drop_values_outside_range is not None:
        points_to_use = (phi_x>drop_values_outside_range[0]) & (phi_x<drop_values_outside_range[1])

    # Compute phi'(x)
    phi_x_prime = torch.autograd.grad(
        outputs=phi_x,
        inputs=x_batch,
        grad_outputs=torch.ones_like(phi_x),
        create_graph=True
    )[0]

    # Compute F(x_batch)
    F_x = F(x_batch)

    # Main loss term: ||phi'(x) F(x) - phi(x)||^2
    dot_prod = (phi_x_prime * F_x).sum(axis=-1, keepdim=True)
    # main_loss = torch.mean((dot_prod - eigenvalue * phi_x) ** 2 * points_to_use,axis=(0))/torch.mean(points_to_use.to(float))
    # dot_prod = torch.log(torch.abs(dot_prod))
    # phi_x = torch.log(torch.abs(phi_x))

    main_loss = normaliser(dot_prod,eigenvalue * phi_x)

    # Variance penalty: |Var(phi(x)) - 1|^2
    # phi_mean = torch.mean(phi_x)
    # standard_dev = torch.std(phi_x * points_to_use,axis=-2)
    # main_loss /= standard_dev

    return main_loss

# \ell norm
def l_norm(x, p=2):
    return torch.norm(x, p=p, dim=-1)


# Gaussian RBF
def rbf_gaussian(x):
    return (-x.pow(2)).exp()

def rbf_laplacian(x):
    return (-x.pow(2).sqrt()).exp()



def train_model_on_trajectories_sgd(trajectories, model, t_values, batch_size=32, num_epochs=1000, learning_rate=0.01, device='cpu'):
    """
    Train the neural network model using SGD with minibatches to minimize the loss:
    |ln(psi(x(t))) - ln(psi(x(0))) - t|^2

    Args:
        trajectories (torch.Tensor): Input tensor of shape (n_trials, T, d).
        model (nn.Module): PyTorch neural network mapping from R^d to R.
        batch_size (int): Minibatch size for SGD.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for SGD optimizer.
        device (str): Device to train on ('cpu' or 'cuda').
    """

    # Move data to the specified device
    trajectories = trajectories.to(device)
    model.to(device)

    # Prepare dataset and data loader for minibatch training
    dataset = TensorDataset(trajectories)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        # model.train()
        epoch_loss = 0.0

        for batch in data_loader:
            batch_trajectories = batch[0]  # Extract trajectories from dataset
            batch_size_here = batch_trajectories.shape[0]

            # Extract initial points x(0) and all time steps x(t)
            x_0 = batch_trajectories[:, 0:1, :]  # shape (batch_size, d)
            x_t = batch_trajectories  # shape (batch_size, T, d)
            # t_values = torch.arange(batch_trajectories.shape[1], device=device).float().detach()  # shape (T,)

            # Compute f(x) and psi(x)
            f_x_0 = torch.abs(model(x_0.view(-1, x_0.shape[-1])))  # shape (batch_size, 1)
            psi_x_0 = torch.exp(f_x_0) - 1  # shape (batch_size, 1)

            f_x_t = torch.abs(model(x_t.view(-1, x_t.shape[-1])))  # shape (batch_size * T, 1)
            psi_x_t = torch.exp(f_x_t) - 1  # shape (batch_size * T, 1)

            # Reshape to (batch_size, T, 1)
            psi_x_0 = psi_x_0.view(batch_size_here, -1, 1)
            psi_x_t = psi_x_t.view(batch_size_here, -1, 1)

            # Compute logarithms
            log_psi_x_0 = torch.log(psi_x_0)  # shape (batch_size, 1, 1)
            log_psi_x_t = torch.log(psi_x_t)  # shape (batch_size, T, 1)

            # Compute the loss |ln(psi(x(t))) - ln(psi(x(0))) - t|^2
            loss = torch.mean((log_psi_x_t - log_psi_x_0 - t_values.unsqueeze(0).unsqueeze(-1)) ** 2)
            loss /= t_values[-1]**2

            # Backpropagation
            optimizer.zero_grad(set_to_none=True)
            loss.backward(retain_graph=True)
            optimizer.step()

            epoch_loss += loss.item()

        # Print progress
        if epoch % 1 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {epoch_loss / len(data_loader):.6f}")

    print("Training completed.")
    return model

def evaluate_param_specific_hyperparams(model,param_specific_hyperparams):
    params = dict(model.named_parameters())
    # print(params)
    param_specific_hyperparams_complete = []
    for param_list in param_specific_hyperparams:
        new_param_list = dict(param_list)
        new_param_list['params'] = [params[p] for p in param_list['params']]
        param_specific_hyperparams_complete.append(new_param_list)
    # for param_list in param_specific_hyperparams:
    #     for p in param_list['params']:
    #         print(type(p))
    #     print(param_list['weight_decay'])
    return param_specific_hyperparams_complete

# def train_with_logger(
#     model, F, dist, dist_requires_dim=True, num_epochs=1000, learning_rate=1e-3, batch_size=64,
#     dynamics_dim=1, decay_module=None, logger=None,
#     eigenvalue = 1, print_every_num_epochs=10, device='cpu',param_specific_hyperparams=[],
# ):
#     """
#     Train the model with optional decay and logging.
#
#     Args:
#         model (torch.nn.Module): The model being trained.
#         F (callable): Dynamical system function.
#         dist (torch.distributions.Distribution): Distribution for sampling inputs.
#         num_epochs (int): Number of epochs for training.
#         learning_rate (float): Learning rate for the optimizer.
#         batch_size (int): Batch size for training.
#         dynamics_dim (int): Dimensionality of the dynamical system.
#         decay_module (DecayModule, optional): Module for handling decay. Defaults to None.
#         logger (None, callable, list of callables): Logger(s) to log metrics.
#     """
#     if len(param_specific_hyperparams) == 0:
#         param_specific_hyperparams = model.parameters()
#     else:
#         param_specific_hyperparams = evaluate_param_specific_hyperparams(model, param_specific_hyperparams)
#
#     optimizer = torch.optim.Adam(
#         param_specific_hyperparams,
#         lr=learning_rate
#     )
#     if dist_requires_dim:
#         sample_shape = (batch_size, dynamics_dim)
#     else:
#         sample_shape = (batch_size,)
#     for epoch in range(num_epochs):
#         # Generate batch of samples
#         x_batch = dist.sample(sample_shape=sample_shape).to(device)
#         # print(x_batch.shape)
#
#         # Enable gradient computation for x_batch
#         x_batch.requires_grad_(True)
#
#         # Forward pass and compute phi(x)
#         phi_x = model(x_batch)
#         # if torch.isnan(phi_x).any():
#         #     raise ValueError("NaN in phi_x")
#         # print(model.get_kernels_centers)
#         # print(model.get_weights)
#         output_dim = phi_x.shape[-1]
#         # phi_x_prime = torch.autograd.grad(
#         #     outputs=phi_x,
#         #     inputs=x_batch,
#         #     grad_outputs=torch.ones_like(phi_x),
#         #     create_graph=True  # True
#         # )[0]  # .detach() #this wasnt there before
#         #
#         # phi_x_prime0 = torch.autograd.grad(
#         #     outputs=phi_x[...,0],
#         #     inputs=x_batch,
#         #     grad_outputs=torch.ones_like(phi_x[...,0]),
#         #     create_graph=True  # True
#         # )[0]
#
#         phi_x_prime = torch.autograd.grad(
#             outputs=phi_x.sum(axis=-1),
#             inputs=x_batch,
#             grad_outputs=torch.ones_like(phi_x.sum(axis=-1)),
#             create_graph=True  # True
#         )[0]
#
#
#         # # Compute phi'(x)
#         # jacobian = torch.zeros(batch_size, dynamics_dim, output_dim)
#         # for i in range(output_dim):
#         #     phi_x_prime = torch.autograd.grad(
#         #         outputs=phi_x[...,i],
#         #         inputs=x_batch,
#         #         grad_outputs=torch.ones_like(phi_x[...,i]),
#         #         create_graph=True # True
#         #     )[0] #.detach() #this wasnt there before
#         #     jacobian[...,i] = phi_x_prime
#
#         # Compute F(x_batch)
#         F_x = F(x_batch.to('cpu')).to(device)
#
#         # Main loss term: ||phi'(x) F(x) - phi(x)||^2
#         dot_prod = (phi_x_prime * F_x).sum(axis=-1, keepdim=True)  # .sum(axis=-1, keepdim=True)
#         # dot_prod = (phi_x_prime * F_x[...,None]).sum(axis=-2, keepdim=True) #.sum(axis=-1, keepdim=True)
#         # dot_prod = (jacobian * F_x[..., None]).sum(axis=-2)  # .sum(axis=-1, keepdim=True)
#         # print(phi_x.shape, dot_prod.shape, phi_x_prime.shape, F_x.shape)
#         pde_diff = dot_prod - eigenvalue * phi_x
#         perm_ids = np.random.permutation(phi_x.shape[0])
#         pde_diff_shufffle = dot_prod[perm_ids] - eigenvalue * phi_x
#         main_loss = torch.mean(torch.abs(pde_diff) ** 2)
#         # main_loss = torch.mean(torch.log(torch.abs(pde_diff)+1))
#         rbf = 1
#         # rbf = torch.exp(
#         #     -1*torch.mean((phi_x-phi_x[perm_ids]) ** 2,axis=-1,keepdims=True)
#         # )
#         shuffle_loss = torch.mean(torch.abs(pde_diff_shufffle) ** 2 * rbf)
#         # shuffle_loss = torch.mean(torch.log(torch.abs(pde_diff_shufffle) + 1 ))
#
#         # Variance penalty: |Var(phi(x)) - 1|^2
#         phi_mean = torch.mean(phi_x)
#         phi_deviations = phi_x - phi_mean
#         variance_penalty = torch.mean(phi_deviations ** 2)
#         variance_penalty_term = (variance_penalty - 1) ** 2
#
#         # Decay term: -l0
#         l0 = torch.abs(phi_x).mean()
#
#         # Compute decay factor if decay_module is provided
#         decay_factor = decay_module.get_decay_factor(epoch) if decay_module else 1.0
#
#         # Total loss
#         # total_loss = main_loss + variance_penalty_term # - decay_factor * l0
#         # print(pde_diff.shape)
#
#         # normalised_loss = main_loss / variance_penalty * output_dim
#         normalised_loss = main_loss / shuffle_loss
#         total_loss = normalised_loss  # + variance_penalty/l0**2
#
#         max_id = torch.argmax((pde_diff**2).mean((-1,-2)))
#         normalised_max_loss = torch.mean(pde_diff[max_id] ** 2) / shuffle_loss
#         # total_loss += 1e-3 * normalised_max_loss
#
#         # Log metrics
#         metrics = {
#             "Loss/Total": total_loss.item(),
#             "Loss/Main": main_loss.item(),
#             "Loss/VariancePenalty": variance_penalty_term.item(),
#             "Loss/DecayTerm": (-decay_factor * l0).item(),
#         }
#         log_metrics(logger, metrics, epoch)
#
#         # Backpropagation and optimization
#         optimizer.zero_grad()
#         total_loss.backward()
#         param_norm = sum([torch.linalg.norm(p.grad) for p in model.parameters()]).item()
#         # Iterate over all parameters in the model
#         for param in model.parameters():
#             if param.grad is not None:
#                 # Replace NaN values in the gradients with 0
#                 param.grad.data[torch.isnan(param.grad.data)] = 0
#         optimizer.step()
#
#         # Logging to console every 100 epochs
#         if epoch % print_every_num_epochs == 0:
#             print(f"Epoch {epoch}, Loss: {total_loss.item()}, Normalised loss: {normalised_loss}, Normalised Max loss: {normalised_max_loss}, l0: {l0}, param norm: {param_norm}, len(model.parameters()):{len(list(model.parameters()))}")

def mutual_information_loss(psi, eps=1e-8):
    """
    psi: tensor of shape (batch_size, num_classes) with softmax outputs.
    Returns the mutual information based loss:
      L_MI = E[H(psi(x))] - H(E[psi(x)])
    """
    # Conditional entropy per sample: H(psi(x))
    cond_entropy = -torch.sum(psi * torch.log(psi + eps), dim=1).mean()

    # Marginal distribution over classes: average over batch
    p_y = psi.mean(dim=0)
    marg_entropy = -torch.sum(p_y * torch.log(p_y + eps))

    return cond_entropy - marg_entropy

def train_with_logger(
        model, F, dist, dist_requires_dim=True, num_epochs=1000, learning_rate=1e-3, batch_size=64,
        dynamics_dim=1, decay_module=None, logger=None, lr_scheduler=None,
        eigenvalue=1, print_every_num_epochs=10, device='cpu', param_specific_hyperparams=[],
        normaliser = partial(shuffle_normaliser,axis=None,return_terms=True)
    ):
    """
    Train the model with optional decay, logging, and learning rate scheduling.

    Args:
        model (torch.nn.Module): The model being trained.
        F (callable): Dynamical system function.
        dist (torch.distributions.Distribution): Distribution for sampling inputs.
        num_epochs (int): Number of epochs for training.
        learning_rate (float): Learning rate for the optimizer.
        batch_size (int): Batch size for training.
        dynamics_dim (int): Dimensionality of the dynamical system.
        decay_module (DecayModule, optional): Module for handling decay. Defaults to None.
        logger (None, callable, list of callables): Logger(s) to log metrics.
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler.
        eigenvalue (float): Eigenvalue used in the PDE loss term.
        print_every_num_epochs (int): Print log every N epochs.
        device (str): Device to perform training on.
        param_specific_hyperparams (list): List specifying parameter-specific hyperparameters.
    """
    # Evaluate parameter-specific hyperparameters if provided
    if len(param_specific_hyperparams) == 0:
        param_specific_hyperparams = model.parameters()
    else:
        param_specific_hyperparams = evaluate_param_specific_hyperparams(model, param_specific_hyperparams)

    optimizer = torch.optim.Adam(
        param_specific_hyperparams,
        lr=learning_rate
    )
    if lr_scheduler is not None:
        lr_scheduler = lr_scheduler(optimizer)

    # Determine the shape for sampling
    if dist_requires_dim:
        sample_shape = (batch_size, dynamics_dim)
    else:
        sample_shape = (batch_size,)

    for epoch in range(num_epochs):
        # Generate a batch of samples
        x_batch = dist.sample(sample_shape=sample_shape).to(device)
        # Enable gradient computation for x_batch
        x_batch.requires_grad_(True)

        # Forward pass: compute phi(x)
        phi_x = model(x_batch)
        output_dim = phi_x.shape[-1]

        # Compute the gradient of the sum of phi(x) with respect to x_batch
        phi_x_prime = torch.autograd.grad(
            outputs=phi_x.sum(axis=-1),
            inputs=x_batch,
            grad_outputs=torch.ones_like(phi_x.sum(axis=-1)),
            create_graph=True
        )[0]

        # Compute F(x_batch)
        F_x = F(x_batch.to('cpu')).to(device)

        # Main loss term: ||phi'(x) F(x) - eigenvalue * phi(x)||^2
        dot_prod = (phi_x_prime * F_x).sum(axis=-1, keepdim=True)
        # dot_prod = torch.log(torch.abs(dot_prod))
        # phi_x = torch.log(torch.abs(phi_x))
        # pde_diff = dot_prod - eigenvalue * phi_x

        # Shuffle the batch to compute the shuffle loss term
        # perm_ids = np.random.permutation(phi_x.shape[0])
        # pde_diff_shufffle = dot_prod[perm_ids] - eigenvalue * phi_x
        # main_loss = torch.mean(torch.abs(pde_diff) ** 2)
        # rbf = 1
        # shuffle_loss = torch.mean(torch.abs(pde_diff_shufffle) ** 2 * rbf)

        # Variance penalty: |Var(phi(x)) - 1|^2
        phi_mean = torch.mean(phi_x)
        phi_deviations = phi_x - phi_mean
        variance_penalty = torch.mean(phi_deviations ** 2)
        variance_penalty_term = (variance_penalty - 1) ** 2

        # Decay term: -l0 (where l0 is the mean absolute value of phi(x))
        l0 = torch.abs(phi_x).mean()

        # Compute decay factor if decay_module is provided
        decay_factor = decay_module.get_decay_factor(epoch) if decay_module else 1.0

        # Total loss: here we use a normalized loss (main_loss divided by shuffle_loss)
        if normaliser is None:
            normaliser = lambda x, y: (
                torch.mean((x - y) ** 2),
                torch.zeros_like(torch.mean((x - y) ** 2)),
                torch.zeros_like(torch.mean((x - y) ** 2))
            )
        normalised_loss, main_loss, shuffle_loss = normaliser(dot_prod, eigenvalue*phi_x) #main_loss / shuffle_loss
        # normalised_loss = main_loss / variance_penalty
        total_loss = normalised_loss
        # total_loss += mutual_information_loss(phi_x)



        # Optionally, add additional loss terms such as a max-loss penalty if desired
        # max_id = torch.argmax((pde_diff ** 2).mean((-1, -2)))
        # normalised_max_loss = torch.mean(pde_diff[max_id] ** 2) / shuffle_loss

        # total_loss += 1e-3 * normalised_max_loss

        # Log metrics
        metrics = {
            "Loss/Total": total_loss.item(),
            "Loss/Main": main_loss.item(),
            "Loss/VariancePenalty": variance_penalty_term.item(),
            "Loss/DecayTerm": (-decay_factor * l0).item(),
            "Learning Rate": optimizer.param_groups[0]['lr'],
        }
        log_metrics(logger, metrics, epoch)

        # Backpropagation and optimization step
        optimizer.zero_grad()
        total_loss.backward()
        param_norm = sum([torch.linalg.norm(p.grad) for p in model.parameters()]).item()
        # Replace NaN gradients with 0 to maintain stability
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data[torch.isnan(param.grad.data)] = 0
        optimizer.step()

        # Step the learning rate scheduler if provided
        if lr_scheduler is not None:
            lr_scheduler.step()

        # Logging to console every 'print_every_num_epochs' epochs
        if epoch % print_every_num_epochs == 0:
            print(
                f"Epoch {epoch}, Loss: {total_loss.item()}, Normalised loss: {normalised_loss}, "
                # f"Normalised Max loss: {normalised_max_loss}, l0: {l0}, "
                f"param norm: {param_norm}, ",
                f"Learning Rate: {optimizer.param_groups[0]['lr']}, "
                f"len(model.parameters()): {len(list(model.parameters()))}"
            )


def train_on_teacher(
        model, F, dist, dist_requires_dim=True, num_epochs=1000, learning_rate=1e-3, batch_size=64,
        dynamics_dim=1, decay_module=None, logger=None, print_every_num_epochs=10,
):
    """
    Train the model to mimic a teacher function F on points sampled from dist.

    Args:
        model (torch.nn.Module): The model to be trained.
        F (callable): Teacher function that maps inputs to target outputs.
        dist (torch.distributions.Distribution): Distribution for sampling input points.
        dist_requires_dim (bool): If True, sample shape is (batch_size, dynamics_dim); else (batch_size,).
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        batch_size (int): Batch size for training.
        dynamics_dim (int): Dimensionality of the input.
        decay_module (optional): Module for handling decay (if any).
        logger (None, callable, or list of callables): Logger(s) to log metrics.
        print_every_num_epochs (int): Frequency (in epochs) of printing progress.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if dist_requires_dim:
        sample_shape = (batch_size, dynamics_dim)
    else:
        sample_shape = (batch_size,)

    for epoch in range(num_epochs):
        # Sample a batch of inputs from the distribution.
        x_batch = dist.sample(sample_shape=sample_shape)

        # Get the teacher's output and the model's prediction.
        y_teacher = F(x_batch)
        y_pred = model(x_batch)

        # Compute Mean Squared Error loss between the model and teacher outputs.
        loss = torch.mean((y_pred - y_teacher) ** 2)

        # Optionally adjust the loss using a decay factor.
        if decay_module is not None:
            decay_factor = decay_module.get_decay_factor(epoch)
            loss = loss * decay_factor

        # Log metrics if a logger is provided.
        metrics = {
            "Loss/Total": loss.item(),
        }
        log_metrics(logger, metrics, epoch)

        # Backpropagation and optimization.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print progress at the specified frequency.
        if epoch % print_every_num_epochs == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")


def train(model, F, dist, num_epochs=1000, learning_rate=1e-3, batch_size=64, dynamics_dim=1, decay_module=None):
    """
    Train the model with an optional decay module for the loss.

    Args:
        model (torch.nn.Module): The model being trained.
        F (callable): Dynamical system function.
        dist (torch.distributions.Distribution): Distribution for sampling inputs.
        num_epochs (int): Number of epochs for training.
        learning_rate (float): Learning rate for the optimizer.
        batch_size (int): Batch size for training.
        dynamics_dim (int): Dimensionality of the dynamical system.
        decay_module (DecayModule, optional): Module for handling decay. Defaults to None.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        # Generate batch of samples
        x_batch = dist.sample(sample_shape=(batch_size, dynamics_dim))

        # Determine decay factor
        if decay_module is not None:
            decay_factor = decay_module.get_decay_factor(epoch)
        else:
            decay_factor = 1.0  # No decay applied

        # Compute loss
        loss = compute_loss(model, x_batch, F, epoch, decay_factor)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging the loss every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}, Decay Factor: {decay_factor}")


# Define the loss function L_regularised = || phi'(x) F(x) - phi(x) ||^2 + |Var(phi(x)) - 1|^2
# def compute_loss(model, x, F, epoch):
#     # Forward pass to compute phi(x)
#     phi_x = model(x)
#
#     # Compute phi'(x) using autograd
#     x.requires_grad_(True)
#     phi_x_prime = torch.autograd.grad(
#         outputs=model(x),
#         inputs=x,
#         grad_outputs=torch.ones_like(model(x)),
#         create_graph=True
#     )[0]
#
#     eps = 1e-2
#     phi_x_square = phi_x**2
#     # weight = 1 / (phi_x_square + eps)
#     # weight = weight / weight.mean()
#
#     # Compute the main loss term
#     # print(phi_x_prime.shape,F(x).shape)
#     dot_prod = (phi_x_prime * F(x)).sum(axis=-1,keepdim=True)
#
#     main_loss = torch.mean((dot_prod - phi_x) ** 2)
#
#     # Compute the regularization term for variance
#     phi_mean = torch.mean(phi_x)
#     phi_variance = torch.mean((phi_x - phi_mean) ** 2)
#     variance_penalty = (phi_variance - 1) ** 2
#     l0 = torch.abs(phi_x).mean()
#
#     # Total regularized loss
#     total_loss = main_loss + variance_penalty - l0
#
#     # if epoch % 100 == 0:
#     #     plt.scatter(x.detach().numpy(),phi_x.detach().numpy(),label=r'$\phi$')
#     #     # plt.scatter(x.detach().numpy(), weight.detach().numpy(), label=r'weight')
#     #     plt.legend()
#     #     plt.savefig(f'test_outputs/epoch{epoch}.png')
#     #     plt.close()
#
#     return total_loss
#
#
# # Training loop with adjustable sigma
# def train(model, F, dist, num_epochs=1000, learning_rate=1e-3, batch_size=64, dynamics_dim=1):
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#
#     for epoch in range(num_epochs):
#         # Generate batch of samples from N(0, sigma^2)
#         x_batch = dist.sample(sample_shape=(batch_size,dynamics_dim)) #torch.randn(batch_size, 1) * sigma
#
#         # Compute loss
#         loss = compute_loss(model, x_batch, F, epoch)
#
#         # Backpropagation and optimization
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         # Logging the loss every 100 epochs
#         if epoch % 100 == 0:
#             print(f"Epoch {epoch}, Regularized Loss: {loss.item()}")



def partialised_RBF_maker(reset_params,**kwargs):
    model = RBFLayer(
        radial_function=rbf_gaussian,
        # radial_function=rbf_laplacian,
        norm_function=partial(l_norm,p=2),
        **kwargs)
    model.reset(**reset_params)
    return model

def partialised_AnisotropicRBF_maker(reset_params,**kwargs):
    model = AnisotropicRBFLayer(
        radial_function=rbf_gaussian,
        **kwargs)
    # model.reset(**reset_params)
    return model

def main():
    # Define function F(x) = x - x^3
    # def F(x):
    #     return x - x ** 3
    # dist = torch.distributions.Uniform(low=0,high=1)
    # model = create_phi_network()
    # train(
    #     model,
    #     F,
    #     dist,
    #     num_epochs=1000,
    # )
    # torch.distributions.Uniform(low=0, high=1)

    # model = ODEBlock(
    #     odefunc = nn.Linear(5,5),
    #     odefunc_dim = 5,
    #     input_dim = 2,
    #     output_dim = 1,
    # )
    # print(
    #     model(torch.randn((1, 2)))
    # )

    # AttentionSelectorDNN()

    # model = ParallelModels(
    #     base_model=partial(nn.Linear,10,5),
    #     num_models=5,
    #     select_max=True,
    # )
    #

    model = RBFLayer(
        in_features_dim=1,
        out_features_dim=1,
        num_kernels=3,
        radial_function=rbf_gaussian,
        norm_function=l_norm,
    )

    x = torch.randn(size=(20,1))
    x.requires_grad_(True)
    print(
        model(x)
    )

    print(
        model.get_kernels_centers.min(),
        model.get_kernels_centers.max()
    )

if __name__ == '__main__':
    main()