from sklearn.base import BaseEstimator
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.multiprocessing as mp
from functools import partial
from pathlib import Path
from compose import compose
import os
from typing import Iterable
from learn_koopman_eig import train_with_logger, train_with_logger_multiple_dists, train_with_logger_ext_inp, eval_loss, runGD


class KoopmanEigenfunctionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)





class SeparatrixLocator(BaseEstimator):
    def __init__(self, num_models=10, dynamics_dim=1, model_class=KoopmanEigenfunctionModel, lr=1e-3, epochs=100, use_multiprocessing=True, verbose=False, device="cpu"):
        super().__init__()
        # self.func = func
        # self.distribution = distribution
        self.num_models = num_models
        self.lr = lr
        self.epochs = epochs
        self.use_multiprocessing = use_multiprocessing
        self.device = device
        self.verbose = verbose
        self.dynamics_dim = dynamics_dim
        self.model_class = model_class
        self.models = []
        self.scores = None

    def init_models(self):
        self.models = [self.model_class(self.dynamics_dim) for _ in range(self.num_models)]

    def fit(self, func, distribution, log_dir=None, **kwargs):
        if isinstance(distribution,Iterable):
            train_single_model_ = partial(train_with_logger_multiple_dists, F=func, dists=distribution, dynamics_dim=self.dynamics_dim, **kwargs)
        else :
            train_single_model_ = partial(train_with_logger_ext_inp, F=func, dist=distribution, dynamics_dim=self.dynamics_dim, **kwargs)
        # else:
        #     raise ValueError("Invalid distribution type. Must be a torch.distributions.Distribution or a list of them.")

        if len(self.models) == 0:
            self.init_models()

        print(self.models)
        if self.use_multiprocessing:
            mp.set_start_method('spawn', force=True)
            devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else ["cpu"]

            with mp.Pool(processes=len(devices)) as pool:
                results = [
                    pool.apply_async(
                        train_single_model_,
                        args=(self.models[i],),
                        kwds=dict(verbose=self.verbose, device=self.device, metadata={"model_id": int(i)}),
                    ) for i in range(self.num_models)
                ]
        else:
            for i, model in enumerate(self.models):
                train_single_model_(
                    model,
                    verbose=self.verbose,
                    device=self.device,
                    metadata={"model_id": int(i)}
                )

        return self
    
    def predict(self, inputs, no_grad=True):
        """
        Predict the KEF outputs for the given inputs using the trained models.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, input_dim).
            no_grad (bool): If True, run without gradient computation.

        Returns:
            torch.Tensor: KEF outputs of shape (num_models, batch_size, output_dim).
        """
        kef_outputs = []
        for model in self.models:
            if no_grad:
                with torch.no_grad():
                    kef_output = model(inputs.to(self.device))
            else:
                kef_output = model(inputs.to(self.device))
            kef_outputs.append(kef_output.cpu())
        return torch.concat(kef_outputs, axis=-1)

    def to(self,device):
        self.device = device
        for model in self.models:
            model.to(device)

    def score(self, func, distribution, **kwargs):
        scores = []
        for model in self.models:
            if isinstance(distribution,Iterable):
                model_scores = []
                for dist in distribution:
                    score = eval_loss(model, func, dist, dynamics_dim=self.dynamics_dim, **kwargs)
                    model_scores.append(score)
                scores.append(torch.stack(model_scores))
            else:
                score = eval_loss(model, func, distribution, dynamics_dim=self.dynamics_dim, **kwargs)
                scores.append(score)
        self.scores = scores
        return torch.stack(self.scores)
    def save_models(self, savedir, filename=None):
        os.makedirs(Path(savedir)/"models", exist_ok=True)
        for i,model in enumerate(self.models):
            if filename is None:
                save_filename = f"{model.__class__.__name__}_{i}.pt"
            else:
                save_filename = f"{filename}.pt"
            torch.save(model.state_dict(), Path(savedir) / "models" / save_filename)

    def load_models(self, savedir, filename=None):
        for i,model in enumerate(self.models):
            if filename is None:
                load_filename = f"{model.__class__.__name__}_{i}.pt"
            else:
                load_filename = f"{filename}.pt"
            state_dict = torch.load(Path(savedir) / "models" / load_filename, weights_only=True, map_location=torch.device(self.device))
            self.models[i].load_state_dict(state_dict)

    def filter_models(self, threshold):
        assert (self.scores is not None)
        scores = self.scores
        self.models = [m for m, s in zip(self.models, scores) if torch.mean(s) < threshold]
        self.num_models = len(self.models)
        return self

    def compose_model_functions(self, model, **kwargs):
        """Compose the transformation functions for a single model without normalization.
        
        Args:
            model: The model to compose functions with
            **kwargs: Additional functions to include in the composition chain.
                     Supported keys:
                     - 'pre_functions': List of functions to apply before the main chain
                     - 'post_functions': List of functions to apply after the main chain
        """
        # Start with pre-functions if provided
        functions = []
        if 'pre_functions' in kwargs:
            functions.extend(kwargs['pre_functions'])
            
        # Add the main chain of functions
        functions.extend([
            torch.log,
            lambda x: x + 1,
            torch.exp,
            partial(torch.sum, dim=-1, keepdims=True),
            torch.log,
            torch.abs,
            model
        ])
        
        # Add post-functions if provided
        if 'post_functions' in kwargs:
            functions.extend(kwargs['post_functions'])
            
        return compose(*functions)

    def normalize_functions(self, functions, distribution, dist_needs_dim=True, **kwargs):
        """Normalize the given functions using samples from the distribution."""
        normalized_functions = []
        for f in functions:
            # Sample initial conditions
            shape = [1000] + ([self.dynamics_dim] if dist_needs_dim else [])
            samples_ic = distribution.sample(sample_shape=shape)
            
            # Handle external inputs if provided
            if "external_input_dist" in kwargs:
                ext_input_dist = kwargs["external_input_dist"]
                ext_input_dim = kwargs.get("external_input_dim", 0)
                shape_ext = [1000] + ([ext_input_dim] if dist_needs_dim else [])
                samples_ext = ext_input_dist.sample(sample_shape=shape_ext)
            else:
                samples_ext = torch.zeros_like(samples_ic)[...,0:0]
            
            # Combine samples and calculate normalization
            combined_samples = torch.cat((samples_ic, samples_ext), dim=-1)
            norm_val = float(torch.mean(torch.sum(f(combined_samples) ** 2, dim=-1)).sqrt().detach().numpy())
            
            # Normalize the function
            normalized_f = compose(lambda x: x / norm_val, f)
            normalized_functions.append(normalized_f)
            
        return normalized_functions

    def prepare_models_for_gradient_descent(self, distribution, **kwargs):
        """Prepare all models for gradient descent by composing and normalizing their functions."""
        if self.verbose:
            print('Preparing models for gradient descent...')
            
        # First compose the functions without normalization
        self.functions_for_gradient_descent = [self.compose_model_functions(model, **kwargs) for model in self.models]
        
        # Then normalize all functions
        self.functions_for_gradient_descent = self.normalize_functions(
            self.functions_for_gradient_descent,
            distribution,
            **kwargs
        )
        
        if self.verbose:
            print('Models are prepared for gradient descent.')
        return self.functions_for_gradient_descent

    def find_separatrix(self, distribution, dist_needs_dim=False,
                        return_indices=False, return_mask=False, **kwargs):
        all_traj, all_below, all_inds, all_masks = [], [], [], []
        # for model in self.models:
        for f in self.functions_for_gradient_descent:
            # f = compose(
            #     # lambda x: x ** 0.1,
            #     torch.log, lambda x: x + 1, torch.exp,
            #     partial(torch.sum, dim=-1, keepdims=True),
            #     torch.log, torch.abs, model
            # )
            # # Sample initial conditions.
            # shape = [1000] + ([self.dynamics_dim] if dist_needs_dim else [])
            # samples_ic = distribution.sample(sample_shape=shape)
            # # If an external input distribution is provided, sample external inputs.
            # if "external_input_dist" in kwargs:
            #     ext_input_dist = kwargs["external_input_dist"]
            #     ext_input_dim = kwargs.get("external_input_dim", 0)
            #     shape_ext = [1000] + ([ext_input_dim] if dist_needs_dim else [])
            #     samples_ext = ext_input_dist.sample(sample_shape=shape_ext)
            # else:
            #     # If not provided, use a dummy tensor (zeros) of the same shape as samples_ic.
            #     samples_ext = torch.zeros_like(samples_ic)[...,0:0]
            # # Concatenate the samples along the last dimension.
            # combined_samples = torch.cat((samples_ic, samples_ext), dim=-1)
            #
            # # Calculate the normalisation value over the combined inputs
            # norm_val = float(torch.mean(torch.sum(f(combined_samples) ** 2, dim=-1)).sqrt().detach().numpy())
            # # Update f to normalize its output.
            # f = compose(lambda x: x / norm_val, f)

            ret = runGD(
                f, distribution, input_dim=self.dynamics_dim, dist_needs_dim=dist_needs_dim,
                return_indices=return_indices, return_mask=return_mask, **kwargs
            )
            all_traj.append(ret[0])
            all_below.append(ret[1])
            off = 2
            if return_indices:
                all_inds.append(ret[off])
                off += 1
            if return_mask:
                all_masks.append(ret[off])
        res = [all_traj, all_below]
        if return_indices:
            res.append(all_inds)
        if return_mask:
            res.append(all_masks)
        return tuple(res)


if __name__ == '__main__':
    # model_class = KoopmanEigenfunctionModel
    from learn_koopman_eig import create_phi_network as model_class
    from torch.distributions import Normal, Uniform, MultivariateNormal
    # model_class = partial(model_class, num_layers=7, output_dim=10)
    model_class = partial(torch.nn.Linear,out_features=1)
    # dist = Normal(0, 1)
    dist = MultivariateNormal(torch.zeros(1), torch.eye(1))
    SL = SeparatrixLocator(
        num_models = 2,
        dynamics_dim = 1,
        use_multiprocessing = False,
        verbose = True,
        model_class = model_class
    )
    SL.init_models()
    # SL.fit(
    #     func = lambda x: x-x**3,
    #     distribution = dist,
    #     dist_requires_dim = True,
    #     batch_size = 2 #000
    # )
    inputs = dist.sample(sample_shape = (2,))
    print(
        SL.predict(inputs).shape
    )

    SL.prepare_models_for_gradient_descent(dist)

    from learn_koopman_eig import runGD_basic

    hidden = dist.sample(sample_shape = (2,))
