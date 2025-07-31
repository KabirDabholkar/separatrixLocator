"""
Minimal rSLDS Example
====================
Minimal code to load a trained rSLDS model and get the dynamics function.
"""

import pickle
import numpy as np
import torch


# Helper functions for plotting results
def get_most_likely_dynamics(model):
    """
    Returns a function that computes the most likely dynamics at any given state.
    
    Args:
        model: The rSLDS model with transitions and dynamics
        
    Returns:
        most_likely_dynamics: A function that takes state x and returns dx/dt
    """
    def most_likely_dynamics(x):
        """
        Compute the most likely dynamics at state x.
        
        Args:
            x: State vector of shape (D_latent,) or (N, D_latent) for batch
            
        Returns:
            dx_dt: Dynamics vector of same shape as x
        """
        # Ensure x is 2D for batch processing
        if x.ndim == 1:
            x = x.reshape(1, -1)
            single_input = True
        else:
            single_input = False
            
        N, D = x.shape
        K = model.K
        
        # For rSLDS, we need to create a dummy sequence to get transition probabilities
        # We'll duplicate each input point to create 2-point sequences
        x_seq = np.stack([x, x], axis=1)  # Shape: (N, 2, D)
        x_seq_flat = x_seq.reshape(-1, D)  # Shape: (2*N, D)
        
        # Create corresponding inputs and masks
        inputs = np.zeros((2*N, 0))  # No external inputs
        masks = np.ones((2*N, D), dtype=bool)
        
        # Get transition probabilities
        log_Ps = model.transitions.log_transition_matrices(
            x_seq_flat, inputs, masks, None)
        
        # Extract transition probabilities for each point
        # log_Ps has shape (2*N-1, K, K)
        # We want the transitions for each point: log_Ps[::2, :, :]
        transition_probs = np.exp(log_Ps[::2, :, :])  # Shape: (N, K, K)
        
        # Compute posterior for each point using uniform prior
        prior = np.ones(K) / K
        posterior = prior @ transition_probs  # Shape: (N, K)
        most_likely_states = np.argmax(posterior, axis=1)  # Shape: (N,)
        
        # Initialize output
        dx_dt = np.zeros_like(x)
        
        # Compute dynamics for all points at once using broadcasting
        for k, (A, b) in enumerate(zip(model.dynamics.As, model.dynamics.bs)):
            mask = most_likely_states == k
            if mask.any():
                dx_dt[mask] = x[mask].dot(A.T) + b - x[mask]
        
        # Return single vector if input was single
        if single_input:
            return dx_dt[0]
        else:
            return dx_dt
    
    return most_likely_dynamics


def get_weighted_average_dynamics(model, beta=1.0):
    """
    Returns a function that computes the weighted average dynamics at any given state.
    
    Args:
        model: The rSLDS model with transitions and dynamics
        
    Returns:
        weighted_average_dynamics: A function that takes state x and returns dx/dt
    """
    def weighted_average_dynamics(x):
        """
        Compute the weighted average dynamics at state x using posterior probabilities.
        
        Args:
            x: State vector of shape (D_latent,) or (N, D_latent) for batch
            
        Returns:
            dx_dt: Dynamics vector of same shape as x
        """
        # Ensure x is 2D for batch processing
        if x.ndim == 1:
            x = x.reshape(1, -1)
            single_input = True
        else:
            single_input = False
            
        N, D = x.shape
        K = model.K
        
        # For rSLDS, we need to create a dummy sequence to get transition probabilities
        # We'll duplicate each input point to create 2-point sequences
        x_seq = np.stack([x, x], axis=1)  # Shape: (N, 2, D)
        x_seq_flat = x_seq.reshape(-1, D)  # Shape: (2*N, D)
        
        # Create corresponding inputs and masks
        inputs = np.zeros((2*N, 0))  # No external inputs
        masks = np.ones((2*N, D), dtype=bool)
        
        # Get transition probabilities
        log_Ps = model.transitions.log_transition_matrices(
            x_seq_flat, inputs, masks, None)
        
        # Extract transition probabilities for each point
        # log_Ps has shape (2*N-1, K, K)
        # We want the transitions for each point: log_Ps[::2, :, :]
        transition_probs = np.exp(log_Ps[::2, :, :])  # Shape: (N, K, K)
        
        # Compute posterior for each point using uniform prior
        prior = np.ones(K) / K
        posterior = prior @ transition_probs  # Shape: (N, K)
        posterior = np.exp(beta * np.log(posterior))
        posterior = posterior / np.sum(posterior, axis=1, keepdims=True)
        
        # Initialize output
        dx_dt = np.zeros_like(x)
        
        # Compute weighted average dynamics across all discrete states
        for k, (A, b) in enumerate(zip(model.dynamics.As, model.dynamics.bs)):
            # Compute dynamics for state k: dx/dt = Ax + b - x
            dynamics_k = x.dot(A.T) + b - x  # Shape: (N, D)
            
            # Weight by posterior probability for state k
            weights = posterior[:, k:k+1]  # Shape: (N, 1) for broadcasting
            dx_dt += weights * dynamics_k
        
        # Return single vector if input was single
        if single_input:
            return dx_dt[0]
        else:
            return dx_dt
    
    return weighted_average_dynamics


# Global variable to cache the loaded model
_rslds_model = None
_model_file_path = None

def _load_rslds_model(model_file_path):
    """Load the rSLDS model lazily."""
    global _rslds_model, _model_file_path
    print(model_file_path)
    if _rslds_model is None or _model_file_path != model_file_path:
        with open(model_file_path, 'rb') as f:
            _rslds_model = pickle.load(f)
        _model_file_path = model_file_path
    return _rslds_model

def dynamics_fn(x, model, use_weighted_average=False, beta=1.0):
    """
    Dynamics function that uses the provided model.
    
    Args:
        x: State vector
        model: The rSLDS model
        use_weighted_average: If True, use weighted average dynamics; if False, use most likely dynamics
    """
    if use_weighted_average:
        return get_weighted_average_dynamics(model, beta=beta)(x)
    else:
        return get_most_likely_dynamics(model)(x)

def torchify(func, model_file):
    """
    Wraps a numpy dynamics function to make it compatible with torch tensors.
    Loads the model once and caches it.
    """
    # Load the model once when torchify is called
    model = _load_rslds_model(model_file)
    
    def wrapped(x, *args, **kwargs):
        return torch.from_numpy(
            func(
                x.detach().cpu().numpy(),
                model,
                *args,
                **kwargs
            )
        ).to(x.device).to(x.dtype)
    return wrapped

 

if __name__ == "__main__":
    # Test numpy dynamics
    model = _load_rslds_model('results/bistable_rslds_k5/rslds_best_model.pkl')
    x_np = np.array([1.0, 0.5])
    
    # Test most likely dynamics
    dx_dt_ml = dynamics_fn(x_np, model, use_weighted_average=False)
    print(f"Numpy state: {x_np}")
    print(f"Most likely dynamics: {dx_dt_ml}")
    
    # Test weighted average dynamics
    dx_dt_wa = dynamics_fn(x_np, model, use_weighted_average=True)
    print(f"Weighted average dynamics: {dx_dt_wa}")

    # Test torch dynamics 
    x_torch = torch.tensor([1.0, 0.5])
    torch_func = torchify(dynamics_fn, 'results/bistable_rslds_k5/rslds_best_model.pkl')
    dx_dt_torch = torch_func(x_torch)
    print(f"\nTorch state: {x_torch}")
    print(f"Torch dynamics: {dx_dt_torch}")