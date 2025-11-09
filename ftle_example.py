#!/usr/bin/env python3
"""
Example script demonstrating FTLE computation using odeint_adjoint.

This script shows how to:
1. Define a dynamical system
2. Compute FTLE field over a grid of initial conditions
3. Visualize the results
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from ftle_computer import FTLEComputer


def van_der_pol_dynamics(t, x, mu=1.0):
    """
    Van der Pol oscillator: dx/dt = y, dy/dt = mu*(1-x^2)*y - x
    
    Args:
        t: Time (unused but required by interface)
        x: State vector of shape (batch_size, 2)
        mu: Parameter controlling the nonlinearity
        
    Returns:
        Derivatives dx/dt of shape (batch_size, 2)
    """
    dx = x[:, 1]
    dy = mu * (1 - x[:, 0]**2) * x[:, 1] - x[:, 0]
    return torch.stack([dx, dy], dim=1)


def lorenz_dynamics(t, x, sigma=10.0, rho=28.0, beta=8/3):
    """
    Lorenz system: dx/dt = sigma*(y-x), dy/dt = x*(rho-z)-y, dz/dt = x*y-beta*z
    
    Args:
        t: Time (unused but required by interface)
        x: State vector of shape (batch_size, 3)
        sigma, rho, beta: System parameters
        
    Returns:
        Derivatives dx/dt of shape (batch_size, 3)
    """
    dx = sigma * (x[:, 1] - x[:, 0])
    dy = x[:, 0] * (rho - x[:, 2]) - x[:, 1]
    dz = x[:, 0] * x[:, 1] - beta * x[:, 2]
    return torch.stack([dx, dy, dz], dim=1)


def compute_ftle_field_2d(dynamics_func, x_range, y_range, t_span, 
                         integration_time=None, **kwargs):
    """
    Compute FTLE field over a 2D grid.
    
    Args:
        dynamics_func: Function defining the dynamical system
        x_range: Tuple of (x_min, x_max, n_x)
        y_range: Tuple of (y_min, y_max, n_y)
        t_span: Time points for integration
        integration_time: Time interval for FTLE computation
        **kwargs: Additional arguments for dynamics_func
        
    Returns:
        Tuple of (X, Y, FTLE_field) where X, Y are meshgrids and FTLE_field is the FTLE values
    """
    x_min, x_max, n_x = x_range
    y_min, y_max, n_y = y_range
    
    # Create grid
    x = np.linspace(x_min, x_max, n_x)
    y = np.linspace(y_min, y_max, n_y)
    X, Y = np.meshgrid(x, y)
    
    # Flatten grid for batch processing
    x_grid = torch.tensor(np.column_stack([X.flatten(), Y.flatten()]), 
                         dtype=torch.float32)
    
    # Initialize FTLE computer
    ftle_comp = FTLEComputer(dynamics_func, device="cpu")
    
    # Compute FTLE field
    ftle_values = ftle_comp.compute_ftle_field_adjoint(
        x_grid, t_span, integration_time, **kwargs
    )
    
    # Reshape back to grid
    FTLE_field = ftle_values.numpy().reshape(X.shape)
    
    return X, Y, FTLE_field


def plot_ftle_field(X, Y, FTLE_field, title="FTLE Field", save_path=None):
    """
    Plot FTLE field.
    
    Args:
        X, Y: Meshgrid coordinates
        FTLE_field: FTLE values
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(10, 8))
    
    # Create contour plot
    contour = plt.contourf(X, Y, FTLE_field, levels=50, cmap='viridis')
    plt.colorbar(contour, label='FTLE')
    
    # Add contour lines for separatrix-like structures
    plt.contour(X, Y, FTLE_field, levels=20, colors='black', alpha=0.3, linewidths=0.5)
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def example_van_der_pol():
    """Example: Compute FTLE field for Van der Pol oscillator."""
    print("Computing FTLE field for Van der Pol oscillator...")
    
    # Parameters
    x_range = (-3.0, 3.0, 50)
    y_range = (-3.0, 3.0, 50)
    t_span = torch.linspace(0, 5.0, 50)
    integration_time = 5.0
    
    # Compute FTLE field
    X, Y, FTLE_field = compute_ftle_field_2d(
        van_der_pol_dynamics, x_range, y_range, t_span, 
        integration_time, mu=1.0
    )
    
    # Plot results
    plot_ftle_field(X, Y, FTLE_field, 
                   title="FTLE Field - Van der Pol Oscillator",
                   save_path="van_der_pol_ftle_field.png")
    
    print(f"FTLE range: [{FTLE_field.min():.4f}, {FTLE_field.max():.4f}]")
    print(f"Mean FTLE: {FTLE_field.mean():.4f}")


def example_lorenz_2d_slice():
    """Example: Compute FTLE field for Lorenz system on a 2D slice."""
    print("Computing FTLE field for Lorenz system (2D slice)...")
    
    # Parameters - slice at z = 1
    x_range = (-20.0, 20.0, 40)
    y_range = (-20.0, 20.0, 40)
    t_span = torch.linspace(0, 3.0, 30)
    integration_time = 3.0
    
    # Define dynamics on 2D slice (z = 1)
    def lorenz_2d_dynamics(t, x, sigma=10.0, rho=28.0, beta=8/3, z_slice=1.0):
        """Lorenz dynamics on 2D slice at z = z_slice"""
        dx = sigma * (x[:, 1] - x[:, 0])
        dy = x[:, 0] * (rho - z_slice) - x[:, 1]
        return torch.stack([dx, dy], dim=1)
    
    # Compute FTLE field
    X, Y, FTLE_field = compute_ftle_field_2d(
        lorenz_2d_dynamics, x_range, y_range, t_span, 
        integration_time, sigma=10.0, rho=28.0, beta=8/3, z_slice=1.0
    )
    
    # Plot results
    plot_ftle_field(X, Y, FTLE_field, 
                   title="FTLE Field - Lorenz System (z=1 slice)",
                   save_path="lorenz_2d_ftle_field.png")
    
    print(f"FTLE range: [{FTLE_field.min():.4f}, {FTLE_field.max():.4f}]")
    print(f"Mean FTLE: {FTLE_field.mean():.4f}")


def example_single_trajectory():
    """Example: Compute FTLE for a single trajectory."""
    print("Computing FTLE for single trajectory...")
    
    # Initialize FTLE computer
    ftle_comp = FTLEComputer(van_der_pol_dynamics, device="cpu")
    
    # Single initial condition
    x0 = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    t_span = torch.linspace(0, 5.0, 50)
    
    # Compute trajectory and FTLE
    trajectories, jacobians = ftle_comp.compute_flow_map_jacobian_adjoint(x0, t_span, mu=1.0)
    ftle = ftle_comp.compute_ftle_adjoint(x0, t_span, integration_time=5.0, mu=1.0)
    ftle_spectrum = ftle_comp.compute_ftle_spectrum_adjoint(x0, t_span, integration_time=5.0, mu=1.0)
    
    print(f"Initial condition: {x0[0].numpy()}")
    print(f"Final position: {trajectories[-1, 0].numpy()}")
    print(f"FTLE: {ftle[0].item():.4f}")
    print(f"FTLE spectrum: {ftle_spectrum[0].numpy()}")
    
    # Plot trajectory
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(trajectories[:, 0, 0].numpy(), trajectories[:, 0, 1].numpy(), 'b-', linewidth=2)
    plt.plot(x0[0, 0].numpy(), x0[0, 1].numpy(), 'ro', markersize=8, label='Initial')
    plt.plot(trajectories[-1, 0, 0].numpy(), trajectories[-1, 0, 1].numpy(), 'go', markersize=8, label='Final')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Trajectory')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(t_span.numpy(), trajectories[:, 0, 0].numpy(), 'b-', label='x(t)')
    plt.plot(t_span.numpy(), trajectories[:, 0, 1].numpy(), 'r-', label='y(t)')
    plt.xlabel('Time')
    plt.ylabel('State')
    plt.title('Time Evolution')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("single_trajectory.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("FTLE Computation Examples using odeint_adjoint")
    print("=" * 50)
    
    # Example 1: Single trajectory
    example_single_trajectory()
    
    # Example 2: Van der Pol FTLE field
    example_van_der_pol()
    
    # Example 3: Lorenz 2D slice
    example_lorenz_2d_slice()
    
    print("\nAll examples completed!")
    print("\nKey features demonstrated:")
    print("✓ Automatic differentiation with odeint_adjoint")
    print("✓ FTLE computation for single trajectories")
    print("✓ FTLE field computation over grids")
    print("✓ Visualization of separatrix-like structures") 