import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_flow_streamlines(F, ax, x_limits=(-2, 2), y_limits=(-2, 2), resolution=500, density=1.0, color='k', linewidth=0.5, alpha=1.0):
    """
    Plots the flow field using streamlines for a given vector field F.

    Parameters:
        F (callable): Function that computes the vector field.
        ax (matplotlib.axes.Axes): The axis object where the plot will be drawn.
        x_limits (tuple): Limits for the x-axis.
        y_limits (tuple): Limits for the y-axis.
        resolution (int): Resolution for the grid.
        density (float): Controls the closeness of streamlines.
        color (str): Color of the streamlines.
        linewidth (float): Width of the streamlines.
        alpha (float): Transparency of the streamlines.
    """
    # Define grid for streamlines
    x = np.linspace(x_limits[0], x_limits[1], resolution)
    y = np.linspace(y_limits[0], y_limits[1], resolution)
    X, Y = np.meshgrid(x, y)
    grid = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1), dtype=torch.float32)

    # Compute F values for the grid
    F_val = F(grid).detach().cpu().numpy()
    U = F_val[:, 0].reshape(resolution, resolution)
    V = F_val[:, 1].reshape(resolution, resolution)

    # Plot streamlines
    lines = ax.streamplot(X, Y, U, V, density=density, color=color, linewidth=linewidth)
    lines.lines.set_alpha(alpha)

def evaluate_on_grid(func, x_limits=(-1.3, 1.3), y_limits=(-1.3, 1.3), resolution=50):
    """
    Evaluates a function on a 2D grid.
    
    Parameters:
        func (callable): Function to evaluate
        x_limits (tuple): Limits for the x-axis
        y_limits (tuple): Limits for the y-axis
        resolution (int): Resolution for the grid
        
    Returns:
        tuple: (X, Y, Z) where X and Y are meshgrid arrays and Z contains the function values
    """
    # Create grid
    x = np.linspace(x_limits[0], x_limits[1], resolution)
    y = np.linspace(y_limits[0], y_limits[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    # Compute function values
    grid = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1), dtype=torch.float32)
    Z = func(grid).detach().cpu().numpy()
    Z = Z.reshape(resolution, resolution)
    
    return X, Y, Z

def dynamics_to_kinetic_energy(F):
    """
    Converts a dynamics function F to a kinetic energy function.
    
    Parameters:
        F (callable): Function that computes the vector field
        
    Returns:
        callable: Function that computes the kinetic energy
    """
    def kinetic_energy(x):
        F_val = F(x)
        return torch.sum(F_val**2, dim=-1)
    return kinetic_energy

def compute_kinetic_energy_grid(F, x_limits=(-1.3, 1.3), y_limits=(-1.3, 1.3), resolution=50):
    """
    Computes the kinetic energy grid for a given vector field F.
    
    Parameters:
        F (callable): Function that computes the vector field
        x_limits (tuple): Limits for the x-axis
        y_limits (tuple): Limits for the y-axis
        resolution (int): Resolution for the grid
        
    Returns:
        tuple: (X, Y, kinetic_energy) where X and Y are meshgrid arrays and kinetic_energy is the computed energy
    """
    kinetic_energy_func = dynamics_to_kinetic_energy(F)
    return evaluate_on_grid(kinetic_energy_func, x_limits, y_limits, resolution)

def compute_separatrix_grid(separatrix_function, x_limits=(-1.3, 1.3), y_limits=(-1.3, 1.3), resolution=50):
    """
    Computes the separatrix function grid.
    
    Parameters:
        separatrix_function (callable): Function that computes the separatrix value
        x_limits (tuple): Limits for the x-axis
        y_limits (tuple): Limits for the y-axis
        resolution (int): Resolution for the grid
        
    Returns:
        tuple: (X, Y, separatrix) where X and Y are meshgrid arrays and separatrix is the computed values
    """
    return evaluate_on_grid(separatrix_function, x_limits, y_limits, resolution)

def plot_kinetic_energy_surface(F, ax, x_limits=(-1.3, 1.3), y_limits=(-1.3, 1.3), resolution=50):
    """
    Plots the kinetic energy as a surface in 3D for a given vector field F.
    
    Parameters:
        F (callable): Function that computes the vector field
        ax (matplotlib.axes.Axes): The 3D axis object where the plot will be drawn
        x_limits (tuple): Limits for the x-axis
        y_limits (tuple): Limits for the y-axis
        resolution (int): Resolution for the grid
    """
    # Get the grid and kinetic energy values
    X, Y, kinetic_energy = compute_kinetic_energy_grid(F, x_limits, y_limits, resolution)
    
    # Plot surface
    surf = ax.plot_surface(X, Y, kinetic_energy, cmap='plasma', 
                          alpha=0.1, linewidth=0, antialiased=True)

def plot_kinetic_energy_contour(F, ax, x_limits=(-1.3, 1.3), y_limits=(-1.3, 1.3), resolution=50, levels=20):
    """
    Plots the kinetic energy as a contour plot for a given vector field F.
    
    Parameters:
        F (callable): Function that computes the vector field
        ax (matplotlib.axes.Axes): The axis object where the plot will be drawn
        x_limits (tuple): Limits for the x-axis
        y_limits (tuple): Limits for the y-axis
        resolution (int): Resolution for the grid
        levels (int): Number of contour levels
    """
    # Get the grid and kinetic energy values
    X, Y, kinetic_energy = compute_kinetic_energy_grid(F, x_limits, y_limits, resolution)
    
    # Plot contour
    contour = ax.contourf(X, Y, np.log(kinetic_energy), levels=levels, cmap='Blues_r')

def plot_kinetic_energy(F, ax, x_limits=(-2, 2), y_limits=(-2, 2), heatmap_resolution=500, quiver_resolution=25,
                        below_threshold_points=None):
    """
    Plots the kinetic energy heatmap with quiver plot for a given vector field F.

    Parameters:
        F (callable): Function that computes the vector field.
        ax (matplotlib.axes.Axes): The axis object where the plot will be drawn.
        x_limits (tuple): Limits for the x-axis.
        y_limits (tuple): Limits for the y-axis.
        heatmap_resolution (int): Resolution for the heatmap.
        quiver_resolution (int): Resolution for the quiver plot.
        below_threshold_points (ndarray, optional): Points to highlight on the plot.
    """
    # Define grid for heatmap
    x_heatmap = torch.linspace(x_limits[0], x_limits[1], heatmap_resolution)
    y_heatmap = torch.linspace(y_limits[0], y_limits[1], heatmap_resolution)
    X_heatmap, Y_heatmap = torch.meshgrid(x_heatmap, y_heatmap, indexing='ij')
    heatmap_grid = torch.stack([X_heatmap.flatten(), Y_heatmap.flatten()], dim=-1)

    # Compute F values for the heatmap grid
    F_val_heatmap = F(heatmap_grid).detach().cpu().numpy()
    Fx_heatmap = F_val_heatmap[:, 0].reshape(heatmap_resolution, heatmap_resolution)
    Fy_heatmap = F_val_heatmap[:, 1].reshape(heatmap_resolution, heatmap_resolution)
    kinetic_energy = Fx_heatmap ** 2 + Fy_heatmap ** 2

    # Plot heatmap
    im = ax.imshow(
        np.log(kinetic_energy).T, extent=[x_limits[0], x_limits[1], y_limits[0], y_limits[1]],
        origin='lower', aspect='auto', cmap='plasma'
    )

    # Define grid for quiver plot
    x_quiver = torch.linspace(x_limits[0], x_limits[1], quiver_resolution)
    y_quiver = torch.linspace(y_limits[0], y_limits[1], quiver_resolution)
    X_quiver, Y_quiver = torch.meshgrid(x_quiver, y_quiver, indexing='ij')
    quiver_grid = torch.stack([X_quiver.flatten(), Y_quiver.flatten()], dim=-1)

    # Compute F values for the quiver grid
    F_val_quiver = F(quiver_grid).detach().cpu().numpy()
    Fx_quiver = F_val_quiver[:, 0].reshape(quiver_resolution, quiver_resolution)
    Fy_quiver = F_val_quiver[:, 1].reshape(quiver_resolution, quiver_resolution)

    # Plot quiver plot
    ax.quiver(
        X_quiver, Y_quiver, Fx_quiver, Fy_quiver, color='white', scale=50, width=0.002, headwidth=3,
        pivot='middle'
    )

    # Plot below threshold points if provided
    if below_threshold_points is not None:
        xlim = ax.get_xlim()  # Store current x limits
        ylim = ax.get_ylim()  # Store current y limits
        ax.scatter(below_threshold_points[:, 0], below_threshold_points[:, 1], c='red', zorder=1000, s=5)
        ax.set_xlim(xlim)  # Reset x limits
        ax.set_ylim(ylim)  # Reset y limits

    # Axis labels and title
    ax.set_title('Kinetic Energy and Vector Field of $F(x, y)$')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')

    return im


def plot_model_contour(model, ax, x_limits=(-2, 2), y_limits=(-2, 2), resolution=500,
                       below_threshold_points=None):
    """
    Plots the contour of log(phi+1) for a given model on the provided matplotlib axis.

    Parameters:
        model_to_GD_on (callable): Function that computes phi values for a given input grid.
        ax (matplotlib.axes.Axes): The axis object where the plot will be drawn.
        x_limits (tuple): Limits for the x-axis.
        y_limits (tuple): Limits for the y-axis.
        resolution (int): Resolution for the heatmap.
        below_threshold_points (ndarray, optional): Points to highlight on the plot.
    """
    # Define grid for heatmap
    x_heatmap = torch.linspace(x_limits[0], x_limits[1], resolution)
    y_heatmap = torch.linspace(y_limits[0], y_limits[1], resolution)
    X_heatmap, Y_heatmap = torch.meshgrid(x_heatmap, y_heatmap, indexing='ij')
    heatmap_grid = torch.stack([X_heatmap.flatten(), Y_heatmap.flatten()], dim=-1)

    # Compute phi values
    phi_val = model(heatmap_grid).detach().cpu().numpy()
    phi_val = phi_val.reshape(resolution, resolution)

    # Plot contour
    contour = ax.contourf(X_heatmap, Y_heatmap, phi_val, levels=20,  cmap='viridis')

    # Plot below threshold points if provided
    if below_threshold_points is not None:
        ax.scatter(below_threshold_points[:, 0], below_threshold_points[:, 1], c='red', zorder=1000)

    # Axis labels and title
    ax.set_title(r'Contour plot of $\log (\phi(x, y)+1)$')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')

    return contour


def remove_frame(ax, spines_to_remove=['top', 'right', 'bottom', 'left']):
    """
    Removes the frame and ticks from a matplotlib axis.

    Parameters:
        ax (matplotlib.axes.Axes): The axis to modify
        spines_to_remove (list): List of spines to remove. Default removes all spines.
    """
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in spines_to_remove:
        ax.spines[spine].set_visible(False)