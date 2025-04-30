"""
This module contains functions for creating cartoon-style visualizations of the separatrix and related quantities.
These plots are designed to be more illustrative and easier to understand than the detailed numerical plots.
"""
from torchdiffeq import odeint
import torch
import numpy as np
import matplotlib.pyplot as plt
from dynamical_functions import affine_bistable2D
from plotting import plot_flow_streamlines
import seaborn as sns
from functools import partial
import operator
colors = sns.color_palette("bright")

def remove_frame(ax):
    """
    Removes the frame and ticks from a matplotlib axis.
    
    Parameters:
        ax (matplotlib.axes.Axes): The axis to modify
    """
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_aspect('equal')

def bistable_affine_separatrix_function(x):
    return (x[...,0] + x[...,1])**2

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

def compute_separatrix_grid(x_limits=(-1.3, 1.3), y_limits=(-1.3, 1.3), resolution=50):
    """
    Computes the separatrix function grid.
    
    Parameters:
        x_limits (tuple): Limits for the x-axis
        y_limits (tuple): Limits for the y-axis
        resolution (int): Resolution for the grid
        
    Returns:
        tuple: (X, Y, separatrix) where X and Y are meshgrid arrays and separatrix is the computed values
    """
    return evaluate_on_grid(bistable_affine_separatrix_function, x_limits, y_limits, resolution)

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

def plot_separatrix_surface(ax, x_limits=(-1.3, 1.3), y_limits=(-1.3, 1.3), resolution=50):
    """
    Plots the separatrix function as a surface in 3D.
    
    Parameters:
        ax (matplotlib.axes.Axes): The 3D axis object where the plot will be drawn
        x_limits (tuple): Limits for the x-axis
        y_limits (tuple): Limits for the y-axis
        resolution (int): Resolution for the grid
    """
    # Get the grid and separatrix values
    X, Y, separatrix = compute_separatrix_grid(x_limits, y_limits, resolution)
    
    # Plot surface
    surf = ax.plot_surface(X, Y, separatrix, cmap='Blues_r', 
                          alpha=0.1, linewidth=0, antialiased=True)

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
    
    # Add colorbar
    # fig = ax.get_figure()
    # fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

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
    
    # Add colorbar
    # fig = ax.get_figure()
    # fig.colorbar(contour, ax=ax, shrink=0.5, aspect=5)

def cartoon_traj_and_inputs(dynamics_function,ax,T=20,steps=100):
    time_points = torch.linspace(0, T, steps)
    traj = [[-1, 0], [-1, 1.2]]
    traj = torch.tensor(traj, dtype=torch.float32)
    traj = torch.concat([
        traj,
        odeint(lambda t, x: dynamics_function(x), traj[-1], time_points).detach().cpu()
    ])
    ax.plot(traj[..., 0], traj[..., 1])

def base_cartoon_plot(ax,x_limits=(-1.3,1.3),y_limits=(-1.3,1.3)):
    # Check if the axis is 3D
    is_3d = hasattr(ax, 'zaxis')
    
    pointsets = [
        {
            'x':[1,-1],
            'y':[0,0],
            'marker':'o',
            'label': 'stable fixed point',
            's' : 100,
            'zorder': 2,
            'color' : colors[2],
        },
        {
            'x':[0],
            'y':[0],
            'marker':'+',
            'label': 'unstable fixed point',
            's'  : 300,
            'zorder': 2,
            'linewidths': 3,
            'color' : colors[2],
        }
    ]
    
    for pointset in pointsets:
        # Convert parameters for 3D scatter if needed
        if is_3d:
            l = len(pointset['x'])
            scatter_params = {
                'xs': pointset.pop('x'),
                'ys': pointset.pop('y'),
                'zs': [0] * l,  # Add z-coordinates
                **pointset
            }
        else:
            scatter_params = pointset
            
        ax.scatter(**scatter_params)

    plot_args = {
        'ls': 'dashed',
        'lw': 3,
        'color' : colors[1],
        'alpha': 1,
        'zorder': 1,
    }
    x = np.linspace(*x_limits)
    y = -x
    if is_3d:
        # For 3D plot, add z=0 coordinates
        z = np.zeros_like(x)
        ax.plot(x, y, z, **plot_args)
    else:
        ax.plot(x, y, **plot_args)
        
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)

def plot_3d_cartoon(ax, x_limits=(-1.3,1.3), y_limits=(-1.3,1.3), z_limits=(0,1.3)):

    # Set 3D view
    ax.view_init(elev=30, azim=-80)
    
    # Set axis limits
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_zlim(*z_limits)
    
    # Add z-axis label
    # ax.set_zlabel('z', rotation=0)
    
    # Add grid
    ax.grid(False)
    
    # Set aspect ratio
    ax.set_box_aspect([1,1,1])

    # ax.set_axis_off()



if __name__ == '__main__':
    fig, ax = plt.subplots()
    # plot_flow_streamlines(affine_bistable2D, ax, x_limits=(-1.3, 1.3), y_limits=(-1.3, 1.3), resolution=50,
    #                       density=0.25, color='k', linewidth=0.5, alpha=0.4)
    base_cartoon_plot(ax)
    cartoon_traj_and_inputs(affine_bistable2D,ax)
    remove_frame(ax)
    # plt.show()
    # fig.savefig('plots_for_publication/cartoon_plot.pdf', bbox_inches='tight', dpi=300)
    fig.savefig('plots_for_publication/cartoon_plot.png', bbox_inches='tight', dpi=300)

    # Create 3D plot
    # fig = plt.figure(figsize=(4,4))
    # ax = fig.add_subplot(111, projection='3d')
    # plot_3d_cartoon(ax)
    # base_cartoon_plot(ax)
    # plot_kinetic_energy_surface(affine_bistable2D, ax)
    # remove_frame(ax)
    # # fig.savefig('plots_for_publication/cartoon_3d_plot.pdf', bbox_inches='tight', dpi=300)
    # fig.savefig('plots_for_publication/cartoon_3d_plot.png', bbox_inches='tight', dpi=300)


    # # Create 2D plot with kinetic energy contours
    # fig, ax = plt.subplots(figsize=(4,4))
    # plot_kinetic_energy_contour(affine_bistable2D, ax, x_limits=(-2, 2), y_limits=(-2, 2), resolution=200, levels=15)
    # base_cartoon_plot(ax)
    # remove_frame(ax)
    
    # # Save 2D figure
    # fig.savefig('plots_for_publication/cartoon_2d_kinetic_energy_contour.png', bbox_inches='tight', dpi=300)

    # Create 2D plot with kinetic energy contours
    fig, ax = plt.subplots(figsize=(4,4))
    X,Y,separatrix_function = evaluate_on_grid(bistable_affine_separatrix_function, x_limits=(-2, 2), y_limits=(-2, 2), resolution=200)
    ax.contourf(X, Y, (separatrix_function)**0.5, levels=15, cmap='Blues_r')
    base_cartoon_plot(ax)
    remove_frame(ax)
    
    # Save 2D figure
    fig.savefig('plots_for_publication/cartoon_2d_separatrix_contour.png', bbox_inches='tight', dpi=300)