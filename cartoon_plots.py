"""
This module contains functions for creating cartoon-style visualizations of the separatrix and related quantities.
These plots are designed to be more illustrative and easier to understand than the detailed numerical plots.
"""
from torchdiffeq import odeint
import torch
import numpy as np
import matplotlib.pyplot as plt
from dynamical_functions import affine_bistable2D
from plotting import (
    plot_flow_streamlines,
    evaluate_on_grid,
    dynamics_to_kinetic_energy,
    compute_kinetic_energy_grid,
    plot_kinetic_energy_surface,
    plot_kinetic_energy_contour,
    compute_separatrix_grid,
    remove_frame
)
import seaborn as sns
from functools import partial
import operator
colors = sns.color_palette("bright")


plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

def plot_arrows(ax, arrow_length = 0.5, arrow_start = [-1, -1.3], fontsize=12, zorder=5):
    # Vertical arrow
    ax.arrow(arrow_start[0], arrow_start[1], 0, arrow_length,
             head_width=0.1, head_length=0.1, fc='k', ec='k', )
    ax.text(arrow_start[0], arrow_start[1] + arrow_length + 0.13, 'input', 
            ha='center', va='bottom', fontsize=fontsize, zorder = zorder)

    # Diagonal arrow
    ax.arrow(arrow_start[0], arrow_start[1], arrow_length / np.sqrt(2), arrow_length / np.sqrt(2),
             head_width=0.1, head_length=0.1, fc='k', ec='k')
    ax.text(arrow_start[0] + arrow_length / np.sqrt(2) + 0.1,
            arrow_start[1] + arrow_length / np.sqrt(2) + 0.1,
            'optimal', fontsize=fontsize, zorder = zorder)
    # Horizontal arrow
    ax.arrow(arrow_start[0], arrow_start[1], arrow_length, 0,
             head_width=0.1, head_length=0.1, fc='k', ec='k')
    ax.text(arrow_start[0] + arrow_length + 0.13, arrow_start[1], 'choice', 
            ha='left', va='center', fontsize=fontsize, zorder = zorder)

def bistable_affine_separatrix_function(x):
    return (x[...,0] + x[...,1])**2

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

def cartoon_traj_and_inputs(dynamics_function,ax,T=20,steps=100):
    time_points = torch.linspace(0, T, steps)
    traj = [[-1, 0], [-1, 1.2]]
    traj = torch.tensor(traj, dtype=torch.float32)
    traj = torch.concat([
        traj,
        odeint(lambda t, x: dynamics_function(x), traj[-1], time_points).detach().cpu()
    ])
    ax.arrow(traj[0, 0], traj[0, 1], traj[1, 0] - traj[0, 0], traj[1, 1] - traj[0, 1],
                 head_width=0.1, head_length=0.1, fc='k', ec='k', length_includes_head=True)
    ax.plot(traj[2:, 0], traj[2:, 1])
    ax.arrow(-1, 0, 0.5, 0.5, head_width=0.1, head_length=0.1, fc='k', ec='k', length_includes_head=True)
    # Add a small square to show 90 degree angle
    # Add 90-degree angle marker at (-0.5, 0.5)
    size = 0.15  # size of the angle marker
    longer_edge = np.sqrt(2*size**2)
    shorter_edge = np.sqrt(0.5*size**2)
    corner = (-0.5-longer_edge, 0.5)
    ax.plot(
        [corner[0], corner[0] + shorter_edge],
        [corner[1], corner[1] + shorter_edge],
        color="k", linewidth=1, zorder=5,
    )
    ax.plot(
        [corner[0], corner[0] + shorter_edge],
        [corner[1], corner[1] - shorter_edge],
        color="k", linewidth=1, zorder=5,
    )

def base_cartoon_plot(ax,x_limits=(-1.3,1.3),y_limits=(-1.3,1.3),fontsize=12,show_separatrix_label=True):
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
            scatter_params = {
                'xs': pointset.pop('x'),
                'ys': pointset.pop('y'),
                'zs': [0] * len(pointset['xs']),  # Add z-coordinates
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
        
    # Add text label for the separatrix if requested
    if show_separatrix_label:
        if is_3d:
            # For 3D plot, position text at midpoint of line
            mid_idx = int(len(x) * 0.75)
            ax.text(x[mid_idx], y[mid_idx], 0, 'separatrix',
                    ha='center', va='center', rotation=-45, color=colors[1], fontsize=fontsize)
        else:
            # For 2D plot, position text at midpoint of line
            mid_idx = int(len(x) * 0.75)
            ax.text(x[mid_idx], y[mid_idx]-0.08, 'separatrix',
                    ha='center', va='bottom', rotation=-45, color=colors[1], fontsize=fontsize)
        
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    if is_3d:
        ax.set_zlim(-1.3, 1.3)

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
    # fig, ax = plt.subplots()
    # plot_flow_streamlines(affine_bistable2D, ax, x_limits=(-1.3, 1.3), y_limits=(-1.3, 1.3), resolution=50,
    #                       density=0.4, color='skyblue', linewidth=0.5, alpha=0.4)
    # base_cartoon_plot(ax,fontsize=19)
    # plot_arrows(ax,arrow_length=0.3, arrow_start=[-1.1,-1],fontsize=19)
    # cartoon_traj_and_inputs(affine_bistable2D,ax)
    # remove_frame(ax)
    # # plt.show()
    # # fig.savefig('plots_for_publication/cartoon_plot.pdf', bbox_inches='tight', dpi=300)
    # fig.savefig('plots_for_publication/cartoon_plot.png', bbox_inches='tight', dpi=300)

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
    # kinetic_energy_function = dynamics_to_kinetic_energy(affine_bistable2D)
    # X, Y, kinetic_energy_vals = evaluate_on_grid(kinetic_energy_function, x_limits=(-2, 2),y_limits=(-2, 2), resolution=200)
    # ax.contourf(X, Y, np.log(kinetic_energy_vals+0.5), levels=15, cmap='Blues_r')
    # # plot_kinetic_energy_contour(affine_bistable2D, ax, x_limits=(-2, 2), y_limits=(-2, 2), resolution=200, levels=15)
    # base_cartoon_plot(ax)
    # remove_frame(ax)
    #
    # # Save 2D figure
    # fig.savefig('plots_for_publication/cartoon_2d_kinetic_energy_contour.png', bbox_inches='tight', dpi=300)
    #
    # # Create 2D plot with kinetic energy contours
    # fig, ax = plt.subplots(figsize=(4,4))
    # X,Y,separatrix_function = evaluate_on_grid(bistable_affine_separatrix_function, x_limits=(-2, 2), y_limits=(-2, 2), resolution=200)
    # ax.contourf(X, Y, np.log(separatrix_function+0.25), levels=10, cmap='Blues_r')
    # base_cartoon_plot(ax)
    # remove_frame(ax)
    #
    # # Save 2D figure
    # fig.savefig('plots_for_publication/cartoon_2d_separatrix_contour.png', bbox_inches='tight', dpi=300)

    # Create subplots with 1 row and 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = [r'$\dot x = f(x)$',r'$q(x):=\Vert f(x)\Vert^2$',r'$\psi(x):=\ ?$']
    for ax,title in zip(axes,titles):
        ax.set_title(title,fontsize=20)

    # First subplot: Flow streamlines
    ax = axes[0]
    plot_flow_streamlines(affine_bistable2D, ax, x_limits=(-1.3, 1.3), y_limits=(-1.3, 1.3), 
                         resolution=50, density=0.4, color='skyblue', linewidth=0.5, alpha=0.4)
    base_cartoon_plot(ax, fontsize=19)
    plot_arrows(ax, arrow_length=0.3, arrow_start=[-1.1,-1], fontsize=19)
    cartoon_traj_and_inputs(affine_bistable2D, ax)
    remove_frame(ax)

    # Second subplot: Kinetic energy contours
    ax = axes[1]
    kinetic_energy_function = dynamics_to_kinetic_energy(affine_bistable2D)
    X, Y, kinetic_energy_vals = evaluate_on_grid(kinetic_energy_function, 
                                               x_limits=(-2, 2), y_limits=(-2, 2), resolution=200)
    ax.contourf(X, Y, np.log(kinetic_energy_vals+0.5), levels=15, cmap='Blues_r')
    base_cartoon_plot(ax, show_separatrix_label=False)
    remove_frame(ax)

    # Third subplot: Separatrix contours
    ax = axes[2]
    X, Y, separatrix_function = evaluate_on_grid(bistable_affine_separatrix_function, 
                                               x_limits=(-2, 2), y_limits=(-2, 2), resolution=200)
    ax.contourf(X, Y, np.log(separatrix_function+0.25), levels=10, cmap='Blues_r')
    base_cartoon_plot(ax, show_separatrix_label=False)
    remove_frame(ax)

    for ax in axs.flatten():
        ax.set_aspect('equal')
    # Save the figure
    fig.savefig('plots_for_publication/cartoon_subplots.png', bbox_inches='tight', dpi=300)
    fig.savefig('plots_for_publication/cartoon_subplots.pdf', bbox_inches='tight', dpi=300)