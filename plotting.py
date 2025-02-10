import torch
import numpy as np
import matplotlib.pyplot as plt


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
