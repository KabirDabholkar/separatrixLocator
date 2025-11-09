from config_utils import instantiate
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
from omegaconf_utils import omegaconf_resolvers
from learn_koopman_eig import train
from pathlib import Path
from functools import partial
import os
from compose import compose
from plotting import plot_model_contour,plot_kinetic_energy

from sklearn.decomposition import PCA
from copy import deepcopy

import matplotlib.pyplot as plt
import matplotlib as mpl
# mpl.use('Qt5Agg') # 'MacOSX') #)
import matplotlib.animation as animation
import seaborn as sns
mpl.rcParams['agg.path.chunksize'] = 10000

plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


from separatrixLocator import SeparatrixLocator

import torch
from torchdiffeq import odeint
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

PATH_TO_FIXED_POINT_FINDER = f'{os.getenv("PROJECT_PATH")}/fixed_point_finder'
import sys
sys.path.insert(0, PATH_TO_FIXED_POINT_FINDER)
from fixed_point_finder.FixedPointFinderTorch import FixedPointFinderTorch


CONFIG_PATH = "configs"
# CONFIG_NAME = "test"
CONFIG_NAME = "main"
# CONFIG_NAME = "main_2bitflipflop3D"
# CONFIG_NAME = "twolimitcycles"
# CONFIG_NAME = "bistable1D"
# CONFIG_NAME = "microbiome_GLV_11D"
# CONFIG_NAME = "finkelstein_fontolan"
# CONFIG_NAME = "main_1bitflipflop32D"
# CONFIG_NAME = "main_1bitflipflop64D"
# CONFIG_NAME = "main_1bitflipflop128D"
# CONFIG_NAME = "main_1bitflipflop256D"
# CONFIG_NAME = "main_1bitflipflop512D"
# CONFIG_NAME = "rslds.yaml"

project_path = os.getenv("PROJECT_PATH")


@hydra.main(version_base='1.3', config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def decorated_main(cfg):
    # return main(cfg)
    # main_multimodel(cfg)
    plot_ftle_2D(cfg)
    # return finkelstein_fontolan(cfg)
    # return finkelstein_fontolan_point_finder_test(cfg)
    # return finkelstein_fontolan_analysis_test(cfg)
    # return test_RNN(cfg)
    # return test_run_GD(cfg)
    # return plot_ODE_line_IC(cfg)
    # lowDapprox_test(cfg)
    # check_basin_of_attraction(cfg)
    # plot_cubichermitesampler(cfg)
    # return RNN_modify_inputs(cfg)
    # plot_dynamics_1D(cfg)
    # plot_dynamics_2D(cfg)
    # plot_dynamics(cfg)
    # plot_task_io(cfg)
    # plot_hermite_polynomials_2d(cfg)
    # plot_KEF_residuals(cfg)
    # plot_KEF_residual_heatmap(cfg)
    # fixed_point_analysis(cfg)
    # check_distribution(cfg)
    # plot_2D_dynamics_reduced_microbiome(cfg)
    # find_saddle_point(cfg)
    # plot_dynamics_3D(cfg)
    # RNN_fixedpoints(cfg)
    # flipflop_separatrix_points(cfg)
    # flipflop2Dsliceof3D(cfg)
    
def plot_ftle_2D(cfg):
    omegaconf_resolvers()
    cfg.savepath = os.path.join(project_path, cfg.savepath)
    dynamics_function = instantiate(cfg.dynamics.function)

    from ftle_computer import FTLEComputer
    from plotting import evaluate_on_grid, plot_flow_streamlines, plot_kinetic_energy
    
    ftle_comp = FTLEComputer(lambda t,x: dynamics_function(x), device="cpu", method="dopri5")

    t_span = torch.linspace(0, 5.0, 2)
    resolution = 200

    ftle_function = lambda x: ftle_comp.compute_ftle(x, t_span)
    X, Y, ftle_vals = evaluate_on_grid(ftle_function,
                                                 x_limits=cfg.dynamics.lims.x, y_limits=cfg.dynamics.lims.y, resolution=resolution)
    
    
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(6, 3))
    
    # Plot flow streamlines on left subplot
    ax = axs[0]
    plot_flow_streamlines(dynamics_function, ax, x_limits=cfg.dynamics.lims.x, y_limits=cfg.dynamics.lims.y,
                         resolution=resolution, density=0.5, color='black', linewidth=0.5, alpha=0.4)
    plot_kinetic_energy(dynamics_function, ax, x_limits=cfg.dynamics.lims.x, y_limits=cfg.dynamics.lims.y, heatmap_resolution=resolution)
    ax.set_title('Flow')
    
    # Plot FTLE on right subplot
    ax = axs[1]
    im = ax.imshow(ftle_vals, extent=cfg.dynamics.lims.x + cfg.dynamics.lims.y, origin='lower', cmap='Blues_r')
    plot_flow_streamlines(dynamics_function, ax, x_limits=cfg.dynamics.lims.x, y_limits=cfg.dynamics.lims.y,
                          resolution=resolution, density=0.5, color='black', linewidth=0.5, alpha=0.4)
    ax.set_title('FTLE')

    # Add colorbar outside plots
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    for ax in axs:
        ax.set_aspect('equal')

    # Create directories if they don't exist
    Path(cfg.savepath).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(cfg.savepath) / f'ftle_2D_{cfg.dynamics.name}.pdf')
    fig.savefig(Path(cfg.savepath) / f'ftle_2D_{cfg.dynamics.name}.png',dpi=200)
    
    
    
    
    
    

def plot_dynamics_1D(cfg):
    omegaconf_resolvers()
    cfg.savepath = os.path.join(project_path, cfg.savepath)
    dynamics_function = instantiate(cfg.dynamics.function)


    # Load separatrix locator model
    SL = instantiate(cfg.separatrix_locator)
    SL.to('cpu')
    SL.models = [instantiate(cfg.model).to(SL.device) for _ in range(SL.num_models)]

    # Set up load path
    new_format_path = Path(cfg.savepath) / cfg.experiment_details
    if os.path.exists(new_format_path):
        load_path = new_format_path
    else:
        load_path = Path(cfg.savepath)
        print(new_format_path, 'does not exist, loading', load_path, 'instead.')
    SL.load_models(load_path)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(7,2.5))

    # Create x values to evaluate dynamics over
    x = torch.linspace(-2, 2, 200).reshape(-1, 1)

    # Evaluate dynamics
    dx = dynamics_function(x)

    # Convert to numpy for plotting
    x_np = x.detach().numpy()
    dx_np = dx.detach().numpy()

    # Plot 1: Dynamics
    ax1.plot(x_np, dx_np, 'b-', linewidth=2)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    ax1.set_title(r'$\frac{dx}{dt}$')
    ax1.set_ylim(-1.1, 1.1)

    # Plot 2: Kinetic Energy
    kinetic_energy = dx**2  # Kinetic energy is velocity squared
    ax2.plot(x_np, kinetic_energy.detach().numpy(), 'r-', linewidth=2)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    ax2.set_title(r'$q(x)$')
    ax2.set_ylim(-0.1,0.4)

    # Plot 3: Separatrix Locator Prediction
    sl_pred = SL.predict(x)
    sl_pred = sl_pred/(sl_pred**2).mean()**0.5
    ana_func = lambda x:x/((x**2-1)**2)**0.25
    # instantiate(cfg.dynamics.analytical_eigenfunction)
    ana_pred = ana_func(x)
    ana_pred = ana_pred/(ana_pred**2).mean()**0.5
    ax3.plot(x_np, sl_pred.detach().numpy(), 'g-', linewidth=2, label='DNN')
    ax3.plot(x_np,ana_pred,c='k',ls='dashed',label='Analytical')


    ax3.set_title(r'$\psi(x)$')
    ax3.set_ylim(-3,3)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax3.legend()

    from plotting import remove_frame
    for ax in (ax1,ax2,ax3):
        remove_frame(ax,['right','bottom','top'])

    for ax in (ax1,ax2,ax3):
        if hasattr(cfg.dynamics,'plot_fixed_points'):
            pointsets = instantiate(cfg.dynamics.plot_fixed_points)
            # Check if fixed point data exists and plot it
            fixed_point_path = Path(cfg.savepath) / 'fixed_point_data.csv'
            if fixed_point_path.exists():
                fixed_points = pd.read_csv(fixed_point_path)
                fixed_points = fixed_points.loc[fixed_points['q']<1e-9]
                for pointset in pointsets:
                    if 'unstable' in pointset['label']:
                        stability = False
                    else:
                        stability = True
                    pointset['x'] = fixed_points.loc[fixed_points['stability']==stability]['x0']
                    pointset['y'] = fixed_points.loc[fixed_points['stability']==stability]['x1']
            for pointset in pointsets:
                ax.scatter(**pointset,c='lightgreen')

    plt.tight_layout()

    # Save figure
    fig.savefig(Path(cfg.savepath) / cfg.experiment_details / f'dynamics1D_{cfg.dynamics.name}.pdf')
    plt.show()














def flipflop_separatrix_points(cfg):
    omegaconf_resolvers()
    cfg.savepath = os.path.join(project_path, cfg.savepath)
    dynamics_function = instantiate(cfg.dynamics.function)
    # Create figure with 3 subplots
    distribution = instantiate(cfg.dynamics.IC_distribution)

    # Sample random initial conditions
    num_trajectories = 100
    dim = cfg.dynamics.dim
    ic_range = torch.tensor([0.0, 1.0])  # Range for initial conditions
    initial_conditions = distribution.sample(sample_shape=(num_trajectories,))

    # Time settings for integration
    t_span = torch.linspace(0, 1000, 2) #500)

    # Run trajectories
    trajectories = odeint(
        lambda t, x: dynamics_function(x),
        initial_conditions,
        t_span,
        # method='r'
    )

    # Convert trajectories to numpy for PCA
    trajectories_np = trajectories.detach().cpu().numpy()

    from sklearn.cluster import KMeans

    # Cluster the final points (shape: num_trajectories x dim)
    kmeans4 = KMeans(n_clusters=4, random_state=42)
    cluster_labels4 = kmeans4.fit_predict(trajectories_np[-1])
    cluster_centers = kmeans4.cluster_centers_  # shape: (4, dim)

    # Get the ordered centers
    from square import order_square_nd
    ordered_centers = order_square_nd(cluster_centers[np.random.permutation(len(cluster_centers))])
    
    # Convert edge points from numpy to torch tensors
    edge_points = {
        'vertical1': (torch.tensor(ordered_centers[0], dtype=torch.float32),
                     torch.tensor(ordered_centers[1], dtype=torch.float32)),
        'vertical2': (torch.tensor(ordered_centers[2], dtype=torch.float32),
                     torch.tensor(ordered_centers[3], dtype=torch.float32)),
        'horizontal1': (torch.tensor(ordered_centers[0], dtype=torch.float32),
                       torch.tensor(ordered_centers[3], dtype=torch.float32)),
        'horizontal2': (torch.tensor(ordered_centers[1], dtype=torch.float32),
                       torch.tensor(ordered_centers[2], dtype=torch.float32))
    }

    from separatrix_point_finder import find_separatrix_point_along_line
    # Find separatrix points along each edge
    separatrix_points = {}
    for edge_name, edge_endpoints in edge_points.items():
        separatrix_points[edge_name] = find_separatrix_point_along_line(
            dynamics_function=dynamics_function,
            external_input=None,
            attractors=edge_endpoints,
            num_points=20,
            num_iterations=2,
            time_points=t_span,
            final_time=1000
        )
    print(separatrix_points)
    print(edge_points)

    # Create a 3D scatter plot of separatrix points

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot separatrix points for each edge in different colors
    colors = ['r', 'g', 'b', 'y']
    for (edge_name, sep_point), color in zip(separatrix_points.items(), colors):
        # Convert separatrix point to numpy array
        point = sep_point.detach().cpu().numpy()
        ax.scatter(point[0], point[1], point[2], c=color, label=edge_name, s=100)

    # Plot edge endpoints and connecting lines
    for (edge_name, (point1, point2)), color in zip(edge_points.items(), colors):
        p1 = point1.detach().cpu().numpy()
        p2 = point2.detach().cpu().numpy()
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c=color, linewidth=2)
        ax.scatter(p1[0], p1[1], p1[2], c=color, marker='*', s=200)
        ax.scatter(p2[0], p2[1], p2[2], c=color, marker='*', s=200)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.view_init(elev=40, azim=150)

    # Save the figure
    plt.savefig(Path(cfg.savepath) / 'separatrix_points_3d.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Create directory for saving points
    save_dir = Path(cfg.savepath) / 'saved_points'
    save_dir.mkdir(exist_ok=True)

    # # Save each separatrix point as a .pt file
    # for edge_name, sep_point in separatrix_points.items():
    #     torch.save(sep_point, save_dir / f'{edge_name}.pt')

    # Save each cluster center as a .pt file
    for i, center in enumerate(ordered_centers):
        torch.save(torch.tensor(center,dtype=torch.float32), save_dir / f'vertex{i}.pt')
        

def load_models_with_different_speeds(SL,new_format_path):
    # state_dict = torch.load(new_format_path / 'models' / 'trained_on_vertical_edges_speed5.pt', map_location='cpu')
    # SL.models[0].load_state_dict(state_dict)
    # state_dict = torch.load(new_format_path / 'models' / 'trained_on_horizontal_edges_speed5.pt', map_location='cpu')
    # SL.models[1].load_state_dict(state_dict)
    # state_dict = torch.load(new_format_path / 'models' / 'trained_on_vertical_edges_speed10.pt', map_location='cpu')
    # SL.models[2].load_state_dict(state_dict)
    # state_dict = torch.load(new_format_path / 'models' / 'trained_on_horizontal_edges_speed10.pt', map_location='cpu')
    # SL.models[3].load_state_dict(state_dict)
    # state_dict = torch.load(new_format_path / 'models' / 'trained_on_vertical_edges_speed20.pt', map_location='cpu')
    # SL.models[4].load_state_dict(state_dict)
    # state_dict = torch.load(new_format_path / 'models' / 'trained_on_horizontal_edges_speed20.pt', map_location='cpu')
    # SL.models[5].load_state_dict(state_dict)

    # state_dict = torch.load(new_format_path / 'models' / 'trained_on_vertical_edges_speed20.pt', map_location='cpu')
    # SL.models[0].load_state_dict(state_dict)
    # state_dict = torch.load(new_format_path / 'models' / 'trained_on_horizontal_edges_speed20.pt', map_location='cpu')
    # SL.models[1].load_state_dict(state_dict)

    state_dict = torch.load(new_format_path / 'models' / 'trained_on_vertical_edges_speed10.pt', map_location='cpu')
    SL.models[0].load_state_dict(state_dict)
    state_dict = torch.load(new_format_path / 'models' / 'trained_on_horizontal_edges_speed10.pt', map_location='cpu')
    SL.models[1].load_state_dict(state_dict)
    return SL


def flipflop2Dsliceof3D(cfg):
    omegaconf_resolvers()
    cfg.savepath = os.path.join(project_path, cfg.savepath)
    dynamics_function = instantiate(cfg.dynamics.function)
    # distribution_fit = instantiate(cfg.dynamics.IC_distribution_fit)
    # distribution = instantiate(cfg.dynamics.IC_distribution)
    from plotting import remove_frame, evaluate_on_grid, plot_flow_streamlines

    # Load saved points
    save_dir = Path(cfg.savepath) / 'saved_points'
    vertex_points = []
    other_points = []
    vertex_filenames = []
    other_filenames = []

    # Load all saved points
    for file in sorted(save_dir.glob("*"), key=lambda x: int(x.stem[-1]) if x.stem[-1].isdigit() else float('inf')):
        point = torch.load(file,weights_only=True)
        if file.stem.startswith('vertex'):
            vertex_points.append(point.cpu().numpy())
            vertex_filenames.append(file.stem)
        else:
            other_points.append(point.cpu().numpy())
            other_filenames.append(file.stem)

    # Combine all points for PCA
    all_points = vertex_points + other_points
    saved_points_st = np.stack(all_points)
    pca = PCA().fit(saved_points_st)

    # Load separatrix locator model
    SL = instantiate(cfg.separatrix_locator)
    SL.to('cpu')
    SL.num_models = 2
    SL.models = [instantiate(cfg.model).to(SL.device) for _ in range(SL.num_models)]

    # Set up load path
    new_format_path = Path(cfg.savepath) / cfg.experiment_details
    if os.path.exists(new_format_path):
        load_path = new_format_path
    else:
        load_path = Path(cfg.savepath)
        print(new_format_path, 'does not exist, loading', load_path, 'instead.')
    # SL.load_models(load_path)
    # state_dict = torch.load(new_format_path / 'models' / 'AdditiveModel_5.pt', map_location='cpu')

    SL = load_models_with_different_speeds(SL, new_format_path)

    def KEF2D(x):
        x_padded = torch.cat([x, torch.zeros_like(x[...,:1])], dim=-1)
        # Convert tensor to numpy for PCA inverse transform
        x_padded_np = x_padded.detach().cpu().numpy()

        # Transform back to original space using loaded PCA
        x_orig = pca.inverse_transform(x_padded_np)

        # Convert back to tensor
        x_orig = torch.tensor(x_orig, dtype=torch.float32, device=x_padded.device)
        pred = SL.predict(x_orig)
        return pred

    x_limits = (-1.1,1.1)
    y_limits = (-1.1,1.1)

    X,Y,KEFvalues_on_grid = evaluate_on_grid(KEF2D,x_limits=x_limits,y_limits=y_limits,resolution=100)


    #######
    def dynamics2D(x,full_dim=3):
        # Pad input with zeros to match full dimension
        padded_x = torch.zeros(*x.shape[:-1], full_dim, device=x.device)
        padded_x[..., :2] = x  # Copy the 2D input into first 2 dimensions
        padded_x = padded_x.detach().numpy()
        x = torch.tensor(pca.inverse_transform(padded_x))
        dyn_vals = dynamics_function(x)
        dyn_vals = dyn_vals.detach().numpy()
        return torch.tensor(pca.transform(dyn_vals)[:,:2])


    ######


    tspan = torch.linspace(0,10,100)
    # base_point = np.array([-0.7,-0.4])
    base_point = np.array([0.0, -0.3])

    target = pca.transform(vertex_points[3:])[0,:2]
    pert1 = np.array([0.0, 0.4])
    perturbations = {
        'pert1': pert1,
        # 'pert2': np.array([0.5 * np.sin(np.pi/4), 0.5 * np.cos(np.pi/4)])  # 45 degree rotation
        'pert2': np.array(np.linalg.norm(pert1) * (target - base_point) / np.linalg.norm(target - base_point))  # in direction of target
    }
    trajectories = {}
    for name, perturbation in perturbations.items():
        endpoint = base_point + perturbation
        orig_endpoint = pca.inverse_transform(np.append(endpoint, 0))
        traj = odeint(
            lambda t,x: dynamics_function(x), torch.tensor(orig_endpoint,dtype=torch.float32), tspan
        )
        trajectories[name] = pca.transform(traj.detach().cpu().numpy())[:,:2]

    ######

    # Create scatter plot of PCA-transformed points
    fig, ax = plt.subplots(figsize=(4, 4))

    # Transform saved points using PCA
    transformed_points = pca.transform(vertex_points)

    # Plot first two principal components
    scatter = ax.scatter(transformed_points[:, 0], transformed_points[:, 1])

    ####
    ax.scatter(*base_point, c='r', marker='*')
    # Plot arrows and trajectories
    for name, perturbation in perturbations.items():
        ax.arrow(base_point[0], base_point[1],
                perturbation[0], perturbation[1],
                head_width=0.05, head_length=0.1, fc='r', ec='r', length_includes_head=True)
        ax.plot(*trajectories[name].T, 'r-', alpha=0.7, linewidth=1)
    # Plot dashed line from base point to target
    ax.plot([base_point[0], target[0]], [base_point[1], target[1]], 'k--', alpha=0.5, lw=0.7)
    # Add circle segment between perturbation endpoints
    arc = plt.matplotlib.patches.Arc(
        xy=base_point, 
        width=2*np.linalg.norm(pert1), 
        height=2*np.linalg.norm(pert1),
        theta1=np.degrees(np.arctan2(perturbations['pert2'][1], perturbations['pert2'][0])),
        theta2=np.degrees(np.arctan2(pert1[1], pert1[0])),
        linestyle=':',
        color='red',
        alpha=0.5,
        lw=1
    )
    ax.add_patch(arc)
    ####
    
    # Add point labels
    for i, txt in enumerate(vertex_filenames):
        ax.annotate(txt, (transformed_points[i, 0], transformed_points[i, 1]), 
                   xytext=(5, 5), textcoords='offset points', 
                   fontsize=8, ha='left', va='bottom')


    for i in range(KEFvalues_on_grid.shape[-1]):
        ax.contour(X,
               Y,
               KEFvalues_on_grid[...,i],
               levels=[0], colors='lightgreen')


    # plot_flow_streamlines(dynamics2D,ax,x_limits=x_limits,y_limits=y_limits,resolution=50,density=0.3,alpha=0.3)

    # Equal aspect ratio for better visualization
    ax.set_aspect('equal')
    remove_frame(ax)

    # Save figure
    plt.savefig(Path(cfg.savepath) / cfg.experiment_details / "perturbation_cartoon.png", dpi=200)
    plt.savefig(Path(cfg.savepath) / cfg.experiment_details / "perturbation_cartoon.pdf", dpi=200)
    plt.show()
    plt.close()





def plot_dynamics_3D(cfg):
    omegaconf_resolvers()
    cfg.savepath = os.path.join(project_path, cfg.savepath)
    dynamics_function = instantiate(cfg.dynamics.function)
    distribution_fit = instantiate(cfg.dynamics.IC_distribution_fit)
    distribution = instantiate(cfg.dynamics.IC_distribution)
    from plotting import remove_frame

    # Sample random initial conditions
    num_trajectories = 100
    dim = cfg.dynamics.dim
    ic_range = torch.tensor([0.0, 1.0])  # Range for initial conditions
    initial_conditions = distribution.sample(sample_shape=(num_trajectories,))

    # Time settings for integration
    t_span = torch.linspace(0, 1000, 500)

    # Run trajectories
    trajectories = odeint(
        lambda t, x: dynamics_function(x),
        initial_conditions,
        t_span,
        # method='rk'
    )

    # Convert trajectories to numpy for PCA
    trajectories_np = trajectories.detach().cpu().numpy()

    # Reshape for PCA: combine all trajectories and time points
    traj_shape = trajectories_np.shape
    reshaped_data_end = trajectories_np[-100:].reshape(-1, traj_shape[-1])
    reshaped_data = trajectories_np.reshape(-1, traj_shape[-1])
    # Apply PCA to find principal components
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    pca.fit(reshaped_data_end)

    # Transform the data
    transformed_data = pca.transform(reshaped_data)

    # Reshape back to original shape but with reduced dimensions
    transformed_trajectories = transformed_data.reshape(traj_shape[0], traj_shape[1], 3)

    #####

    # Load separatrix locator model
    SL = instantiate(cfg.separatrix_locator)
    SL.to('cpu')
    SL.num_models = 2
    SL.models = [instantiate(cfg.model).to(SL.device) for _ in range(SL.num_models)]

    # Set up load path
    new_format_path = Path(cfg.savepath) / cfg.experiment_details
    if os.path.exists(new_format_path):
        load_path = new_format_path
    else:
        load_path = Path(cfg.savepath)
        print(new_format_path, 'does not exist, loading', load_path, 'instead.')
    # SL.load_models(load_path)
    # state_dict = torch.load(new_format_path / 'models' / 'AdditiveModel_5.pt', map_location='cpu')


    SL = load_models_with_different_speeds(SL, new_format_path)
    cfg.separatrix_locator_score_kwargs['eigenvalue'] = 1 #0.5 #1.0 #0.5
    scores = SL.score(
        dynamics_function,
        distribution_fit,
        external_input_dist = None,
        ** instantiate(cfg.separatrix_locator_score_kwargs)
    )
    print(scores)
    # SL.models = SL.models[:9]
    # SL.num_models = 9
    #####

    # Create a 2D grid in the first two principal components, centered at the mean
    n_grid = 100  # Grid resolution
    
    # Calculate the mean of the transformed trajectories
    mean_pc1 = transformed_trajectories[..., 0].mean()
    mean_pc2 = transformed_trajectories[..., 1].mean()
    
    # Calculate the range to determine grid size
    pc1_range = transformed_trajectories[..., 0].max() - transformed_trajectories[..., 0].min()
    pc2_range = transformed_trajectories[..., 1].max() - transformed_trajectories[..., 1].min()
    
    # Add padding and center around mean
    padding = -0.2
    half_width1 = (1 + padding) * pc1_range / 2
    half_width2 = (1 + padding) * pc2_range / 2
    # Create grid centered at mean and save limits
    xlims = (mean_pc1 - half_width1, mean_pc1 + half_width1)
    ylims = (mean_pc2 - half_width2, mean_pc2 + half_width2)
    pc1 = np.linspace(xlims[0], xlims[1], n_grid)
    pc2 = np.linspace(ylims[0], ylims[1], n_grid)
    PC1, PC2 = np.meshgrid(pc1, pc2)

    # Flatten the grid and add zeros for the third PC
    grid_points_pca = np.stack([PC1.ravel(), PC2.ravel(), np.zeros_like(PC1.ravel())], axis=-1)

    # Inverse transform from PCA space back to original coordinates
    grid_points_orig = pca.inverse_transform(grid_points_pca)  # shape (n_grid*n_grid, dim)

    # Convert to torch tensor for model prediction
    grid_points_torch = torch.from_numpy(grid_points_orig).float().to(SL.device)

    KEFvalues_on_grid = SL.predict(grid_points_torch)[:,0::cfg.model.output_size]
    print('KEFvalues_on_grid.shape',KEFvalues_on_grid.shape)

    fig,axs = plt.subplots(1,2,sharex=True,sharey=True)
    for i,ax in enumerate(axs.flatten()):
        if i<20:
            ax.contourf(PC1.reshape(n_grid,n_grid),
                       PC2.reshape(n_grid,n_grid),
                       np.abs(KEFvalues_on_grid[:,i]).reshape(n_grid,n_grid),levels=15,cmap='Blues_r')
            ax.contour(PC1.reshape(n_grid,n_grid),
                      PC2.reshape(n_grid,n_grid),
                      KEFvalues_on_grid[:,i].reshape(n_grid,n_grid),
                      levels=[0], colors='lightgreen')
            # Plot PCA of the 4 points
            # ax.scatter(transformed_trajectories[-1,:,0], transformed_trajectories[-1,:,1],
            #           c='red', s=10, alpha=0.1)
            ax.set_aspect('equal')
    
    # Load and plot saved points
    saved_points_dir = Path(cfg.savepath) / "saved_points"
    saved_points = []
    
    # Load only vertex points
    for file in saved_points_dir.glob("vertex*"):
        if file.stem[len("vertex"):].isdigit():  # Check if suffix is an integer
            point = torch.load(file)
            saved_points.append(point.cpu().numpy())
    
    if saved_points:  # Only process if points were found
        # Convert to array and transform to PCA space
        saved_points_array = np.array(saved_points)
        transformed_points = pca.transform(saved_points_array)
        
        # Plot on both axes
        for ax in axs.flatten():
            # Scatter plot the transformed points
            ax.scatter(transformed_points[:, 0], transformed_points[:, 1], 
                      c='lightgreen', marker='o', s=100)
    def dynamics2D(x,full_dim=3):
        # Pad input with zeros to match full dimension
        padded_x = torch.zeros(*x.shape[:-1], full_dim, device=x.device)
        padded_x[..., :2] = x  # Copy the 2D input into first 2 dimensions
        padded_x = padded_x.detach().numpy()
        x = torch.tensor(pca.inverse_transform(padded_x))
        dyn_vals = dynamics_function(x)
        dyn_vals = dyn_vals.detach().numpy()
        return torch.tensor(pca.transform(dyn_vals)[:,:2])

    from plotting import plot_flow_streamlines
    for ax in axs.flatten():
        plot_flow_streamlines(dynamics2D,ax,x_limits=xlims,y_limits=ylims,resolution=100,density=0.4,alpha=0.5,color='red')
        remove_frame(ax)
    fig.tight_layout()
    fig.savefig(Path(cfg.savepath) / cfg.experiment_details / 'KEFs_2Dslice.png',dpi=300)
    fig.savefig(Path(cfg.savepath) / cfg.experiment_details / 'KEFs_2Dslice.pdf')
    plt.show()

    #####

    # Create a grid in PCA space and evaluate SL.predict at the inverse-PCA-mapped points
    # We'll plot the surface where SL.predict(X) == 0

    # Define the grid in PCA space
    n_grid = 15 #60  # You can adjust this for resolution
    # Calculate the mean from PCA object and transform it to PCA space
    mean_original = pca.mean_
    mean_pca = pca.transform(mean_original.reshape(1, -1))[0]
    mean_pc1 = mean_pca[0]
    mean_pc2 = mean_pca[1]
    mean_pc3 = mean_pca[2]
    
    # Calculate the range to determine grid size
    pc1_range = transformed_trajectories[..., 0].max() - transformed_trajectories[..., 0].min()
    pc2_range = transformed_trajectories[..., 1].max() - transformed_trajectories[..., 1].min() 
    pc3_range = transformed_trajectories[..., 2].max() - transformed_trajectories[..., 2].min()
    
    # Add padding and center around mean
    padding = -0.2
    half_width1 = (1 + padding) * pc1_range / 2
    half_width2 = (1 + padding) * pc2_range / 2
    half_width3 = (1 + padding) * pc3_range / 2
    
    # Create grid centered at mean
    pc1 = np.linspace(mean_pc1 - half_width1, mean_pc1 + half_width1, n_grid)
    pc2 = np.linspace(mean_pc2 - half_width2, mean_pc2 + half_width2, n_grid)
    pc3 = np.linspace(mean_pc3 - half_width3, mean_pc3 + half_width3, n_grid)
    PC1, PC2, PC3 = np.meshgrid(pc1, pc2, pc3, indexing='ij')

    # Flatten the meshgrid to a list of points (N, 3)
    grid_points_pca = np.stack([PC1.ravel(), PC2.ravel(), PC3.ravel()], axis=-1)  # shape (n_grid**3, 3)

    # Inverse transform to original state space
    grid_points_orig = pca.inverse_transform(grid_points_pca)  # shape (n_grid**3, dim)

    # Convert to torch tensor for SL.predict
    grid_points_orig_torch = torch.from_numpy(grid_points_orig).float().to(SL.device)

    # Evaluate SL.predict in batches to avoid memory issues
    batch_size = 4096
    preds = []
    with torch.no_grad():
        for i in range(0, grid_points_orig_torch.shape[0], batch_size):
            batch = grid_points_orig_torch[i:i+batch_size]
            pred = SL.predict(batch)
            preds.append(pred.cpu().numpy())
    preds = np.concatenate(preds, axis=0)  # shape (n_grid**3,)

    # Plot in 3D using PCA-transformed data
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create two meshes for the two predictions
    for pred_idx in range(2):
        preds_reshaped = preds[...,pred_idx]

        # Reshape predictions to grid shape
        preds_grid = preds_reshaped.reshape(PC1.shape)

        # Plot the zero level set using matplotlib's contour3D
        from skimage import measure

        # The marching cubes algorithm expects the grid in (z, y, x) order, so we transpose
        verts, faces, normals, values = measure.marching_cubes(preds_grid, level=0, spacing=(
            pc1[1] - pc1[0], pc2[1] - pc2[0], pc3[1] - pc3[0]
        ))

        # Create mesh with different colors for each prediction
        mesh = Poly3DCollection(verts[faces], alpha=0.2, 
                              facecolor='cyan' if pred_idx == 0 else 'magenta', 
                              edgecolor='none')
        ax.add_collection3d(mesh)

    # SL.prepare_models_for_gradient_descent(distribution)
    #
    # separatrix_data = SL.find_separatrix(distribution,**instantiate(cfg.separatrix_find_separatrix_kwargs))
    # all_points = np.concatenate(separatrix_data[1])
    # all_points_pca = pca.transform(all_points)

    #####

    # Find points with lowest 3 percentile of predictions for each model
    percentile_3 = np.percentile(np.abs(preds), 3, axis=0)
    low_pred_mask = np.abs(preds) < percentile_3

    # Get the PCA coordinates of these points
    grid_points_pca_reshaped = grid_points_pca.reshape(-1, 3)

    # Plot points for each model prediction
    for pred_idx in range(2):
        low_pred_points = grid_points_pca_reshaped[low_pred_mask[:, pred_idx]]
        ax.scatter(
            low_pred_points[:, 0],
            low_pred_points[:, 1],
            low_pred_points[:, 2],
            c='blue' if pred_idx == 0 else 'green',
            s=10,
            alpha=0.5,
            label=f'Model {pred_idx+1} low KEF'
        )

    t = -450
    # for traj in transformed_trajectories.transpose(1, 0, 2):  # Each trajectory
        # ax.plot(
        #     traj[t:, 0],
        #     traj[t:, 1],
        #     traj[t:, 2],
        #     alpha=0.6
        # )
        # Plot final points
    ax.scatter(
        transformed_trajectories[-1,:, 0],
        transformed_trajectories[-1,:, 1],
        transformed_trajectories[-1,:, 2],
        c='red',
        s=50
    )
    # ax.add_collection3d(mesh)
    # ax.scatter(*all_points_pca.T,c='blue')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2f})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2f})')
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2f})')

    # Change the 3D view angle to be more from above (e.g., elev=30, azim=60)
    # ax.view_init(elev=10, azim=60)
    ax.view_init(elev=90, azim=90)
    ax.legend()

    # plt.title('PCA-transformed Trajectories')
    fig.savefig(Path(cfg.savepath) / cfg.experiment_details / "3d_trajectories_pca.png")
    fig.savefig(Path(cfg.savepath) / cfg.experiment_details / "3d_trajectories_pca.pdf")

    plt.show()

    # ##### Function to rotate the plot ######
    # def rotate(angle):
    #     ax.view_init(elev=30, azim=angle)
    #
    # # Create animation
    # num_frames = 360  # Number of frames for a full rotation
    # rotation_animation = animation.FuncAnimation(fig, rotate, frames=num_frames, interval=1000 / 30)
    #
    # # Save the animation to a file
    # rotation_animation.save(Path(cfg.savepath) / 'PCA_3d_rotation.mp4', writer='ffmpeg', fps=30, dpi=100)

def RNN_fixedpoints(cfg):
    omegaconf_resolvers()
    cfg.savepath = os.path.join(project_path, cfg.savepath)
    dynamics_function = instantiate(cfg.dynamics.function)

    rnn_model = instantiate(cfg.dynamics.loaded_RNN_model)  # (device=SL.device)
    # cfg.dynamics.RNN_dataset.batch_size = 5000
    # cfg.dynamics.RNN_dataset.n_trials = 1000
    dataset = instantiate(cfg.dynamics.RNN_dataset)
    inp, targ = dataset()

    torch_inp = torch.from_numpy(inp).type(torch.float)  # .to(device)
    outputs, hidden_traj = rnn_model(torch_inp, return_hidden=True)  # , deterministic=False)
    outputs, hidden_traj = outputs.detach().cpu().numpy(), hidden_traj.detach().cpu().numpy()

    cfg.fpf_hps['max_iters'] = 20000

    FPF = FixedPointFinderTorch(
        rnn_model.rnn if hasattr(rnn_model, "rnn") else rnn_model,
        **instantiate(cfg.fpf_hps)
    )
    num_trials = 500
    # initial_conditions = dist.sample(sample_shape=(num_trials,)).detach().cpu().numpy()
    # inputs = np.zeros((1, cfg.dynamics.RNN_model.act_size))
    # inputs[...,2] = 1.0
    # torch_inp[..., :2] = 0.0
    torch_inp = torch_inp * 0
    fp_inputs = torch_inp.reshape(-1, torch_inp.shape[-1]).detach().cpu().numpy()

    # inputs[...,0] = 1
    initial_conditions = hidden_traj.reshape(-1, hidden_traj.shape[-1])
    select = np.random.choice(initial_conditions.shape[0], size=num_trials, replace=False)
    initial_conditions = initial_conditions[select]
    fp_inputs = fp_inputs[select]
    # fp_inputs[:,:2] = 0
    initial_conditions += np.random.normal(size=initial_conditions.shape) * 2.0  # 0.5 #2.0
    # print('initial_conditions', initial_conditions.shape)
    unique_fps, all_fps = FPF.find_fixed_points(
        deepcopy(initial_conditions),
        fp_inputs
    )


    fixed_point_data = {
        'stability': unique_fps.is_stable,
        'q': unique_fps.qstar,
        'x0': unique_fps.xstar[..., 0],
        'x1': unique_fps.xstar[..., 1],
        'x2': unique_fps.xstar[..., 2],
    }
    # print(fixed_point_data)
    fixed_point_data = pd.DataFrame(fixed_point_data)
    fixed_point_data.to_csv(Path(cfg.savepath) / 'fixed_point_data.csv', index=False)


def find_saddle_point(cfg):
    omegaconf_resolvers()
    cfg.savepath = os.path.join(project_path, cfg.savepath)
    dynamics_function = instantiate(cfg.dynamics.function)
    point_on_separatrix = instantiate(cfg.dynamics.point_on_separatrix)
    # saddle_point = instantiate(cfg.dynamics.saddle_point)
    from odeint_utils import run_odeint_to_final
    from plotting import dynamics_to_kinetic_energy

    # iid_gammas = instantiate(cfg.dynamics.iid_gammas)

    # from separatrix_point_finder import find_saddle_point
    # saddle_point, eigenvalues, trajectory, ke_traj = find_saddle_point(dynamics_function, point_on_separatrix, T=1000, return_all=True)
    # saddle_point[saddle_point<0] = 0
    #
    # Save saddle point to file
    # save_path = Path(cfg.savepath) / 'saddle_point.pt'
    # torch.save(saddle_point, save_path)
    #
    # save_path = Path(cfg.savepath) / 'point_on_separatrix.pt'
    # torch.save(point_on_separatrix, save_path)
    # #
    # kinetic_energy_function = dynamics_to_kinetic_energy(dynamics_function)
    # with torch.no_grad():
    #     print(kinetic_energy_function(saddle_point))
    # #
    # # # Run trajectory starting from point on separatrix
    # # T = 1.0
    # # trajectory = run_odeint_to_final(dynamics_function, point_on_separatrix, T=T,steps=100, return_last_only=False)
    # # ke_traj = kinetic_energy_function(trajectory)
    # #
    # plt.figure()
    # plt.plot(ke_traj)
    # plt.yscale('log')
    # plt.show()
    #
    # # # Find point with minimum kinetic energy along trajectory
    # # min_ke_idx = torch.argmin(ke_traj)
    # # saddle_point = trajectory[min_ke_idx]
    # #
    # # Compute Jacobian at saddle point using autograd
    # x = saddle_point.clone().detach().requires_grad_(True)
    # y = dynamics_function(x)
    # jacobian = torch.autograd.functional.jacobian(dynamics_function, x)
    # #
    # # Compute eigenvalues
    # eigenvalues = torch.linalg.eigvals(jacobian)
    #
    # print("Saddle point:", saddle_point)
    # print("Eigenvalues at saddle point:", eigenvalues)
    #
    # plt.figure()
    # plt.scatter(eigenvalues.real,eigenvalues.imag)
    # plt.show()
    #
    # # # The final point should be near a saddle point
    # # # saddle_point = trajectory[-1]
    # # # return saddle_point
    # #
    #
    # from ssr_module.steady_state_reduction_example import get_all_stein_steady_states, ssrParams, Params
    # p = Params()
    # ssr_steady_states = get_all_stein_steady_states(p)
    # s = ssrParams(p, ssr_steady_states['E'], ssr_steady_states['C'])
    #
    # # Sample initial conditions around saddle point with Gaussian noise
    # num_samples = 100
    # noise_std = 0.1
    # noise = torch.randn(num_samples, saddle_point.shape[0]) * noise_std
    # initial_conditions = point_on_separatrix + noise[:,:2] @ torch.tensor(np.stack([s.ssa, s.ssb]),dtype=torch.float32)
    #
    # # Ensure all values are non-negative
    # initial_conditions = torch.clamp(initial_conditions, min=0.0)
    #
    # # Run trajectories from these initial conditions
    # T = 10.0
    # trajectories = run_odeint_to_final(dynamics_function, initial_conditions, T=T, steps=1000, return_last_only=False)
    #
    # # Reshape trajectories for PCA
    # traj_shape = trajectories.shape
    # X = trajectories.reshape(-1, traj_shape[-1])
    #
    # # Perform k-means clustering on final points of trajectories
    # from sklearn.cluster import KMeans
    # kmeans = KMeans(n_clusters=2, random_state=0)
    # final_points = trajectories[-1]
    # cluster_labels = kmeans.fit_predict(final_points)
    #
    # # Perform PCA
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=2)
    # # X_pca = pca.fit_transform(X.detach().numpy())
    # X_pca = p.project_to_2D(X,ssa=s.ssa,ssb=s.ssb)
    #
    # # Reshape back to trajectory form
    # X_pca = X_pca.reshape(traj_shape[0], traj_shape[1], 2)
    #
    # # Plot trajectories in PCA space
    # plt.figure(figsize=(5, 4))
    # for i in range(num_samples):
    #     plt.plot(X_pca[:, i, 0], X_pca[:, i, 1], alpha=0.3, c=f'C{cluster_labels[i]}')
    #
    # # Plot saddle point in PCA space
    # # saddle_pca = pca.transform(saddle_point.detach().numpy().reshape(1, -1))
    # saddle_pca = p.project_to_2D(saddle_point[None].detach(),ssa=s.ssa,ssb=s.ssb)
    # plt.scatter(saddle_pca[0, 0], saddle_pca[0, 1], c='red', marker='x', s=100, label='Saddle Point')
    #
    # # Plot attractors in PCA space
    # # attractors_pca = pca.transform(attractors.detach().numpy())
    # # plt.scatter(attractors_pca[:, 0], attractors_pca[:, 1], c='black', marker='o', s=100, label='Attractors')
    #
    # plt.xlabel('First Principal Component')
    # plt.ylabel('Second Principal Component')
    # plt.title('Trajectories from Saddle Point (PCA)')
    # plt.legend()
    # plt.savefig(Path(cfg.savepath)/'trajectories2D.png')
    # plt.show()


def run_one_training_step_and_compute_grad(cfg):
    omegaconf_resolvers()
    dynamics_function = instantiate(cfg.dynamics.function)
    distribution_fit = instantiate(cfg.dynamics.IC_distribution_fit)

def plot_2D_dynamics_reduced_microbiome(cfg):
    omegaconf_resolvers()
    cfg.savepath = os.path.join(project_path, cfg.savepath)
    from ssr_module.steady_state_reduction_example import Params
    dynamics_function = instantiate(cfg.dynamics.function)
    point_on_separatrix = instantiate(cfg.dynamics.point_on_separatrix)
    attractors = instantiate(cfg.dynamics.attractors)

    saddle_point = instantiate(cfg.dynamics.saddle_point)

    p = Params()
    from ssr_module.steady_state_reduction_example import get_all_stein_steady_states, ssrParams, plot_ND_separatrix
    ssr_steady_states = get_all_stein_steady_states(p)
    s = ssrParams(p, ssr_steady_states['E'], ssr_steady_states['C'])

    # Load separatrix locator model
    SL = instantiate(cfg.separatrix_locator)
    SL.to('cpu')
    SL.models = [instantiate(cfg.model).to(SL.device) for _ in range(cfg.separatrix_locator.num_models)]

    # Set up load path
    new_format_path = Path(cfg.savepath) / cfg.experiment_details
    if os.path.exists(new_format_path):
        load_path = new_format_path
    else:
        load_path = Path(cfg.savepath)
        print(new_format_path, 'does not exist, loading', load_path, 'instead.')
    SL.load_models(load_path)

    # Create a grid of points spanning the space between attractors
    num_points = 20  # Number of points along each dimension

    # Get attractors as numpy arrays
    attractors_np = attractors.detach().cpu().numpy()

    # Create basis vectors from zero to each attractor
    v1 = attractors_np[0]  # Vector to first attractor
    v2 = attractors_np[1]  # Vector to second attractor

    # Create meshgrid of coefficients from 0 to 1
    x = np.linspace(0, 1, num_points)
    y = np.linspace(0, 1, num_points)
    X, Y = np.meshgrid(x, y)

    # Compute all grid points in parallel using broadcasting
    grid_points = (X[..., None] * v1) + (Y[..., None] * v2)

    # Convert to torch tensor
    grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32)

    # Reshape grid points for prediction
    original_shape = grid_points_tensor.shape
    grid_points_flat = grid_points_tensor.reshape(-1, grid_points_tensor.shape[-1])

    # Dynamics
    dynamics_vals = dynamics_function(grid_points_flat)

    # Project dynamics onto v1 and v2 basis vectors
    dynamics_np = dynamics_vals.detach().numpy()
    v1_norm = v1 / np.linalg.norm(v1) ** 2
    v2_norm = v2 / np.linalg.norm(v2) ** 2

    # Project dynamics onto 2D space using SSR projection
    dynamics_2d = p.project_to_2D(dynamics_np,ssa=v1,ssb=v2)

    # Reshape dynamics vectors to match grid shape
    dynamics_v1 = dynamics_2d[:,0].reshape(X.shape)
    dynamics_v2 = dynamics_2d[:,1].reshape(X.shape)
    # Stack into single array and reshape back to grid
    # dynamics_2d = np.stack([dynamics_v1, dynamics_v2], axis=-1)
    # dynamics_2d = dynamics_2d.reshape(original_shape[:-1] + (2,))


    # Get KEF predictions
    KEFvals = SL.predict(grid_points_flat)

    # Reshape predictions back to grid shape
    KEFvals = KEFvals.reshape(original_shape[:-1] + (-1,))

    # Plot the KEF values as a heatmap
    fig,ax= plt.subplots(figsize=(4, 4))
    # plt.contour(X, Y, KEFvals[..., 0], levels=20, cmap='RdBu')
    ax.contourf(X, Y, np.abs(KEFvals)[...,0], levels=15, cmap='Blues_r')
    CS = ax.contour(X, Y, KEFvals[...,0], levels=[0], colors='lightgreen')

    # Plot streamlines
    density = 0.5
    color='red'
    linewidth = 1
    alpha = 0.5
    lines = ax.streamplot(X, Y, dynamics_v1, dynamics_v2, density=density, color=color)
    lines.lines.set_alpha(alpha)
    # ax.colorbar(label='KEF Value')

    # Plot attractors
    # ax.scatter([0, 1], [0, 1], c='red', s=100, label='Attractors')
    # Project point_on_separatrix onto v1 and v2 basis vectors
    # Normalize the basis vectors v1 and v2 if they aren't already normalized

    # Project onto normalized basis vectors
    point_on_separatrix2D = p.project_to_2D(point_on_separatrix[None],ssa=v1,ssb=v2)
    saddle_point2D = p.project_to_2D(saddle_point[None], ssa=s.ssa, ssb=s.ssb)

    # Plot the projected point
    # ax.scatter(point_on_separatrix2D[:,0], point_on_separatrix2D[:,1], c='green', s=100, label='Separatrix Point')
    ax.scatter(saddle_point2D[:, 0], saddle_point2D[:, 1], c='lightgreen', s=200, marker='x')#, label='Saddle Point')
    ax.scatter(0, 1, c='lightgreen', s=200, marker='o') #, label='Saddle Point')
    ax.scatter(1, 0, c='lightgreen', s=200, marker='o') #, label='Saddle Point')

    plot_ND_separatrix(p,s,color='orange',ax=ax,sep_filename=Path('/home/kabird/separatrixLocator/')/'ssr_module/11D_separatrix_1e-2.data',label=None)

    # ax.set_xlabel('Coefficient of v1')
    # ax.set_ylabel('Coefficient of v2')
    ax.set_title('KEF Values in 2D Reduced Space')
    ax.set_xlim(0,1)
    ax.set_ylim(0, 1)
    # remove_frame(ax)
    ax.get_legend().remove()

    fig.tight_layout()
    fig.savefig(Path(cfg.savepath) / "KEF_2D_reduced.png", dpi=300)
    fig.savefig(Path(cfg.savepath) / "KEF_2D_reduced.pdf")
    plt.show()

    # np.unravel_index(np.argmin(((grid_points - point_on_separatrix.detach().numpy()) ** 2).sum(-1)), X.shape)
    # reconstructed_point = point_projected_v1 * v1 + point_projected_v2 * v2
    # # Calculate Euclidean distance between reconstructed point and original separatrix point
    # distance = np.linalg.norm(reconstructed_point - point_on_separatrix.detach().numpy())
    # print(f"Distance between reconstructed point and original separatrix point: {distance:.6f}")
    # distance_att = np.linalg.norm(attractors[0] - attractors[1])

    # After getting point_on_separatrix and before plotting
    # Get the point and basis vectors
    point = point_on_separatrix.detach().numpy()
    # point = saddle_point

    # Gram-Schmidt orthogonalization
    v1_norm = v1 / np.linalg.norm(v1)
    v2_orth = v2 - np.dot(v2, v1_norm) * v1_norm
    v2_norm = v2_orth / np.linalg.norm(v2_orth)

    # Project point onto the orthogonal basis
    proj_v1 = np.dot(point, v1_norm) * v1_norm
    proj_v2 = np.dot(point, v2_norm) * v2_norm
    point_in_plane = proj_v1 + proj_v2

    # Calculate distance from point to plane
    distance_to_plane = np.linalg.norm(point - point_in_plane)
    print(f"Distance of separatrix point from the plane: {distance_to_plane:.6f}")

    # Calculate the angle between the point and the plane
    angle = np.arcsin(distance_to_plane / np.linalg.norm(point)) * 180 / np.pi
    print(f"Angle between separatrix point and the plane: {angle:.6f} degrees")


def check_distribution(cfg):
    omegaconf_resolvers()
    cfg.savepath = os.path.join(project_path, cfg.savepath)
    dynamics_function = instantiate(cfg.dynamics.function)
    distribution = instantiate(cfg.dynamics.IC_distribution)
    point_on_separatrix = instantiate(cfg.dynamics.point_on_separatrix)
    attractors = instantiate(cfg.dynamics.attractors)
    distributions = instantiate(cfg.dynamics.IC_distribution_fit)

    # Create subplots for each dimension
    dim = cfg.dynamics.dim
    fig, axs = plt.subplots(dim, 1, figsize=(4, 2*dim))
    if dim == 1:
        axs = [axs]  # Make axs iterable when dim=1
    # Get samples from each distribution
    num_samples = 1000
    alpha = 0.3  # Transparency for all distributions

    for i in range(dim):
        # Plot histogram for each distribution
        for j, dist in enumerate(distributions):
            samples = dist.sample((num_samples,))
            axs[i].hist(samples[:,i].detach().numpy(), bins=50, density=True,
                       alpha=alpha, color=f'C{j}', label=f'Distribution {j+1}')

        # Plot vertical lines for attractors
        axs[i].axvline(x=attractors[0,i].item(), color='C3', linestyle='--', label='Attractor 1')
        axs[i].axvline(x=attractors[1,i].item(), color='C4', linestyle='--', label='Attractor 2')

        # Plot vertical line for separatrix point
        axs[i].axvline(x=point_on_separatrix[i].item(), color='C5', linestyle=':', label='Separatrix')

        axs[i].set_xlabel(f'Dimension {i+1}')
        axs[i].set_ylabel('Density')
        axs[i].legend()

    plt.tight_layout()
    plt.savefig(Path(cfg.savepath) / "distribution_check.png", dpi=300)
    plt.show()

def fixed_point_analysis(cfg):
    omegaconf_resolvers()

    from custom_distributions import gamma_from_mean_var

    dynamics_function = instantiate(cfg.dynamics.function)
    distribution = instantiate(cfg.dynamics.IC_distribution)
    point_on_separatrix = instantiate(cfg.dynamics.point_on_separatrix)
    attractors = instantiate(cfg.dynamics.attractors)
    from plotting import dynamics_to_kinetic_energy
    kinetic_energy_function = dynamics_to_kinetic_energy(dynamics_function)

    from odeint_utils import run_odeint_to_final

    distributions = instantiate(cfg.dynamics.IC_distribution_fit)

    # Sample points from distribution
    num_samples = 1000
    samples = distribution.sample((num_samples,))

    # Generate trajectories using odeint in batch
    total_time = 100
    initial_points = distribution.sample((200,))

    trajectories = run_odeint_to_final(dynamics_function, initial_points, total_time, return_last_only=False, steps = 100)

    # Do PCA on trajectories
    from sklearn.decomposition import PCA

    # Reshape trajectories to 2D array (time*batch_size, features)
    traj_reshaped = trajectories.reshape(-1, trajectories.shape[-1])

    # Fit PCA and transform data
    pca = PCA(n_components=2)
    traj_pca = pca.fit_transform(traj_reshaped.detach().numpy())

    # Reshape back to original dimensions (time, batch_size, 2)
    traj_pca = traj_pca.reshape(trajectories.shape[0], trajectories.shape[1], 2)

    # Plot PCA results
    plt.figure(figsize=(8,6))
    plt.plot(traj_pca[..., 0], traj_pca[..., 1], alpha=0.5)
    plt.scatter(traj_pca[-1,:,0], traj_pca[-1,:,1], alpha=0.5)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA of Final Trajectory Points')
    plt.show()

    # times = np.linspace(0,total_time,traj_pca.shape[0])
    # plt.figure()
    # plt.plot(times,traj_pca[:, :, 0], alpha=0.5)
    # plt.show()

    # Collect points from trajectories and add noise
    trajectory_points = trajectories.reshape(-1, trajectories.shape[-1])
    trajectory_points = trajectory_points[np.random.choice(trajectory_points.shape[0],size=num_samples)]
    noise = torch.randn_like(trajectory_points) * 0.1
    samples = trajectory_points + noise

    samples.requires_grad_(True)



    # Setup optimizer
    optimizer = torch.optim.Adam([samples], lr=0.05)

    # Optimize to find minima
    num_steps = 2000
    for step in range(num_steps):
        optimizer.zero_grad()

        # Calculate kinetic energy at current points
        ke = kinetic_energy_function(samples)

        # Loss is just the kinetic energy (we want to minimize it)
        loss = ke.mean()

        # Backprop and optimize
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f'Step {step}, Average KE: {loss.item():.6f}')

    # Get final optimized points
    with torch.no_grad():
        final_points = samples.detach()
        final_ke = kinetic_energy_function(final_points)

    # Use KMeans to cluster the points with lowest kinetic energy
    from sklearn.cluster import KMeans

    # Convert to numpy for KMeans
    points_np = final_points.detach().cpu().numpy()
    ke_np = final_ke.detach().cpu().numpy()

    # Take points with lowest kinetic energy for clustering
    # n_lowest = 1000 # Take more points initially to cluster
    lowest_indices, = np.where(final_ke.squeeze()<1e-5) #torch.argsort(final_ke.squeeze())[:n_lowest]
    print('len(lowest_indices)',len(lowest_indices))
    points_for_clustering = points_np[lowest_indices]

    # Perform KMeans clustering
    n_clusters = 4 # Adjust based on expected number of fixed points
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(points_for_clustering)

    # Get cluster centers as the unique minima
    minima = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

    # Calculate kinetic energy at minima points
    with torch.no_grad():
        minima_ke = kinetic_energy_function(minima)

    print("\nUnique minima found via clustering:")
    print(minima)
    print("\nKinetic energy at minima:")
    print(minima_ke)

    # Linearize dynamics around fixed points (minima)
    def compute_jacobian(dynamics_function, point):
        point.requires_grad_(True)
        dynamics = dynamics_function(point)
        jacobian = torch.autograd.functional.jacobian(dynamics_function, point)
        point.requires_grad_(False)
        return jacobian.squeeze()

    # First compute and store all jacobian info in a dictionary
    fixed_point_info = {}
    for i, fixed_point in enumerate(minima):
        info = {}
        info['point'] = fixed_point

        # Compute jacobian and eigenvalues
        J = compute_jacobian(dynamics_function, fixed_point)
        eigenvalues = torch.linalg.eigvals(J)
        real_parts = eigenvalues.real

        info['jacobian'] = J
        info['eigenvalues'] = eigenvalues

        # Classify stability
        if torch.all(real_parts < 0):
            info['stability'] = 'Stable'
            info['marker'] = '*'
            info['color'] = 'green'
        elif torch.all(real_parts > 0):
            info['stability'] = 'Unstable'
            info['marker'] = 'X'
            info['color'] = 'red'
        else:
            info['stability'] = 'Saddle'
            info['marker'] = 'D'
            info['color'] = 'orange'

        fixed_point_info[i] = info

        # Print analysis
        print(f"\nFixed point {i}:")
        print(f"Point: {fixed_point.numpy()}")
        print(f"Jacobian:\n{J.numpy()}")
        print(f"Eigenvalues: {eigenvalues.numpy()}")
        print(f"Classification: {info['stability']}")

    # Save fixed point analysis results
    import pickle

    # Save to pickle file in the same directory as other outputs
    pickle_path = Path(cfg.savepath) / "fixed_point_info.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(fixed_point_info, f)
    # Plot results
    plt.figure(figsize=(8, 6))

    # Do PCA if dimension > 2
    if points_np.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        points_pca = pca.fit_transform(points_np[lowest_indices])
        fixed_points_pca = pca.transform(np.vstack([info['point'].detach().cpu().numpy() for info in fixed_point_info.values()]))
        plot_x = points_pca[:,0]
        plot_y = points_pca[:,1]
        xlabel = 'First Principal Component'
        ylabel = 'Second Principal Component'
    else:
        plot_x = points_np[lowest_indices, 0]
        plot_y = points_np[lowest_indices, 1]
        xlabel = 'x'
        ylabel = 'y'

    # Plot all sampled points colored by kinetic energy
    plt.scatter(plot_x, plot_y,
                c=final_ke[lowest_indices].detach().cpu().numpy(),
                cmap='viridis', alpha=0.3, label='Sampled Points')

    # Plot fixed points with markers based on stability
    for i, info in fixed_point_info.items():
        if points_np.shape[1] > 2:
            point = fixed_points_pca[i]
        else:
            point = info['point'].detach().cpu().numpy()
        plt.scatter(point[0], point[1],
                   c=info['color'],
                   marker=info['marker'],
                   s=200,
                   label=f"{info['stability']} Fixed Point {i+1}",
                   zorder=5)
    plt.colorbar(label='Kinetic Energy')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('Fixed Points and Their Stability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(Path(cfg.savepath) / "fixed_points.png", dpi=300)
    plt.show()
    plt.close()

    for key,point in fixed_point_info.items():
        print(torch.max(point['eigenvalues'].real))

    idx = np.argmin([torch.max(point['eigenvalues'].real)-1 for key,point in fixed_point_info.items()])

    jac_saddle = fixed_point_info[1]['jacobian']
    eigvecs = np.linalg.eig(jac_saddle)[1]
    diff = eigvecs[:,1]
    proj = jac_saddle @ diff
    ratio = np.linalg.norm(proj) / np.linalg.norm(diff)

    attractors = instantiate(cfg.dynamics.attractors)
    diff = attractors[1]-attractors[0]
    jac = compute_jacobian(dynamics_function, point_on_separatrix)

    eigvals,eigvecs = np.linalg.eig(jac)

    diff = np.linalg.eig(jac_saddle)[1][:, 1]
    proj = jac @ diff

    ratio = np.linalg.norm(proj)/np.linalg.norm(diff)
    jac = compute_jacobian(dynamics_function, point_on_separatrix)


def plot_task_io(cfg):

    from plotting import remove_frame
    dataset = instantiate(cfg.dynamics.RNN_dataset)
    inputs, targets = dataset()
    omegaconf_resolvers()
    fig,axs = plt.subplots(2,1,figsize=np.array([3,2])*0.6,sharex=True)
    ax = axs[0]
    ax.plot(inputs[:,4,0],lw=2)
    ax = axs[1]
    ax.plot(targets[:,4,0],lw=2)
    for ax in axs:
        remove_frame(ax,['right','top','bottom'])
        ax.set_ylim(-1.1,1.1)
        ax.set_yticks([-1,1])
        ax.set_yticklabels([])
        ax.spines['left'].set_bounds(-1, 1)
    fig.savefig(Path('plots_for_publication')/'flip_flop_io.pdf')
    plt.show()

def plot_dynamics(cfg):
    """Plot dynamics streamlines and kinetic energy contours."""
    omegaconf_resolvers()
    cfg.savepath = os.path.join(project_path, cfg.savepath)
    dynamics_function = instantiate(cfg.dynamics.function)

    # attractors = instantiate(cfg.dynamics.attractors_from_authors)
    # attractors = instantiate(cfg.dynamics.attractors)
    from plotting import (
        plot_flow_streamlines,
        dynamics_to_kinetic_energy,
        evaluate_on_grid,
        remove_frame
    )

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    resolution = 75
    # Plot streamlines
    plot_flow_streamlines(dynamics_function, ax,
                         x_limits=cfg.dynamics.lims.x,
                         y_limits=cfg.dynamics.lims.y,
                         resolution=resolution, density=0.5,
                         color='red', linewidth=0.5, alpha=0.4)

    # Plot kinetic energy contours
    kinetic_energy_function = dynamics_to_kinetic_energy(dynamics_function)
    X, Y, kinetic_energy_vals = evaluate_on_grid(kinetic_energy_function,
                                                x_limits=cfg.dynamics.lims.x,
                                                y_limits=cfg.dynamics.lims.y,
                                                resolution=resolution)
    ax.contourf(X, Y, np.log(kinetic_energy_vals), levels=25, cmap='Blues_r')

    ax.set_title(r'$q(x)$')

    ax.set_xlim(*cfg.dynamics.lims.x)
    ax.set_ylim(*cfg.dynamics.lims.y)
    remove_frame(ax)
    ax.set_aspect('equal')
    # plt.show()
    plt.savefig(Path(cfg.savepath) / cfg.experiment_details / "dynamics.png", dpi=200)

    # Plot trajectories from random initial conditions
    distribution = instantiate(cfg.dynamics.IC_distribution)
    num_trajectories = 10
    total_time = 5
    times = torch.linspace(0, total_time, 200)
    initial_conditions = distribution.sample((num_trajectories,))

    # Run trajectories
    trajectories = odeint(lambda t,x: dynamics_function(x), initial_conditions, times)

    # Plot trajectories
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    
    # Plot x1 component
    for traj in trajectories.transpose(0,1):
        ax1.plot(times, traj[:,0], alpha=0.5)
    ax1.set_ylabel('$x_1$')
    
    # Plot x2 component  
    for traj in trajectories.transpose(0,1):
        ax2.plot(times, traj[:,1], alpha=0.5)
    ax2.set_ylabel('$x_2$')
    ax2.set_xlabel('Time')

    plt.tight_layout()
    plt.savefig(Path(cfg.savepath) / cfg.experiment_details / "trajectories.png", dpi=200)
    plt.close()
    

def plot_KEF_residual_heatmap(cfg):
    """Plot KEF residual as a 2D heatmap showing (LHS-RHS)/sqrt(RHS**2)."""
    omegaconf_resolvers()
    cfg.savepath = os.path.join(project_path, cfg.savepath)

    # Load dynamics function and separatrix locator
    dynamics_function = instantiate(cfg.dynamics.function)
    SL = instantiate(cfg.separatrix_locator)
    SL.to('cpu')
    SL.models = [instantiate(cfg.model).to(SL.device) for _ in range(cfg.separatrix_locator.num_models)]

    # Set up load path
    new_format_path = Path(cfg.savepath) / cfg.experiment_details
    if os.path.exists(new_format_path):
        load_path = new_format_path
    else:
        load_path = Path(cfg.savepath)
        print(new_format_path, 'does not exist, loading', load_path, 'instead.')
    SL.load_models(load_path)

    # Get first model
    model = SL.models[0]

    from plotting import evaluate_on_grid

    def residual_function(x):
        x = torch.tensor(x, requires_grad=True)
        phi_x = model(x)[...,0:1]
        phi_x_prime = torch.autograd.grad(
            outputs=phi_x,
            inputs=x,
            grad_outputs=torch.ones_like(phi_x),
            create_graph=True
        )[0]
        dynamics_vals = dynamics_function(x)
        dot_prod = torch.sum(phi_x_prime * dynamics_vals, dim=1, keepdim=True)
        residual = (dot_prod - phi_x) / torch.sqrt(phi_x**2)
        return residual.detach()

    # Evaluate residual on grid
    X, Y, residual = evaluate_on_grid(residual_function,
                                     x_limits=cfg.dynamics.lims.x,
                                     y_limits=cfg.dynamics.lims.y,
                                     resolution=100)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(residual,
                   extent=[cfg.dynamics.lims.x[0], cfg.dynamics.lims.x[1],
                          cfg.dynamics.lims.y[0], cfg.dynamics.lims.y[1]],
                   origin='lower',
                   aspect='equal',
                   cmap='RdBu',
                   vmin=-1, vmax=1)

    # Add colorbar
    plt.colorbar(im, ax=ax, label=r'$\frac{\nabla \psi(x) \cdot f(x) - \lambda\psi(x)}{\sqrt{\psi(x)^2}}$')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('KEF Residual')

    # Save figure
    fig.savefig(Path(cfg.savepath) / cfg.experiment_details / "KEF_residual_heatmap.png", dpi=200)
    plt.close(fig)



def plot_KEF_residuals(cfg):
    """Plot KEF values vs residuals, standard deviation vs residuals, and LHS vs RHS."""
    omegaconf_resolvers()
    cfg.savepath = os.path.join(project_path, cfg.savepath)

    # Load dynamics function
    dynamics_function = instantiate(cfg.dynamics.function)

    # Load distribution
    distribution = instantiate(cfg.dynamics.IC_distribution)
    if hasattr(cfg.dynamics, 'IC_distribution_fit'):
        distribution_fit = instantiate(cfg.dynamics.IC_distribution_fit)
    else:
        distribution_fit = distribution

    # Load separatrix locator model
    SL = instantiate(cfg.separatrix_locator)
    SL.to('cpu')
    SL.models = [instantiate(cfg.model).to(SL.device) for _ in range(cfg.separatrix_locator.num_models)]

    # Set up load path
    new_format_path = Path(cfg.savepath) / cfg.experiment_details
    if os.path.exists(new_format_path):
        load_path = new_format_path
    else:
        load_path = Path(cfg.savepath)
        print(new_format_path, 'does not exist, loading', load_path, 'instead.')
    SL.load_models(load_path)
    # SL = load_models_with_different_speeds(SL,new_format_path)

    # Define distributions to test
    distributions = distribution_fit
    dist_names = range(len(distribution_fit))
    batch_size = 1000

    for dist, name in zip(distributions, dist_names):
        # Sample input_tensor from the distribution
        input_tensor = dist.sample(sample_shape=(batch_size,))
        input_tensor.requires_grad_(True)

        # Lists to store values from all models
        all_phi_x = []
        all_dot_prod = []
        all_residuals = []

        # Compute values for each model
        for model in SL.models:
            # Compute phi(x)
            phi_x = model(input_tensor)[...,0:1]

            # Compute phi'(x) using autograd
            phi_x_prime = torch.autograd.grad(
                outputs=phi_x,
                inputs=input_tensor,
                grad_outputs=torch.ones_like(phi_x),
                create_graph=True
            )[0]

            F_x = dynamics_function(input_tensor)

            # Compute the dot product
            dot_prod = (phi_x_prime * F_x).sum(axis=-1, keepdim=True)

            residual = torch.abs(phi_x - dot_prod)

            all_phi_x.append(phi_x.detach().cpu().numpy())
            all_dot_prod.append(dot_prod.detach().cpu().numpy())
            all_residuals.append(residual.detach().cpu().numpy())

        # Plot KEF values vs residuals
        fig, ax = plt.subplots()
        for i, (phi_x, residual) in enumerate(zip(all_phi_x, all_residuals)):
            ax.scatter(np.abs(phi_x), residual,
                      label=f'Model {i}', alpha=0.5)
        ax.set_xlabel(r'$\psi(x)$')
        ax.set_ylabel(r'$|\nabla \psi(x) \cdot f(x) - \lambda\psi(x)|$')
        ax.legend()
        fig.tight_layout()
        fig.savefig(Path(cfg.savepath) / cfg.experiment_details / f"KEFvals_vs_residuals_{name}.png", dpi=200)
        plt.show()
        plt.close(fig)

        # Plot standard deviation vs residuals
        std_i = torch.std(input_tensor.detach(), axis=-1, keepdims=True)
        fig, ax = plt.subplots()
        for i, residual in enumerate(all_residuals):
            ax.scatter(std_i.cpu().numpy(), residual,
                      label=f'Model {i}', s=5, alpha=0.5)
        ax.set_xlabel(r'$std(x_i)$')
        ax.set_ylabel(r'$|\nabla \psi(x) \cdot f(x) - \lambda\psi(x)|$')
        ax.legend()
        fig.tight_layout()
        plt.show()
        fig.savefig(Path(cfg.savepath) / cfg.experiment_details / f"Stdi_vs_residuals_{name}.png", dpi=200)
        plt.close(fig)

        # Plot LHS vs RHS
        fig, ax = plt.subplots()
        for i, (phi_x, dot_prod) in enumerate(zip(all_phi_x, all_dot_prod)):
            ax.scatter(phi_x, dot_prod,
                      label=f'Model {i}', alpha=0.5)
        ax.set_xlabel(r'$\lambda\psi(x)$')
        ax.set_ylabel(r'$\nabla \psi(x) \cdot f(x)$')
        ax.legend()

        # Add a dashed line for x=y
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.plot([xlim[0], xlim[1]], [xlim[0], xlim[1]], color='black', linestyle='dashed', linewidth=1)

        fig.tight_layout()
        fig.savefig(Path(cfg.savepath) / cfg.experiment_details / f"KEF_LHS_RHS_{name}.png", dpi=200)
        plt.show()
        plt.close(fig)

        # Plot violin plot of phi_x values
        fig, ax = plt.subplots()
        violin_data = [phi_x.flatten() for phi_x in all_phi_x]  # Flatten arrays to 1D
        labels = [f'Model {i}' for i in range(len(all_phi_x))]
        ax.violinplot(violin_data, showmedians=True)
        ax.set_xticks(range(1, len(all_phi_x) + 1))
        ax.set_xticklabels(labels)
        ax.set_xlabel('Model')
        ax.set_ylabel(r'$\psi(x)$')
        fig.tight_layout()
        fig.savefig(Path(cfg.savepath) / cfg.experiment_details / f"KEFvals_violin_{name}.png", dpi=200)
        plt.show()
        plt.close(fig)

        # Compute and print mean/std ratio for each model's phi_x values
        for i, phi_x in enumerate(all_phi_x):
            mean_std_ratio = np.mean(phi_x) / np.std(phi_x)
            print(f"Model {i} mean/std ratio: {mean_std_ratio:.3f}")


    # separatrix_point = instantiate(cfg.dynamics.point_on_separatrix)
    # KEFval_at_separatrix = SL.predict(separatrix_point)
    # KEFval_at_separatrix

def plot_dynamics_2D(cfg):
    omegaconf_resolvers()
    cfg.savepath = os.path.join(project_path, cfg.savepath)

    dynamics_function = instantiate(cfg.dynamics.function)

    distribution = instantiate(cfg.dynamics.IC_distribution)


    ### Loading separatrix locator models
    SL = instantiate(cfg.separatrix_locator)
    SL.to('cpu')
    # SL.init_models()
    SL.models = [instantiate(cfg.model).to(SL.device) for _ in range(cfg.separatrix_locator.num_models)]
    new_format_path = Path(cfg.savepath) / cfg.experiment_details
    print(new_format_path)
    if os.path.exists(new_format_path):
        load_path = new_format_path
    else:
        load_path = Path(cfg.savepath)
        print(new_format_path, 'does not exist, loading', load_path, 'instead.')
    SL.load_models(load_path)
    KEF_func = SL.prepare_models_for_gradient_descent(distribution)[0]

    from plotting import (
        plot_flow_streamlines,
        dynamics_to_kinetic_energy,
        evaluate_on_grid,
        remove_frame
    )

    colors = sns.color_palette("bright")
    # Plot the trajectories
    fig, axs = plt.subplots(1, 2, figsize=(5, 3))
    # Plot initial conditions
    ax = axs[0]
    # plot_flow_streamlines(dynamics_function, ax, x_limits=cfg.dynamics.lims.x, y_limits=cfg.dynamics.lims.y,
    #                       resolution=200, density=0.5, color='red', linewidth=0.5, alpha=0.4)
    kinetic_energy_function = dynamics_to_kinetic_energy(dynamics_function)
    X, Y, kinetic_energy_vals = evaluate_on_grid(kinetic_energy_function,
                                                 x_limits=cfg.dynamics.lims.x, y_limits=cfg.dynamics.lims.y, resolution=200)
    print(X.shape, Y.shape, np.log(kinetic_energy_vals).shape)
    ax.contourf(X, Y, np.log(kinetic_energy_vals), levels=15, cmap='Blues_r')
    ax.set_title(r'$q(x)$')
    # for pointset in pointsets:
    #     ax.scatter(**pointset)

    # Plot streamlines
    ax = axs[1]

    X, Y, log_KEF_vals = evaluate_on_grid(KEF_func,
                                                 x_limits=cfg.dynamics.lims.x, y_limits=cfg.dynamics.lims.y,
                                                 resolution=200)
    X, Y, KEF_vals_raw = evaluate_on_grid(lambda x: SL.predict(x)[...,0],
                    x_limits=cfg.dynamics.lims.x, y_limits=cfg.dynamics.lims.y,
                    resolution=200)
    KEF_vals_abs = np.abs(KEF_vals_raw)
    # KEF_vals_abs[KEF_vals_abs>1] = np.inf
    ax.contourf(X, Y, KEF_vals_abs, levels=15, cmap='Blues_r')
    CS = ax.contour(X, Y, KEF_vals_raw, levels=[0], colors='lightgreen')
    ax.clabel(CS, CS.levels, fontsize=10)
    plot_flow_streamlines(dynamics_function, axs.flatten(), x_limits=cfg.dynamics.lims.x, y_limits=cfg.dynamics.lims.y,
                          resolution=200, density=0.7, color='red', linewidth=0.5, alpha=0.4)

    ax.set_title(r'$\psi(x)$')
    # im = ax.imshow(torch.log(torch.abs(KEFvalgrid)), extent=[*cfg.dynamics.lims.x, *cfg.dynamics.lims.y], origin='lower', cmap='viridis')

    for ax in axs:
        if hasattr(cfg.dynamics, 'point_on_separatrix'):
            separatrix_point = instantiate(cfg.dynamics.point_on_separatrix)
            ax.scatter(separatrix_point[0],separatrix_point[1])
        if hasattr(cfg.dynamics, 'saddle_point'):
            saddle_point = instantiate(cfg.dynamics.saddle_point)
            ax.scatter(saddle_point[0],saddle_point[1],marker='x',c='red')

    # Plot trajectories if specified in config
    if hasattr(cfg, 'plot_trajectories_in_2D') and cfg.plot_trajectories_in_2D:
        # Sample initial conditions uniformly
        num_trajectories = 20
        initial_conditions = torch.FloatTensor(num_trajectories, 2).uniform_(-4, 4)
        
        # Time settings for integration
        t_span = torch.linspace(0, 10, 100)
        
        # Run trajectories
        trajectories = odeint(
            lambda t, x: dynamics_function(x),
            initial_conditions,
            t_span
        )
        
        # Plot trajectories on both axes
        for ax in axs:
            trajectories_np = trajectories.detach().cpu().numpy()
            for traj in trajectories_np.transpose(1,0,2):
                ax.plot(traj[:,0], traj[:,1], 'k-', alpha=0.3, linewidth=0.5)

    for ax in axs.flatten():
        ax.set_xlim(*cfg.dynamics.lims.x)
        ax.set_ylim(*cfg.dynamics.lims.y)
        remove_frame(ax)
        ax.set_aspect('equal')


        if hasattr(cfg.dynamics,'plot_fixed_points'):
            pointsets = instantiate(cfg.dynamics.plot_fixed_points)
            # Check if fixed point data exists and plot it
            fixed_point_path = Path(cfg.savepath) / 'fixed_point_data.csv'
            if fixed_point_path.exists():
                fixed_points = pd.read_csv(fixed_point_path)
                fixed_points = fixed_points.loc[fixed_points['q']<1e-9]
                for pointset in pointsets:
                    if 'unstable' in pointset['label']:
                        stability = False
                    else:
                        stability = True
                    pointset['x'] = fixed_points.loc[fixed_points['stability']==stability]['x0']
                    pointset['y'] = fixed_points.loc[fixed_points['stability']==stability]['x1']
            for pointset in pointsets:
                ax.scatter(**pointset,c='lightgreen')

    if hasattr(cfg, 'plot_limit_cycles'):
        limit_cycles = instantiate(cfg.plot_limit_cycles)
        for ax in axs.flatten():
            # Plot stable limit cycles in black
            if 'stable' in limit_cycles:
                for radius in limit_cycles['stable']:
                    theta = np.linspace(0, 2*np.pi, 100)
                    x = radius * np.cos(theta)
                    y = radius * np.sin(theta)
                    ax.plot(x, y, 'k-', alpha=0.8, linewidth=1)
            
            # Plot unstable limit cycles in orange  
            if 'unstable' in limit_cycles:
                for radius in limit_cycles['unstable']:
                    theta = np.linspace(0, 2*np.pi, 100)
                    x = radius * np.cos(theta)
                    y = radius * np.sin(theta)
                    ax.plot(x, y, 'orange', linestyle='--', alpha=0.8, linewidth=1)

    fig.tight_layout()
    fig.savefig(Path(cfg.savepath) / cfg.experiment_details / f'results2D_{cfg.dynamics.name}.pdf')
    fig.savefig(Path(cfg.savepath) / cfg.experiment_details / f'results2D_{cfg.dynamics.name}.png',dpi=300)
    plt.show()


def RNN_modify_inputs(cfg):
    omegaconf_resolvers()

    cfg.savepath = os.path.join(project_path, cfg.savepath)

    orig_model = instantiate(cfg.dynamics.loaded_RNN_model)

    ### Loading separatrix locator models
    SL = instantiate(cfg.separatrix_locator)
    # SL.init_models()
    SL.models = [instantiate(cfg.model).to(SL.device) for _ in range(cfg.separatrix_locator.num_models)]
    new_format_path = Path(cfg.savepath) / cfg.experiment_details
    print(new_format_path)
    if os.path.exists(new_format_path):
        load_path = new_format_path
    else:
        load_path = Path(cfg.savepath)
        print(new_format_path, 'does not exist, loading', load_path, 'instead.')
    SL.load_models(load_path)


    # dataset = instantiate(cfg.dynamics.RNN_dataset)
    # dataset = instantiate(cfg.dynamics.RNN_analysis_dataset)
    dataset = instantiate(cfg.dynamics.RNN_analysis_dataset_opposite)
    # print(model)

    num_models = 50
    models = []
    for _ in range(num_models):
        models.append(deepcopy(orig_model))

    frac_to_permute = 0.25

    for model in models:
        # Get the current input weights
        current_weights = model.rnn.weight_ih_l0
        # Create a random permutation of the input dimension
        num_to_permute = int(current_weights.shape[0] * frac_to_permute)
        perm = torch.arange(current_weights.shape[0])
        indices_to_permute = torch.randperm(current_weights.shape[0])[:num_to_permute]
        perm[indices_to_permute] = perm[indices_to_permute[torch.randperm(num_to_permute)]]
        # Apply the permutation to the input weights
        model.rnn.weight_ih_l0 = torch.nn.Parameter(current_weights[perm])

    inputs,targets = dataset()
    inputs[:20] = 0
    # inputs *= 30
    inputs = torch.tensor(inputs,dtype=torch.float32)
    # Run all models on inputs
    outputs = []
    hiddens = []
    for model in models:
        with torch.no_grad():
            output,hidden = model(inputs,return_hidden=True)
            outputs.append(output.detach())
            hiddens.append(hidden.detach())

    orig_output = orig_model(inputs).detach()
    orig_output_np = orig_output.detach().numpy()

    KEFvals = SL.predict(torch.stack(hiddens))

    # Convert outputs to numpy for plotting
    outputs_np = [out.detach().cpu().numpy() for out in outputs]

    outputs_np = np.stack(outputs_np,axis=0)
    outputs_np_sign = np.sign(outputs_np[:,-1]).astype(int)
    outputs_np_sign = (outputs_np_sign+1)//2
    # Plot the outputs
    import matplotlib.pyplot as plt
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    trial_num = 7

    # Plot inputs in upper panel
    ax1.plot(inputs[:, trial_num, 0], 'b-', label='Input', linewidth=1)
    ax1.set_ylabel('Input')
    ax1.set_title('Input Signal')
    ax1.grid(True)

    # Plot outputs in middle panel
    for i, output in enumerate(outputs_np):
        color = f'C{outputs_np_sign[i,trial_num,0]}'
        ax2.plot(output[:, trial_num, 0], alpha=0.3, label=f'Model {i+1}',c=color, linewidth=2)
    # ax2.plot(targets[:, trial_num, 0], 'k--', label='Target', linewidth=2)
    ax2.plot(orig_output_np[:, trial_num, 0], 'k--', label='Original model output', linewidth=2)
    ax2.set_ylabel('Output')
    ax2.set_title('Outputs from Different Models with Permuted Input Weights')
    # ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True)

    # Plot KEF values in lower panel
    KEFmodelnum = 0
    for i in range(len(outputs_np)):
        color = f'C{outputs_np_sign[i,trial_num,0]}'
        ax3.plot(KEFvals[i, :, trial_num, KEFmodelnum], label='KEF Values', linewidth=1, color=color)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('KEF Value')
    ax3.set_title('KEF Values Over Time')
    ax3.grid(True)

    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(250,270)

    plt.tight_layout()
    plt.show()

    plt.savefig(os.path.join(cfg.savepath, 'RNN_permuted_input_weights.png'))


def plot_hermite_polynomials_2d(cfg):
    """
    Plot 2D visualization of cubic Hermite polynomial interpolations between attractors

    Args:
        cfg: Config object containing dynamics and plotting parameters
    """
    from interpolation import cubic_hermite, generate_curves_between_points

    omegaconf_resolvers()
    cfg.savepath = os.path.join(project_path, cfg.savepath)

    # Get attractors from config
    attractors = instantiate(cfg.dynamics.attractors)
    x, y = attractors.detach().cpu().numpy()



    # Set up tangent vectors at endpoints
    m_x = -x + y  # Tangent at x
    m_y = -x + y  # Tangent at y

    # Generate multiple curves with random perturbations
    num_curves = 100
    num_points = 100
    rand_scale = 2.0

    all_points = []
    for _ in range(num_curves):
        # Generate points until we get a valid curve with no negative points
        num_tries = 0
        while num_tries < 1000:
            # Generate random perturbations for tangents
            m_x_perturbed = m_x * np.random.uniform(size=m_x.shape[0]) * rand_scale
            m_y_perturbed = m_y * np.random.uniform(size=m_x.shape[0]) * rand_scale

            # Generate points on the cubic Hermite curve
            points = cubic_hermite(x, y, m_x_perturbed, m_y_perturbed, num_points, lims=[0.05, 0.95])

            # Check if any points are negative
            if not np.any(points < 0):
                all_points.append(points)
                print(num_tries)
                break
            num_tries += 1

    all_points_st = np.stack(all_points)
    all_points_st = generate_curves_between_points(x, y, lims=[0.05, 0.95])
    assert np.all(all_points_st>=0), "points are negative"
    # assert np.all(points >= 0), "points are negative"

    # Plot attractor
    plt.figure(figsize=(10, 10))
    # plt.scatter(points[...,0],points[...,2])
    plt.scatter(all_points_st[...,0], all_points_st[...,2])
    plt.scatter(x[0], x[2], c='red', s=100, label='Attractor 1')
    plt.scatter(y[0], y[2], c='red', s=100, label='Attractor 2')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Cubic Hermite Polynomial Interpolations')
    plt.legend()
    plt.grid(True)

    # Save figure
    plt.savefig(Path(cfg.savepath) / "hermite_polynomials_2d.png", dpi=300)
    plt.show()
    plt.close()

    np.where(points<0)


def plot_cubichermitesampler(cfg):
    omegaconf_resolvers()

    cfg.savepath = os.path.join(project_path, cfg.savepath)


    attractors = instantiate(cfg.dynamics.attractors)

    distributions = list(instantiate(cfg.dynamics.IC_distribution_fit))
    # distribution = distributions[5]

    # distribution = CubicHermiteSampler(attractors, alpha_dist=torch.distributions.Beta(4,4), scale=20.0)
    # Sample points from all distributions
    all_points = []
    for dist in distributions:
        points = dist.sample(sample_shape=(200,))
        all_points.append(points)

    # Concatenate all points
    points = torch.cat(all_points, dim=0)
    # points = distribution.sample(sample_shape=(200,))
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    # Convert points to numpy for PCA
    points_np = points.detach().cpu().numpy()

    # Convert attractors to numpy
    attractors_np = attractors.detach().cpu().numpy()

    # Fit PCA on points
    pca = PCA(n_components=2)
    points_pca = pca.fit_transform(points_np)
    attractors_pca = pca.transform(attractors_np)

    # Plot points and attractors
    plt.figure(figsize=(6, 4))
    plt.scatter(points_pca[:, 0], points_pca[:, 1], alpha=0.5, label='Sampled Points')
    plt.scatter(attractors_pca[:, 0], attractors_pca[:, 1], c='red', s=100, marker='*', label='Attractors')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Initial Conditions and Attractors in PCA Space')
    plt.legend()
    plt.grid(True)
    plt.show()

def check_basin_of_attraction(cfg):
    omegaconf_resolvers()

    cfg.savepath = os.path.join(project_path, cfg.savepath)

    from odeint_utils import run_odeint_to_final
    from custom_distributions import CubicHermiteSampler
    dynamics_function = instantiate(cfg.dynamics.function)

    attractors = instantiate(cfg.dynamics.attractors)

    distributions = list(instantiate(cfg.dynamics.IC_distribution_fit))
    xtick_labels = range(len(distributions)) #[f'Cubic {d.scale}' for d in cfg.dynamics.IC_distribution_fit]

    # distributions += list(instantiate(cfg.dynamics.IC_distribution_fit_old))
    # xtick_labels += [f'Isotropic {r}' for r in list(cfg.dynamics.scale_range.object)]
    #
    # # distributions += [
    # #     CubicHermiteSampler(*attractors,scale=0.1),
    # #     CubicHermiteSampler(*attractors, scale=0.5),
    # #     CubicHermiteSampler(*attractors, scale=1.0),
    # #     CubicHermiteSampler(*attractors, scale=2.0),
    # #     CubicHermiteSampler(*attractors, scale=4.0),
    # # ]
    # # xtick_labels += ['Cubic 0.1', 'Cubic 0.5', 'Cubic 1.0', 'Cubic 2.0', 'Cubic 4.0']
    finals = []
    initial_conditions_list = []
    for i, dist in enumerate(distributions):
        initial_conditions = dist.sample(sample_shape=(500,))
        initial_conditions_list.append(initial_conditions)

        T = 50
        final = run_odeint_to_final(
            dynamics_function,
            initial_conditions,
            inputs=instantiate(cfg.dynamics.static_external_input),
            T=T,
            return_last_only=True
        )
        finals.append(final)

    # Stack both initial conditions and final states
    stacked_initials = torch.stack(initial_conditions_list)
    stacked_finals = torch.stack(finals)


    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt

    # Convert final tensors to numpy arrays for KMeans
    final_nps = stacked_finals.detach().cpu().numpy().reshape(-1,stacked_finals.shape[-1])

    # Try different numbers of clusters for each distribution
    n_clusters_range = range(1, 11)
    inertias = []


    dist_inertias = []
    for n_clusters in n_clusters_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(final_nps)
        dist_inertias.append(kmeans.inertia_)
    inertias.append(dist_inertias)

    ########## KNN to predict final basin from initial conditions ##########
    labels = kmeans.labels_.reshape(*stacked_finals.shape[:2])
    unique_labels, counts = np.unique(labels[0], return_counts=True)
    sorted_indices = np.argsort(counts)[::-1]  # Sort in descending order
    top_two_labels = unique_labels[sorted_indices[:2]]
    # other_labels = np.setdiff1d(all_unique_labels, top_two_labels)  # Get all labels except top two

    is_top_two_labels = np.isin(kmeans.labels_,top_two_labels)

    # Train KNN to predict final basin from initial conditions
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split

    # Reshape initial conditions to 2D array for sklearn
    initial_nps = stacked_initials.detach().cpu().numpy().reshape(-1, stacked_initials.shape[-1])

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        initial_nps,
        # kmeans.labels_,
        is_top_two_labels,
        test_size=0.2,
        random_state=42
    )
    # Try different numbers of neighbors
    n_neighbors_range = range(1, 11, 2)  # Odd numbers from 1 to 20
    train_accuracies = []
    test_accuracies = []

    for n_neighbors in n_neighbors_range:
        # Train KNN classifier
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)

        # Evaluate accuracy
        train_accuracies.append(knn.score(X_train, y_train))
        test_accuracies.append(knn.score(X_test, y_test))

    # Plot accuracies vs number of neighbors
    plt.figure(figsize=(8, 4))
    plt.plot(n_neighbors_range, train_accuracies, 'o-', label='Training accuracy')
    plt.plot(n_neighbors_range, test_accuracies, 'o-', label='Test accuracy')
    plt.xlabel('Number of neighbors')
    plt.ylabel('Accuracy')
    plt.title('KNN Classifier Performance vs Number of Neighbors')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print best results
    best_k_idx = np.argmax(test_accuracies)
    best_k = n_neighbors_range[best_k_idx]
    print(f"\nBest KNN Results (k={best_k}):")
    print(f"Training accuracy: {train_accuracies[best_k_idx]:.3f}")
    print(f"Test accuracy: {test_accuracies[best_k_idx]:.3f}")

    ############### SVM to predict final basin from initial conditions ##############
    # Train SVM to predict final basin from initial conditions
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler

    # Scale the features for better SVM performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define parameter combinations
    C_range = [0.1, 1, 10, 100, 1000]
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    svm_train_accuracies = []
    svm_test_accuracies = []

    # Create list of SVM classifiers with different parameters
    svm_classifiers = [
        SVC(C=C, kernel=kernel)
        for C in C_range
        for kernel in kernels
    ]

    # Train and evaluate each classifier
    for svm in svm_classifiers:
        svm.fit(X_train_scaled, y_train)

        # Evaluate accuracy
        svm_train_accuracies.append(svm.score(X_train_scaled, y_train))
        svm_test_accuracies.append(svm.score(X_test_scaled, y_test))

    # Plot accuracies vs C parameter
    plt.figure(figsize=(8, 4))
    plt.semilogx(range(len(svm_classifiers)), svm_train_accuracies, 'o-', label='Training accuracy')
    plt.semilogx(range(len(svm_classifiers)), svm_test_accuracies, 'o-', label='Test accuracy')
    plt.xlabel('C parameter')
    plt.ylabel('Accuracy')
    plt.title('SVM Classifier Performance vs C Parameter')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print best results
    best_C_idx = np.argmax(svm_test_accuracies)
    best_C = C_range[best_C_idx]
    print(f"\nBest SVM Results (C={best_C}):")
    print(f"Training accuracy: {svm_train_accuracies[best_C_idx]:.3f}")
    print(f"Test accuracy: {svm_test_accuracies[best_C_idx]:.3f}")
    ##########

    # Plot the elbow curves
    plt.figure(figsize=(5,4))
    for i, dist_inertias in enumerate(inertias):
        plt.plot(n_clusters_range, dist_inertias, 'bo-', label=f'Distribution {i+1}')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal Number of Clusters')
    plt.legend()
    plt.grid(True)
    plt.show()

    labels = kmeans.labels_.reshape(*stacked_finals.shape[:2])

    # Count occurrences of each label for each distribution
    for i, dist_labels in enumerate(labels):
        unique_labels, counts = np.unique(dist_labels, return_counts=True)
        print(f"\nDistribution {i+1} label counts:")
        for label, count in zip(unique_labels, counts):
            print(f"Label {label}: {count} points")

    num_distributions = labels.shape[0]
    all_unique_labels = np.unique(labels)

    # Build a count matrix: rows = distributions, columns = label counts
    count_matrix = np.zeros((num_distributions, len(all_unique_labels)))

    for i, dist_labels in enumerate(labels):
        unique_labels, counts = np.unique(dist_labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            label_idx = np.where(all_unique_labels == label)[0][0]
            count_matrix[i, label_idx] = count

    # stacked_finals_np = stacked_finals.detach().cpu().numpy()
    stacked_initials_np = stacked_initials.detach().cpu().numpy()
    # Get the two most frequent labels in the first distribution
    unique_labels, counts = np.unique(labels[0], return_counts=True)
    sorted_indices = np.argsort(counts)[::-1]  # Sort in descending order
    top_two_labels = unique_labels[sorted_indices[:2]]
    other_labels = np.setdiff1d(all_unique_labels, top_two_labels)  # Get all labels except top two

    from suppressed_distributions import find_lda_direction, construct_anisotropic_cov

    for i,dist in enumerate(distributions):
        if i == 6:
            dist_label = labels[i]
            is_top_label = np.isin(dist_label, top_two_labels)
            points_not_in_twobasins = stacked_initials_np[i,~is_top_label]
            # lda_dir = find_lda_direction(stacked_initials_np[i], dist_label, leaky_labels=other_labels)
            # Sigma_aniso = construct_anisotropic_cov(1.0, lda_dir, suppression=0.1)
            # if np.sum(~is_top_label)>0:
            points_in_twobasins = stacked_initials_np[i, is_top_label]
            p = PCA().fit(points_in_twobasins)
            lam = p.explained_variance_
            print(np.sum(lam)**2/np.sum(lam**2))

    # Estimate mean and covariance from points in two basins
    # mean = torch.tensor(np.mean(points_in_twobasins, axis=0), dtype=torch.float32)
    # mean = distributions[0].mean
    # cov = torch.cov(torch.tensor(points_in_twobasins.T,dtype=torch.float32))
    # cov = cov*0.4 + 1e-3 * torch.eye(cov.shape[0])
    # Create multivariate normal distribution
    # dist4 = torch.distributions.MultivariateNormal(mean, cov)
    # samples = dist4.sample(sample_shape=(500,))
    samples = distributions[6].sample(sample_shape=(500,))
    kmeans_pred = kmeans.predict(samples.detach().cpu().numpy())
    print(np.unique(kmeans_pred,return_counts=True))
    final = run_odeint_to_final(
        dynamics_function,
        samples,
        inputs=instantiate(cfg.dynamics.static_external_input),
        T=T,
        return_last_only=True
    )
    # Get labels for new points using existing kmeans model
    final_np = final.detach().cpu().numpy()
    final_labels = kmeans.predict(final_np)
    np.unique(final_labels,return_counts=True)

    # Now plot
    fig, ax = plt.subplots(figsize=(8, 6))

    bottom = np.zeros(num_distributions)  # To stack bars on top of each other
    for j, label in enumerate(all_unique_labels):
        ax.bar(
            np.arange(num_distributions),
            count_matrix[:, j],
            bottom=bottom,
            label=f'Cluster {label}'
        )
        bottom += count_matrix[:, j]


    ax.set_xlabel('Distribution radius')
    ax.set_xticks(np.arange(len(xtick_labels)))
    ax.set_xticklabels(xtick_labels)
    ax.set_xticklabels(xtick_labels, rotation=90)
    ax.set_ylabel('Number of points')
    ax.set_title('Final point clustering for different IC distributions')
    ax.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(cfg.savepath, 'clustering.png'))

    # plt.close()

    # # Print the inertia values for analysis
    # print("Inertia values for different numbers of clusters:")
    # for n, inertia in zip(n_clusters_range, inertias):
    #     print(f"n_clusters={n}: inertia={inertia:.2f}")
    #
    # # Run KMeans with the optimal number of clusters (you can adjust this based on the elbow plot)
    # optimal_n_clusters = 3  # This can be adjusted based on the elbow plot
    # kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42)
    # cluster_labels = kmeans.fit_predict(final_np)
    #
    # # Print cluster sizes
    # unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    # print("\nCluster sizes:")
    # for label, count in zip(unique_labels, counts):
    #     print(f"Cluster {label}: {count} points")




def lowDapprox_test(cfg):
    omegaconf_resolvers()

    cfg.savepath = os.path.join(project_path, cfg.savepath)

    from odeint_utils import run_odeint_to_final
    from torch import nn

    dynamics_function = instantiate(cfg.dynamics.function)
    distribution = instantiate(cfg.dynamics.IC_distribution_fit)[1]
    # distribution = instantiate(cfg.dynamics.IC_distribution_task_relevant)
    # distribution = instantiate(cfg.dynamics.IC_interpolation_line_2)

    device = 'cpu'

    # print(traj.shape)

    compressed_dim = 20
    T = 30
    original_dim = cfg.dynamics.dim

    encoder = nn.Linear(in_features=original_dim, out_features=compressed_dim)
    small_dynamics_func = nn.Sequential(
        nn.Linear(in_features=compressed_dim, out_features=compressed_dim),
        nn.Tanh(),
        nn.Linear(in_features=compressed_dim, out_features=compressed_dim)
    )
    decoder = nn.Linear(in_features=compressed_dim, out_features=original_dim)

        # Move models to device
    encoder = encoder.to(device)
    small_dynamics_func = small_dynamics_func.to(device)
    decoder = decoder.to(device)

    # # Create optimizer for all models
    # optimizer = torch.optim.Adam([
    #     {'params': encoder.parameters()},
    #     {'params': small_dynamics_func.parameters()},
    #     {'params': decoder.parameters()}
    # ], lr=1e-1)
    #
    # # Pre-training without ODE integration
    # num_pretrain_epochs = 500
    # pretrain_batch_size = 500
    #
    # # Sample initial conditions and run ODE integration
    # x_batch = distribution.sample(sample_shape=(pretrain_batch_size,))
    # x_batch = run_odeint_to_final(
    #     dynamics_function,
    #     x_batch,
    #     inputs=torch.tensor([0.0, 0.0, 0.9]),
    #     T=30,
    #     steps=30,
    #     return_last_only=True
    # )
    #
    # for pretrain_epoch in range(num_pretrain_epochs):
    #     optimizer.zero_grad()
    #
    #     # Sample batch from distribution
    #     # x_batch = distribution.sample(sample_shape=(pretrain_batch_size,))
    #
    #     # Forward pass through the full model
    #     encoded = encoder(x_batch)
    #     compressed = small_dynamics_func(encoded)
    #     decoded = decoder(compressed)
    #
    #     # Compute target using original dynamics
    #     target = dynamics_function(x_batch)
    #
    #     # Calculate MSE loss
    #     pretrain_loss = torch.nn.functional.mse_loss(decoded, target)
    #
    #     # Backpropagate
    #     pretrain_loss.backward()
    #     optimizer.step()
    #
    #     if pretrain_epoch % 10 == 0:
    #         print(f"Pre-train Epoch {pretrain_epoch}, Loss: {pretrain_loss.item()}")

    # Create optimizer for all models
    optimizer = torch.optim.Adam([
        {'params': encoder.parameters()},
        {'params': small_dynamics_func.parameters()},
        {'params': decoder.parameters()}
    ], lr=1e-2)


    initial_conditions = distribution.sample(sample_shape=(200,))
    traj = run_odeint_to_final(
        dynamics_function,
        initial_conditions,
        inputs=torch.tensor([0.0, 0.0, 0.9]),
        T=T,
        steps=300,
        return_last_only=False
    )

    # Training loop
    num_epochs = 1 #30 #30 #00
    for epoch in range(num_epochs):
        optimizer.zero_grad()


        # Encode initial conditions
        encoded_ic = encoder(initial_conditions)

        # Run dynamics in compressed space
        compressed_traj = run_odeint_to_final(
            small_dynamics_func,
            encoded_ic,
            T=T,
            steps=300,
            return_last_only=False,
            no_grad=False,
        )

        # Decode trajectories back to original space
        decoded_traj = decoder(compressed_traj)

        # Calculate MSE loss
        loss = torch.nn.functional.mse_loss(decoded_traj, traj)

        # Backpropagate
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    # Test the trained models
    with torch.no_grad():
        encoded_ic = encoder(initial_conditions)
        compressed_traj = run_odeint_to_final(
            small_dynamics_func,
            encoded_ic,
            T=T,
            steps=300,
            return_last_only=False
        )
        decoded_traj = decoder(compressed_traj)

        # Concatenate trajectories for PCA
        combined_traj = torch.cat([traj, decoded_traj], dim=0)
        combined_traj_reshaped = combined_traj.reshape(-1, combined_traj.shape[-1])

        # Perform PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(combined_traj_reshaped.detach().cpu().numpy())

        # Split back into original and reconstructed
        n_points = traj.shape[0] * traj.shape[1]
        original_pca = pca_result[:n_points].reshape(traj.shape[0], traj.shape[1], 2)
        reconstructed_pca = pca_result[n_points:].reshape(traj.shape[0], traj.shape[1], 2)

        # Plot PCA results
        plt.figure(figsize=(10, 5))
        for i in range(traj.shape[1]):
            plt.plot(original_pca[:, i, 0], original_pca[:, i, 1], alpha=0.1, label='Original' if i == 0 else None, c='C0')
            # plt.plot(reconstructed_pca[:, i, 0], reconstructed_pca[:, i, 1], alpha=0.5, label='Reconstructed' if i == 0 else None, c='C1')
        plt.scatter(original_pca[-1, :, 0], original_pca[-1, :, 1], alpha=1, c='red')
        plt.title('PCA of Original vs Reconstructed Trajectories')
        plt.legend()
        plt.show()

    # Save the trained models in separate files
    torch.save(encoder.state_dict(), 'encoder.pth')
    torch.save(decoder.state_dict(), 'decoder.pth')
    torch.save(small_dynamics_func.state_dict(), 'small_dynamics_func.pth')

    # Save optimizer state separately
    torch.save(optimizer.state_dict(), 'optimizer.pth')

def test_run_GD(cfg):
    omegaconf_resolvers()

    from learn_koopman_eig import runGD_basic
    hidden = torch.randn(size=(10,1))
    def f(x):
        return torch.abs(x)

    traj, below_thr_points = runGD_basic(
        f,
        initial_conditions=hidden.clone(),
        save_trajectories_every=1,
        threshold=1e-2
    )
    # traj = traj.detach().cpu().numpy()
    plt.plot(traj[:,:,0])
    plt.scatter(torch.zeros_like(hidden),hidden)
    plt.scatter(torch.zeros_like(below_thr_points), below_thr_points,marker='x')
    plt.show()

def plot_ODE_line_IC(cfg):
    omegaconf_resolvers()
    cfg.savepath = os.path.join(project_path, cfg.savepath)

    dynamics_function = instantiate(cfg.dynamics.function)

    attractors = instantiate(cfg.dynamics.attractors)

    # Create 100 points along the line joining the attractors
    num_points = 100
    T = 25
    steps = 100
    times = torch.linspace(0, T, steps)
    alpha = torch.linspace(0, 1, num_points)[:, None]
    line_points = attractors[0] * (1 - alpha) + attractors[1] * alpha

    from odeint_utils import run_odeint_to_final

    # Run dynamics for each point
    trajectories = run_odeint_to_final(
        dynamics_function,
        line_points,
        T,
        inputs = instantiate(cfg.dynamics.static_external_input),
        steps = steps,
        return_last_only=False
    )
    trajectories = trajectories.detach().cpu().numpy()

    # Perform PCA on trajectories
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca_trajectories = pca.fit_transform(trajectories.reshape(-1, trajectories.shape[-1]))
    pca_trajectories = pca_trajectories.reshape(trajectories.shape[0], -1, 2)

    # Plot PC1 against time
    plt.figure(figsize=(10, 5))
    # for i in range(num_points):
    plt.plot(times[:,None].repeat(1,pca_trajectories.shape[1]),pca_trajectories[:, :, 0])
    plt.xlabel('Time')
    plt.ylabel('PC1')
    plt.title('PC1 vs Time for Trajectories')
    # plt.legend()
    # plt.show()
    plt.savefig(Path(cfg.savepath) / 'PC1_line_trajectories.png')

def test_RNN(cfg):
    omegaconf_resolvers()

    # hidden = instantiate(cfg.dynamics.hidden_full)
    # print(hidden)
    # attractors = instantiate(cfg.dynamics.attractors)
    print(instantiate(cfg.dynamics.point_on_separatrix))

def finkelstein_fontolan_analysis_test(cfg):
    omegaconf_resolvers()

    cfg.savepath = os.path.join(project_path, cfg.savepath)


    if 'finkelstein_fontolan' in cfg.dynamics.name:
        dynamics_function = instantiate(cfg.dynamics.function)
        attractors = instantiate(cfg.dynamics.attractors2)

        from interpolation import cubic_hermite

        num_points = 40
        rand_scale = 3.1 #10.0
        # Example usage
        x,y = attractors.detach().cpu().numpy()#
        # x = np.array([0, 0])  # Start point
        # y = np.array([1, 1])  # End point
        m_x = -x + y  # Tangent at x
        m_y = -x + y  # Tangent at y

        num_curves = 20 #20  # Number of random curves to generate
        plt.figure(figsize=(10, 10))

        # Accumulate all points
        all_points = []
        for _ in range(num_curves):
            # Generate random perturbations for tangents
            m_x_perturbed = m_x * 1.4 + np.random.randn(m_x.shape[0]) * rand_scale
            m_y_perturbed = m_y * 1.4 + np.random.randn(m_x.shape[0]) * rand_scale

            # Generate points on the cubic Hermite curve
            points = cubic_hermite(x, y, m_x_perturbed, m_y_perturbed, num_points)
            all_points.append(points)

        # Stack all points into a single array
        all_points = np.vstack(all_points)

        # Perform PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        pca_points = pca.fit_transform(all_points)

        # Convert points to torch tensor and run ODE integration
        points_tensor = torch.tensor(all_points, dtype=torch.float32)
        time_points = torch.linspace(0, 1000, 20)  # Adjust time range and steps as needed

        ext_input = torch.tensor([0.0,0.0,0.91]).type_as(points_tensor)
        ext_input = ext_input[None]
        ext_input = ext_input.repeat(all_points.shape[0],1)

        # Run ODE integration for all points
        with torch.no_grad():
            trajectories = odeint(lambda t, x: dynamics_function(x,ext_input), points_tensor, time_points)

        # Convert trajectories to numpy and reshape for PCA
        trajectories_np = trajectories.detach().cpu().numpy()
        trajectories_reshaped = trajectories_np.reshape(-1, trajectories_np.shape[-1])


        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=2, random_state=42)
        cluster_labels = kmeans.fit_predict(trajectories_np[-1])

        # Plot PCA results
        plt.figure(figsize=(10, 10))
        plt.scatter(pca_points[:, 0], pca_points[:, 1], c=['C0' if label == 0 else 'C1' for label in cluster_labels], alpha=1.0)

        # Plot endpoints in PCA space
        endpoints = np.array([x, y])
        pca_endpoints = pca.transform(endpoints)
        plt.scatter(pca_endpoints[:, 0], pca_endpoints[:, 1], color='red', label="Endpoints")

        plt.legend()
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('PCA of Cubic Hermite Interpolations')
        plt.grid(True)
        plt.savefig("test_plots/hermite_cubic_interpolations_plus_clustering.png")
        plt.show()

        print(points_tensor.shape, ext_input.shape)
        full_input_to_KEF = torch.concat([points_tensor, ext_input], axis=-1)
        full_input_to_KEF = full_input_to_KEF.reshape(num_curves, num_points, -1)

        cluster_labels = kmeans.fit_predict(trajectories_np[-1])
        cluster_labels = cluster_labels.reshape(num_curves, num_points)
        changes = np.diff(cluster_labels, axis=1) != 0
        change_points = np.argmax(changes, axis=1)
        # Handle cases where there are no changes (all zeros or all ones)
        no_changes = ~np.any(changes, axis=1)
        change_points[no_changes] = num_points // 2

        change_points = np.linspace(0,1, num_points)[change_points]

        # Plot the change points
        fig = plt.figure()
        plt.hist(change_points, bins=num_points, alpha=0.7)
        plt.xlabel('Point Index')
        plt.ylabel('Number of Curves')
        plt.title('Distribution of Change Points')
        plt.grid(True)
        fig.savefig(Path(cfg.savepath) / "change_points_distribution.png", dpi=300)
        plt.show()


def finkelstein_fontolan_point_finder_test(cfg):
    omegaconf_resolvers()

    cfg.savepath = os.path.join(project_path, cfg.savepath)
    dynamics_function = instantiate(cfg.dynamics.function)

    attractors = instantiate(cfg.dynamics.attractors2)

    from separatrix_point_finder import find_separatrix_point_along_line

    # Find separatrix point along line between attractors
    separatrix_point = find_separatrix_point_along_line(
        dynamics_function=dynamics_function,
        external_input=torch.tensor([0.0,0.0,0.90]),
        attractors=attractors,
        num_points=20,
        num_iterations=5
    )

    ####

    # Sample points near separatrix point with different noise scales
    num_samples = 100
    noise_scales = np.logspace(-7,0,15)
    class_ratios = []
    all_final_points = []
    all_initial_points = []

    for noise_scale in noise_scales:
        initial_points = separatrix_point + torch.randn(num_samples, separatrix_point.shape[0]) * noise_scale
        all_initial_points.append(initial_points)

        # Run trajectories from these points
        time_points = torch.linspace(0, 1000, 2)
        with torch.no_grad():
            trajectories = odeint(
                lambda t, x: dynamics_function(x, torch.tensor([0.0,0.0,0.90])[None].repeat(num_samples,1)),
                initial_points,
                time_points
            ).detach().cpu()

        # Get final points
        final_points = trajectories[-1]
        all_final_points.append(final_points)

    # Combine all final points and perform k-means clustering
    combined_final_points = torch.cat(all_final_points, dim=0)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, random_state=0).fit(combined_final_points)
    labels = kmeans.labels_

    # Calculate class ratios for each noise scale
    start_idx = 0
    for i, noise_scale in enumerate(noise_scales):
        end_idx = start_idx + num_samples
        scale_labels = labels[start_idx:end_idx]
        class_ratio = np.sum(scale_labels == 0) / len(scale_labels)
        class_ratios.append(class_ratio)
        start_idx = end_idx

    # Plot class ratios vs noise scales
    plt.figure(figsize=(8, 6))
    plt.plot(noise_scales, class_ratios, 'o-')
    plt.xlabel('Noise Scale')
    plt.ylabel('Class Ratio (Cluster 0 / Total)')
    plt.title('Class Ratio vs Noise Scale')
    plt.grid(True)
    plt.xscale('log')
    plt.show()

    ######

    # # Sample points near separatrix point with Gaussian noise
    # num_samples = 100
    # noise_scale = 0.1
    # initial_points = separatrix_point + torch.randn(num_samples, separatrix_point.shape[0]) * noise_scale

    # # Run trajectories from these points
    # time_points = torch.linspace(0, 5000, 200)
    # with torch.no_grad():
    #     trajectories = odeint(
    #         lambda t, x: dynamics_function(x, torch.tensor([0.0,0.0,0.90])[None].repeat(num_samples,1)),
    #         initial_points,
    #         time_points
    #     ).detach().cpu()

    # # Reshape trajectories for PCA (flatten time dimension)
    # trajectories_reshaped = trajectories.reshape(-1, trajectories.shape[-1])

    # # Perform PCA
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=2)
    # pca_trajectories = pca.fit_transform(trajectories_reshaped)

    # # Reshape back to separate trajectories
    # pca_trajectories = pca_trajectories.reshape(trajectories.shape[0], trajectories.shape[1], 2)

    # # Plot PCA trajectories
    # plt.figure(figsize=(8, 8))
    # plt.plot(pca_trajectories[..., 0], pca_trajectories[..., 1], 'b-', alpha=0.1)
    # plt.scatter(pca_trajectories[0, 0, 0], pca_trajectories[0, 0, 1], c='r', label='Initial Points')
    # plt.scatter(pca_trajectories[0, -1, 0], pca_trajectories[0, -1, 1], c='g', label='Final Points')
    # plt.title('PCA of Trajectories Near Separatrix')
    # plt.xlabel('PC1')
    # plt.ylabel('PC2')
    # plt.legend()
    # plt.show()

    # print(f"Found separatrix point at: {separatrix_point}")


def finkelstein_fontolan(cfg):
    omegaconf_resolvers()

    cfg.savepath = os.path.join(project_path, cfg.savepath)
    dynamics_function = instantiate(cfg.dynamics.function)

    attractors = instantiate(cfg.dynamics.attractors2)

    def cubic_hermite(x, y, m_x, m_y, num_points=100):
        """
        Generate points on a cubic Hermite curve joining points x and y with tangents m_x and m_y.

        Arguments:
        - x: Starting point of the curve.
        - y: Ending point of the curve.
        - m_x: Tangent vector at the starting point.
        - m_y: Tangent vector at the ending point.
        - num_points: Number of points to generate on the curve.

        Returns:
        - points: Array of points on the cubic Hermite curve.
        """
        alpha = np.linspace(0, 1, num_points)

        # Cubic Hermite interpolation formula
        points = (2 * alpha ** 3 - 3 * alpha ** 2 + 1)[:, np.newaxis] * x + \
                 (-2 * alpha ** 3 + 3 * alpha ** 2)[:, np.newaxis] * y + \
                 (alpha ** 3 - 2 * alpha ** 2 + alpha)[:, np.newaxis] * m_x + \
                 (alpha ** 3 - alpha ** 2)[:, np.newaxis] * m_y

        return points

    num_points = 40
    rand_scale = 4.1 #10.0
    # Example usage
    x,y = attractors.detach().cpu().numpy()#
    # x = np.array([0, 0])  # Start point
    # y = np.array([1, 1])  # End point
    m_x = -x + y  # Tangent at x
    m_y = -x + y  # Tangent at y

    num_curves = 20 #20  # Number of random curves to generate
    plt.figure(figsize=(10, 10))

    # Accumulate all points
    all_points = []
    for _ in range(num_curves):
        # Generate random perturbations for tangents
        m_x_perturbed = m_x * 1.4 + np.random.randn(m_x.shape[0]) * rand_scale
        m_y_perturbed = m_y * 1.4 + np.random.randn(m_x.shape[0]) * rand_scale

        # Generate points on the cubic Hermite curve
        points = cubic_hermite(x, y, m_x_perturbed, m_y_perturbed, num_points)
        all_points.append(points)

    # Stack all points into a single array
    all_points = np.vstack(all_points)

    # Perform PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca_points = pca.fit_transform(all_points)

    # Plot PCA results
    plt.figure(figsize=(10, 10))
    plt.plot(pca_points[:, 0], pca_points[:, 1], alpha=0.3)

    # Plot endpoints in PCA space
    endpoints = np.array([x, y])
    pca_endpoints = pca.transform(endpoints)
    plt.scatter(pca_endpoints[:, 0], pca_endpoints[:, 1], color='red', label="Endpoints")

    plt.legend()
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of Cubic Hermite Interpolations')
    plt.grid(True)
    plt.show()


    # Convert points to torch tensor and run ODE integration
    points_tensor = torch.tensor(all_points, dtype=torch.float32)
    time_points = torch.linspace(0, 1000, 20)  # Adjust time range and steps as needed

    ext_input = torch.tensor([0.0,0.0,0.91]).type_as(points_tensor)
    ext_input = ext_input[None]
    ext_input = ext_input.repeat(all_points.shape[0],1)

    # Run ODE integration for all points
    with torch.no_grad():
        trajectories = odeint(lambda t, x: dynamics_function(x,ext_input), points_tensor, time_points)
    # Convert trajectories to numpy and reshape for PCA
    trajectories_np = trajectories.detach().cpu().numpy()
    trajectories_reshaped = trajectories_np.reshape(-1, trajectories_np.shape[-1])


    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, random_state=42)
    cluster_labels = kmeans.fit_predict(trajectories_np[-1])

    # Plot PCA results
    plt.figure(figsize=(10, 10))
    plt.scatter(pca_points[:, 0], pca_points[:, 1], c=['C0' if label == 0 else 'C1' for label in cluster_labels], alpha=1.0)

    # Plot endpoints in PCA space
    endpoints = np.array([x, y])
    pca_endpoints = pca.transform(endpoints)
    plt.scatter(pca_endpoints[:, 0], pca_endpoints[:, 1], color='red', label="Endpoints")

    plt.legend()
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of Cubic Hermite Interpolations')
    plt.grid(True)
    plt.savefig("test_plots/hermite_cubic_interpolations_plus_clustering.png")
    plt.show()

    # # Perform PCA on trajectories using the same PCA object as before
    # pca_trajectories = pca.transform(trajectories_reshaped)
    # pca_trajectories = pca_trajectories.reshape(trajectories_np.shape[0], trajectories_np.shape[1], -1)

    # # Plot trajectories in PCA space
    # plt.figure(figsize=(10, 10))
    # for i in range(len(all_points)):
    #     plt.plot(pca_trajectories[:, i, 0],
    #             pca_trajectories[:, i, 1],
    #             alpha=0.3,
    #              c=f'C{cluster_labels[i]}')
    #     plt.scatter(pca_trajectories[-1, i, 0],
    #              pca_trajectories[-1, i, 1],
    #              alpha=0.3,marker='x')
    # plt.scatter(pca_endpoints[:, 0], pca_endpoints[:, 1], color='red', label="Endpoints")
    # plt.legend()
    # plt.xlabel('Principal Component 1')
    # plt.ylabel('Principal Component 2')
    # plt.title('Trajectories from Cubic Hermite Interpolations')
    # plt.grid(True)
    # plt.show()



def main_multimodel(cfg):
    """
    Uses the SeparatrixLocator class.
    """
    omegaconf_resolvers()
    cfg.savepath = os.path.join(project_path, cfg.savepath)


    # OmegaConf.resolve(cfg.model)

    print(OmegaConf.to_yaml(cfg))

    dynamics_function = instantiate(cfg.dynamics.function)
    input_distribution = instantiate(cfg.dynamics.external_input_distribution) if hasattr(cfg.dynamics, 'external_input_distribution') else None
    distribution = instantiate(cfg.dynamics.IC_distribution)
    if hasattr(cfg.dynamics, 'IC_distribution_fit'):
        distribution_fit = instantiate(cfg.dynamics.IC_distribution_fit)
    else:
        distribution_fit = distribution

    if hasattr(cfg.dynamics, 'external_input_distribution_fit'):
        input_distribution_fit = instantiate(cfg.dynamics.external_input_distribution_fit)
    else:
        input_distribution_fit = input_distribution

    if input_distribution is not None:
        cfg.model.input_size = cfg.dynamics.dim + cfg.dynamics.external_input_dim
        OmegaConf.resolve(cfg.model)

    if hasattr(cfg, 'dmd'):
        num_trajectories = 100
        time_horizon = 10
        num_time_steps = 20
        # Sample initial conditions from the distribution
        initial_conditions = distribution.sample(sample_shape=(num_trajectories,))

        # Define time points for the trajectory
        time_points = torch.linspace(0, time_horizon, num_time_steps)

        # Function to run trajectories using the dynamics function
        trajectories = odeint(lambda t,x: dynamics_function(x), initial_conditions, time_points).detach().cpu()

        from pydmd import DMD, EDMD
        from pydmd.plotter import plot_eigs, plot_summary
        from pydmd.preprocessing import hankel_preprocessing

        X = trajectories[:,0] #.numpy()
        d = 3
        dmd = DMD(svd_rank=4)
        delay_dmd = hankel_preprocessing(dmd, d=d)
        delay_dmd.fit(X.T)
        plot_summary(delay_dmd, x=X[:,0], t=time_points, d=d)

        X = trajectories.permute(1, 0, 2).reshape(-1, trajectories.shape[-1])  # Reshape to (batch*time_steps, dimensions)
        dmd = EDMD(svd_rank=4)
        dmd.fit(X.T)
        x_grid = torch.linspace(-3, 3, 10).detach().cpu().numpy()  # Generate a grid of points
        eigenfunctions_values = dmd.eigenfunctions(x_grid)
        plt.plot(x_grid.numpy(), eigenfunctions_values)
        plt.show()
        # plot_summary(dmd, x=X[:, 0], t=time_points, d=d)


    SL = instantiate(cfg.separatrix_locator)
    SL.models = [instantiate(cfg.model).to(SL.device) for _ in range(cfg.separatrix_locator.num_models)]

    if hasattr(cfg, 'classifier_based_separatrix_locator'):
        CSL = instantiate(cfg.classifier_based_separatrix_locator)
        print("Models:",CSL.models)
        CSL.fit(
            dynamics_function,
            distribution,
            **instantiate(cfg.classifier_based_separatrix_locator_fit_kwargs)
        )
        scores = CSL.score(
            dynamics_function,
            distribution
        )
        print('Separatrix classifer scores',scores)


        if cfg.dynamics.dim == 2:
            num_samples = 5000
            samples = distribution.sample(
                sample_shape=(num_samples,))
            labels = CSL.models[0].predict(samples.cpu().numpy())

            # Plot the generated samples colored by their kmeans label
            plt.figure(figsize=(5,5))
            for i in range(num_samples):
                plt.scatter(samples[i, 0].cpu().numpy(), samples[i, 1].cpu().numpy(), color='C' + str(labels[i]),
                            alpha=0.5,s=3)  # Color by label
            plt.title('Generated Samples Colored by SVM Label')
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            plt.savefig(Path(cfg.savepath) / "classifier_predicted_labels.png", dpi=200)

    if cfg.load_KEF_model:
        new_format_path = Path(cfg.savepath)/cfg.experiment_details
        print(new_format_path)
        if os.path.exists(new_format_path):
            load_path = new_format_path
        else:
            load_path = Path(cfg.savepath)
            print(new_format_path, 'does not exist, loading', load_path, 'instead.')
        SL.load_models(load_path)

    import time
    
    start_time = time.time()
    SL.fit(
        dynamics_function,
        distribution_fit,
        external_input_dist=input_distribution_fit,
        dist_weights = cfg.dynamics.dist_weights if hasattr(cfg.dynamics,"dist_weights") else None,
        **instantiate(cfg.separatrix_locator_fit_kwargs),
    )
    end_time = time.time()
    print(f"SL.fit took {end_time - start_time:.2f} seconds")

    if cfg.save_KEF_model:
        SL.save_models(Path(cfg.savepath)/cfg.experiment_details,filename=cfg.dynamics.special_model_name if hasattr(cfg.dynamics,'special_model_name') else None)

    SL.to('cpu')

    scores = SL.score(
        dynamics_function,
        distribution_fit,
        external_input_dist=input_distribution,
        **instantiate(cfg.separatrix_locator_score_kwargs)
    )
    # scales = [d.covariance_matrix[0,0] for d in distribution_fit]
    print('Scores:\n', scores.detach().cpu().numpy())
    # print(scales)
    if hasattr(cfg,'separatrix_locator_score_kwargs_2'):
        scores2 = SL.score(
            dynamics_function,
            distribution,
            external_input_dist=input_distribution,
            **instantiate(cfg.separatrix_locator_score_kwargs_2)
        )
        print('Scores over 2x scaled distribution:\n',scores2.detach().cpu().numpy())


    if hasattr(cfg,'save_results') and cfg.save_results:
        scores = scores[...,0].detach().cpu().numpy()
        # Save scores to CSV with descriptive column names
        scores_df = pd.DataFrame(scores)
        scores_df.columns = [f'distribution_{i+1}' for i in range(scores.shape[1])]
        scores_df.index = [f'model_{i+1}' for i in range(scores.shape[0])]
        scores_df.index.name = 'model'
        scores_df.to_csv(Path(cfg.savepath) / cfg.experiment_details / 'model_scores.csv')


    # if hasattr(cfg.dynamics,'analytical_eigenfunction'):
    #     num_samples = 5000
    #     sampled_points = distribution.sample(sample_shape=(num_samples,))
    #
    #     analytical_eigenfunction = instantiate(cfg.dynamics.analytical_eigenfunction)
    #     analytical_values = analytical_eigenfunction(sampled_points)
    #
    #     correlations = []
    #     for model in SL.models:
    #         model_values = model(sampled_points).detach()
    #         x,y = model_values.flatten(), analytical_values.flatten()
    #         x = torch.abs(x)
    #         y = torch.abs(y)
    #         correlation = torch.sum(x*y)/torch.sqrt(torch.sum(x**2)*torch.sum(y**2))
    #         correlations.append(correlation)
    #
    #     print("Correlations between models and analytical eigenfunction:", correlations)

    all_below_threshold_points = None
    if cfg.runGD:
        external_inputs = None
        if input_distribution is not None:
            external_inputs = input_distribution.sample(sample_shape=(cfg.separatrix_find_separatrix_kwargs.batch_size,))
        _, all_below_threshold_points = SL.find_separatrix(
            distribution,
            external_inputs = external_inputs,
            dist_needs_dim = cfg.dynamics.dist_requires_dim if hasattr(cfg.dynamics,"dist_requires_dim") else True,
            **instantiate(cfg.separatrix_find_separatrix_kwargs)
        )
        print('all_below_threshold_points',all_below_threshold_points)

    if cfg.run_fixed_point_finder:
        assert hasattr(cfg.dynamics,"loaded_RNN_model")
        rnn_model = instantiate(cfg.dynamics.loaded_RNN_model) #(device=SL.device)
        # cfg.dynamics.RNN_dataset.batch_size = 5000
        # cfg.dynamics.RNN_dataset.n_trials = 1000
        dataset = instantiate(cfg.dynamics.RNN_dataset)
        inp, targ = dataset()

        torch_inp = torch.from_numpy(inp).type(torch.float)  # .to(device)
        outputs, hidden_traj = rnn_model(torch_inp, return_hidden=True) #, deterministic=False)
        outputs, hidden_traj = outputs.detach().cpu().numpy(), hidden_traj.detach().cpu().numpy()


        FPF = FixedPointFinderTorch(
            rnn_model.rnn if hasattr(rnn_model,"rnn") else rnn_model,
            **instantiate(cfg.fpf_hps)
        )
        num_trials = 500
        # initial_conditions = dist.sample(sample_shape=(num_trials,)).detach().cpu().numpy()
        # inputs = np.zeros((1, cfg.dynamics.RNN_model.act_size))
        # inputs[...,2] = 1.0
        torch_inp[...,:2] = 0.0
        fp_inputs = torch_inp.reshape(-1, torch_inp.shape[-1]).detach().cpu().numpy()

        # inputs[...,0] = 1
        initial_conditions = hidden_traj.reshape(-1, hidden_traj.shape[-1])
        select = np.random.choice(initial_conditions.shape[0], size=num_trials, replace=False)
        initial_conditions = initial_conditions[select]
        fp_inputs = fp_inputs[select]
        # fp_inputs[:,:2] = 0
        initial_conditions += np.random.normal(size=initial_conditions.shape) * 2.0 #0.5 #2.0
        # print('initial_conditions', initial_conditions.shape)
        unique_fps, all_fps = FPF.find_fixed_points(
            deepcopy(initial_conditions),
            fp_inputs
        )

        # print(all_fps.shape)
        KEF_val_at_fp = {}
        for i in range(SL.num_models):
            # below_threshold_points = all_below_threshold_points[i] if all_below_threshold_points is not None else None
            mod_model = compose(
                torch.log,
                lambda x: x + 1,
                torch.exp,
                partial(torch.sum, dim=-1, keepdims=True),
                torch.log,
                torch.abs,
                SL.models[i]
            )
            # KEF_val_at_fp[f'KEF{i}'] = mod_model(torch.from_numpy(unique_fps.xstar).to(SL.device)).detach().cpu().numpy().flatten()

        fixed_point_data = {
            'stability': unique_fps.is_stable,
            'q': unique_fps.qstar,
            'x0' : unique_fps.xstar[...,0],
            'x1': unique_fps.xstar[...,1],
        }
        fixed_point_data.update(KEF_val_at_fp)
        # print(fixed_point_data)
        fixed_point_data = pd.DataFrame(fixed_point_data)
        fixed_point_data.to_csv(Path(cfg.savepath) / 'fixed_point_data.csv', index=False)

    if cfg.run_analysis:

        #### Plotting log prob vs KEF amplitude
        # num_samples = 1000
        # needs_dim = True
        # if hasattr(cfg.dynamics, 'dist_requires_dim'):
        #     needs_dim = cfg.dynamics.dist_requires_dim
        #
        # samples = distribution.sample(
        #     sample_shape=[num_samples] + ([cfg.dynamics.dim] if needs_dim else []))
        #
        # samples.requires_grad_(True)
        #
        # fig,axs = plt.subplots(2,1,sharex=True,figsize=(6,8))
        # for j in range(SL.num_models):
        #     mod_model = compose(
        #         lambda x: x.sum(axis=-1,keepdims=True),
        #         torch.log,
        #         torch.abs,
        #         SL.models[j]
        #     )
        #     log_probs = distribution.log_prob(samples).detach().cpu().numpy()
        #     phi_x = mod_model(samples.to(SL.device))#.detach().cpu().numpy()
        #     # print(log_probs.shape,phi_x.shape)
        #     # from learn_koopman_eig import eval_loss
        #     # losses = eval_loss(
        #     #     model,
        #     #     normaliser=lambda x,y:(x - y) ** 2
        #     # )
        #     # Compute phi'(x)
        #     phi_x_prime = torch.autograd.grad(
        #         outputs=phi_x,
        #         inputs=samples,
        #         grad_outputs=torch.ones_like(phi_x),
        #         create_graph=True
        #     )[0]
        #     # Compute F(x_batch)
        #     F_x = dynamics_function(samples)
        #
        #     # Main loss term: ||phi'(x) F(x) - phi(x)||^2
        #     dot_prod = (phi_x_prime * F_x).sum(axis=-1, keepdim=True)
        #     errors = torch.abs(dot_prod - phi_x).detach().cpu().numpy()
        #
        #     phi_x = phi_x.detach().cpu().numpy()
        #
        #     ax = axs[0]
        #     ax.scatter(np.repeat(log_probs[...,None],repeats=phi_x.shape[-1],axis=-1),np.abs(phi_x),s=10)
        #     ax.set_ylabel(r'$|$KEF$(x)|$')
        #     ax.set_yscale('log')
        #     ax.set_xlabel(r'$\log p(x)$')
        #
        #     ax = axs[1]
        #     ax.scatter(log_probs[..., None], errors, s=10)
        #     ax.set_ylabel('PDE error')
        #     ax.set_yscale('log')
        #     ax.set_xlabel(r'$\log p(x)$')
        # fig.tight_layout()
        # fig.savefig(Path(cfg.savepath) / 'log_prob_and_KEF_amplitude.png')



        if cfg.model.input_size == 1:
            pass
        elif cfg.model.input_size == 2:
            pass
            # fig,axs = plt.subplots(7,4,figsize=np.array([4,7])*2.3,sharey=True,sharex=True)
            # for j in range(SL.num_models):
            #     for i in range(cfg.model.output_size):
            #         mod_model = compose(
            #             lambda x: x**0.01,
            #             torch.log,
            #             lambda x: x + 1,
            #             torch.exp,
            #             # partial(torch.sum, dim=-1, keepdims=True),
            #             lambda x: x[...,i:i+1],
            #             torch.log,
            #             torch.abs,
            #             SL.models[j]
            #         )
            #         ax = axs[i,j]
            #
            #         x_limits = (-2, 2)  # Limits for x-axis
            #         y_limits = (-2, 2)  # Limits for y-axis
            #         if hasattr(cfg.dynamics, 'lims'):
            #             x_limits = cfg.dynamics.lims.x
            #             y_limits = cfg.dynamics.lims.y
            #         plot_model_contour(
            #             mod_model,
            #             ax,
            #             x_limits=x_limits,
            #             y_limits=y_limits,
            #         )
            #         below_threshold_points = all_below_threshold_points[j] if all_below_threshold_points is not None else None
            #         # print(below_threshold_points.shape)
            #         if below_threshold_points is not None:
            #             xlim = ax.get_xlim()  # Store current x limits
            #             ylim = ax.get_ylim()  # Store current y limits
            #
            #             ax.scatter(below_threshold_points[:, 0], below_threshold_points[:, 1], c='red', s=10)
            #
            #             ax.set_xlim(xlim)  # Reset x limits
            #             ax.set_ylim(ylim)  # Reset y limits
            #         # ax.set_aspect('auto')
            #         ax.set_title(f'Model-{j},output{i}'+'\n'+f", loss:{float(scores[j, i]):.5f}")
            #         ax.set_xlabel('')
            #         ax.set_ylabel('')
            #
            #         # ax.scatter(*unique_fps.xstar[unique_fps.is_stable, :].T, c='blue',marker='x',s=100,zorder=1001)
            #         # ax.scatter(*unique_fps.xstar[~unique_fps.is_stable, :].T, c='red',marker='x',s=100,zorder=1000)
            # fig.tight_layout()
            # fig.savefig(Path(cfg.savepath)/"all_KEF_contours.png",dpi=300)
            # plt.close(fig)
            #
            # x_limits = (-2, 2)  # Limits for x-axis
            # y_limits = (-2, 2)  # Limits for y-axis
            # if hasattr(cfg.dynamics, 'lims'):
            #     x_limits = cfg.dynamics.lims.x
            #     y_limits = cfg.dynamics.lims.y
            #
            # fig,ax = plt.subplots(1,1, figsize=(5,5))
            # plot_kinetic_energy(
            #     dynamics_function,
            #     ax,
            #     x_limits=x_limits,
            #     y_limits=y_limits,
            #     below_threshold_points = np.concatenate(all_below_threshold_points,axis=0) if all_below_threshold_points is not None else None,
            # )
            # if cfg.run_fixed_point_finder:
            #     ax.scatter(*unique_fps.xstar[unique_fps.is_stable, :].T, c='blue', marker='x', s=100, zorder=1001)
            #     ax.scatter(*unique_fps.xstar[~unique_fps.is_stable, :].T, c='red', marker='x', s=100, zorder=1000)
            # fig.tight_layout()
            # fig.savefig(Path(cfg.savepath) / "kinetic_energy.png",dpi=300)
            # plt.close(fig)

        if 'hypercube' in cfg.dynamics.name or 'bistable' in cfg.dynamics.name:
            # Number of models and number of random vectors
            num_models = SL.num_models
            num_positions = cfg.dynamics.dim
            num_positions = np.minimum(num_positions, 10)
            n_trials = 20
            num_random_vectors = 10

            # Create subplots with 10 rows and 10 columns
            fig, axs = plt.subplots(num_models, num_positions, figsize=(num_positions, num_models+1), sharex=True, sharey=True)

            # Iterate over each model
            for model_idx, model in enumerate(SL.models):
                # Iterate over each position for x
                for pos in range(num_positions):
                    ax = axs[model_idx, pos] if SL.num_models>1 else axs[pos]
                    # Generate multiple random vectors for n_1 to n_10
                    for trial in range(n_trials):
                        n_values = np.random.uniform(-1, 1, cfg.dynamics.dim)
                        x_values = np.linspace(-1.5, 1.5, 100)

                        # Create an array to store the results for this trial
                        trial_results = []

                        # Create a batch of input arrays with x values swept from -2 to 2
                        input_batch = np.tile(n_values, (len(x_values), 1))
                        input_batch[:, pos] = x_values

                        # Convert input_batch to torch tensor
                        input_tensor = torch.from_numpy(input_batch).float()

                        # Run the model on the input tensor
                        with torch.no_grad():
                            output = model(input_tensor)
                        # Store the results
                        trial_results = output[:, 0].tolist()  # Assuming single output

                        # Plot the results for this trial
                        ax.plot(x_values, trial_results, lw=1, alpha=0.5)

                    # Evaluate when all n_values are zero
                    zero_values = np.zeros(cfg.dynamics.dim)
                    zero_input_batch = np.tile(zero_values, (len(x_values), 1))
                    zero_input_batch[:, pos] = x_values

                    zero_input_tensor = torch.from_numpy(zero_input_batch).float()

                    with torch.no_grad():
                        zero_output = model(zero_input_tensor)

                    ax.plot(x_values, zero_output[:, 0].tolist(), lw=1, alpha=1, color='black')
            # Set labels for the first column and first row
            # Set titles for the top row and labels for the first column
            for pos in range(num_positions):
                ax = axs[0, pos] if SL.num_models>1 else axs[pos]
                ax.set_title(f'Position {pos}')
            for model_idx in range(len(SL.models)):
                ax = axs[model_idx, 0] if SL.num_models > 1 else axs[0]
                ax.set_ylabel(f'Model {model_idx}')

            plt.tight_layout()
            fig.savefig(Path(cfg.savepath) / cfg.experiment_details / "model_output_vs_x_positions.png", dpi=80)
            # fig.savefig(Path(cfg.savepath) / "model_output_vs_x_positions.pdf")
            plt.close(fig)

            # Define distributions to test
            distributions = distribution_fit
            dist_names = range(len(distribution_fit))#[f"N(0,{d.dist.scale})" for d in cfg.dynamics.IC_distribution_fit]
            # colors = ['g', 'b', 'y', 'purple']

            batch_size = 1000
            for dist, name in zip(distributions, dist_names):
                # Sample input_tensor from the distribution
                input_tensor = dist.sample(sample_shape=(batch_size,))
                # input_tensor = input_tensor - 0.5
                input_tensor.requires_grad_(True)

                # Compute phi(x)
                phi_x = model(input_tensor)

                # Compute phi'(x) using autograd
                phi_x_prime = torch.autograd.grad(
                    outputs=phi_x,
                    inputs=input_tensor,
                    grad_outputs=torch.ones_like(phi_x),
                    create_graph=True
                )[0]

                F_x = dynamics_function(input_tensor)

                # Compute the dot product
                dot_prod = (phi_x_prime * F_x).sum(axis=-1, keepdim=True)

                residual = torch.abs(phi_x - dot_prod)

                # Plot KEF values vs residuals
                fig, ax = plt.subplots()
                ax.scatter(torch.abs(phi_x).detach().cpu().numpy(), residual.detach().cpu().numpy(),
                           label=name, alpha=0.5)
                ax.set_xlabel(r'$\psi(x)$')
                ax.set_ylabel(r'$|\nabla \psi(x) \cdot f(x) - \lambda\psi(x)|$')
                ax.legend()
                fig.tight_layout()
                fig.savefig(Path(cfg.savepath) / cfg.experiment_details / f"KEFvals_vs_residuals_{name}.png", dpi=200)
                plt.close(fig)

                # Plot standard deviation vs residuals
                std_i = torch.std(input_tensor.detach(), axis=-1, keepdims=True)
                fig, ax = plt.subplots()
                ax.scatter(std_i.cpu().numpy(), residual.detach().cpu().numpy(),
                          label=name, s=5, alpha=0.5)
                ax.set_xlabel(r'$std(x_i)$')
                ax.set_ylabel(r'$|\nabla \psi(x) \cdot f(x) - \lambda\psi(x)|$')
                ax.legend()
                fig.tight_layout()
                fig.savefig(Path(cfg.savepath) / cfg.experiment_details / f"Stdi_vs_residuals_{name}.png", dpi=200)
                plt.close(fig)

                # Plot LHS vs RHS
                fig, ax = plt.subplots()
                ax.scatter(phi_x.detach().cpu().numpy(), dot_prod.detach().cpu().numpy(),
                             label=name, alpha=0.5)
                ax.set_xlabel(r'$\lambda\psi(x)$')
                ax.set_ylabel(r'$\nabla \psi(x) \cdot f(x)$')
                ax.legend()

                # Add a dashed line for x=y
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                ax.plot([xlim[0], xlim[1]], [xlim[0], xlim[1]], color='black', linestyle='dashed', linewidth=1)

                fig.tight_layout()
                fig.savefig(Path(cfg.savepath) / cfg.experiment_details / f"KEF_LHS_RHS_{name}.png", dpi=200)
                plt.close(fig)

        # elif cfg.dynamics.dim > 2:
        #     # pass
        #     num_trials = 50
        #     times = torch.linspace(0, 500, 100)
        #     needs_dim = True
        #     if hasattr(cfg.dynamics, 'dist_requires_dim'):
        #         needs_dim = cfg.dynamics.dist_requires_dim
        #
        #     distribution_relevant = instantiate(cfg.dynamics.IC_distribution_task_relevant)
        #
        #     initial_conditions = distribution_relevant.sample(
        #         sample_shape=[num_trials] + ([cfg.dynamics.dim] if needs_dim else []))
        #     external_inputs = input_distribution.sample(sample_shape=[num_trials] + ([cfg.dynamics.dim] if needs_dim else []))
        #
        #     trajectories = odeint(lambda t, x: dynamics_function(x, external_inputs), initial_conditions, times)
        #     trajectories = trajectories.detach().cpu().numpy()
        #
        #     # Instantiate the dataset
        #     dataset = instantiate(cfg.dynamics.RNN_analysis_dataset)
        #     inputs, targets = dataset()
        #     inputs = torch.from_numpy(inputs).type(torch.float)
        #
        #     inputs = inputs
        #
        #     # Run trajectories using rnn and inputs from dataset
        #     rnn = instantiate(cfg.dynamics.loaded_RNN_model)
        #     outputs, hidden_trajectories = rnn(inputs, return_hidden=True, deterministic=False)
        #     hidden_trajectories = hidden_trajectories.detach().cpu().numpy()
        #
        #     # Perform another odeint run using the first time step from hidden_trajectories as the initial conditions
        #     hidden_initial_conditions = torch.from_numpy(hidden_trajectories[0]).type(torch.float)
        #     external_inputs_last_step = inputs[-1000]  # Use the last time step of inputs as external inputs
        #     new_trajectories = odeint(lambda t, x: dynamics_function(x, external_inputs_last_step), hidden_initial_conditions, times)
        #     new_trajectories = new_trajectories.detach().cpu().numpy()
        #
        #
        #     # Option to fit PCA on one set of trajectories, the other, or both
        #     fit_on = 'both'  # Options: 'hidden', 'trajectories', 'both'
        #
        #     if fit_on == 'hidden':
        #         pca = PCA(n_components=2)
        #         hidden_trajectories_reshaped = hidden_trajectories.reshape(-1, hidden_trajectories.shape[-1])
        #         pca.fit(hidden_trajectories_reshaped)
        #     elif fit_on == 'trajectories':
        #         pca = PCA(n_components=2)
        #         trajectories_reshaped = trajectories.reshape(-1, trajectories.shape[-1])
        #         pca.fit(trajectories_reshaped)
        #     else:  # fit_on == 'both'
        #         combined_trajectories = np.concatenate((hidden_trajectories.reshape(-1, hidden_trajectories.shape[-1]),
        #                                                 trajectories.reshape(-1, trajectories.shape[-1])), axis=0)
        #         pca = PCA(n_components=2)
        #         pca.fit(combined_trajectories)
        #
        #     # Transform the new trajectories using PCA
        #     pca_new_trajectories = pca.transform(new_trajectories.reshape(-1, new_trajectories.shape[-1]))
        #     pca_new_trajectories = pca_new_trajectories.reshape(new_trajectories.shape[0], new_trajectories.shape[1], -1)
        #
        #     # Transform both sets of trajectories
        #     pca_hidden_trajectories = pca.transform(hidden_trajectories.reshape(-1, hidden_trajectories.shape[-1]))
        #     pca_hidden_trajectories = pca_hidden_trajectories.reshape(hidden_trajectories.shape[0], hidden_trajectories.shape[1], -1)
        #
        #     pca_trajectories = pca.transform(trajectories.reshape(-1, trajectories.shape[-1]))
        #     pca_trajectories = pca_trajectories.reshape(trajectories.shape[0], trajectories.shape[1], -1)
        #
        #     from_t = 0
        #
        #     # Plot combined PCA of both sets of trajectories
        #     plt.figure(figsize=(10, 6))
        #     for i in range(pca_hidden_trajectories.shape[1]):
        #         plt.plot(pca_hidden_trajectories[:, i, 0], pca_hidden_trajectories[:, i, 1], lw=1, alpha=0.5, label='Hidden Trajectories' if i == 0 else "")
        #         plt.scatter(pca_hidden_trajectories[0, i, 0], pca_hidden_trajectories[0, i, 1], c='blue', marker='o', zorder=5)
        #         plt.scatter(pca_hidden_trajectories[-1, i, 0], pca_hidden_trajectories[-1, i, 1], c='orange', marker='o', zorder=5)
        #
        #     for i in range(num_trials):
        #         plt.plot(pca_trajectories[from_t:, i, 0], pca_trajectories[from_t:, i, 1], lw=1, alpha=0.5, label='Trajectories' if i == 0 else "")
        #         plt.scatter(pca_trajectories[from_t, i, 0], pca_trajectories[from_t, i, 1], c='green', marker='o', zorder=5)
        #         plt.scatter(pca_trajectories[-1, i, 0], pca_trajectories[-1, i, 1], c='red', marker='o', zorder=5)
        #
        #     for i in range(pca_new_trajectories.shape[1]):
        #         plt.plot(pca_new_trajectories[:, i, 0], pca_new_trajectories[:, i, 1], lw=1, alpha=0.5, label='New Trajectories' if i == 0 else "")
        #         plt.scatter(pca_new_trajectories[0, i, 0], pca_new_trajectories[0, i, 1], c='purple', marker='o', zorder=5)
        #         plt.scatter(pca_new_trajectories[-1, i, 0], pca_new_trajectories[-1, i, 1], c='yellow', marker='o', zorder=5)
        #
        #     plt.xlabel('PC1')
        #     plt.ylabel('PC2')
        #     plt.title('Combined PCA of Hidden and Regular Trajectories')
        #     plt.legend()
        #     plt.tight_layout()
        #     # plt.show()
        #     plt.savefig(Path(cfg.savepath) / "pca_trajectories.png", dpi=300)
        #     plt.close()


        if hasattr(cfg.dynamics,'attractors'):
            dynamics_function = instantiate(cfg.dynamics.function)
            attractors = instantiate(cfg.dynamics.attractors)

            from interpolation import cubic_hermite, generate_curves_between_points
            from plotting import remove_frame

            torch.manual_seed(0)

            num_points = 100
            rand_scale = 5.0 #1.0 #4.0 #4.1  # 10.0
            # Example usage
            x, y = attractors.detach().cpu().numpy()  #
            # x = np.array([0, 0])  # Start point
            # y = np.array([1, 1])  # End point
            m_x = -x + y  # Tangent at x
            m_y = -x + y  # Tangent at y

            num_curves = 100  # 20  # Number of random curves to generate

            all_points,alpha_range = generate_curves_between_points(x, y, lims=[0.0, 1.0],num_points=num_points,num_curves=num_curves,rand_scale=rand_scale, return_alpha=True)
            all_points = all_points.reshape(-1,all_points.shape[-1])
            # Perform PCA
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            pca_points = pca.fit_transform(all_points)

            # Convert points to torch tensor and run ODE integration
            points_tensor = torch.tensor(all_points, dtype=torch.float32)
            time_points = torch.linspace(0, 30, 2)  # Adjust time range and steps as needed

            # Only instantiate external input if it exists in config
            if hasattr(cfg.dynamics, 'static_external_input'):
                ext_input = instantiate(cfg.dynamics.static_external_input)
                ext_input = ext_input[None]
                ext_input = ext_input.repeat(all_points.shape[0], 1)
                # Run ODE integration for all points with external input
                with torch.no_grad():
                    trajectories = odeint(lambda t, x: dynamics_function(x, ext_input), points_tensor, time_points)
            else:
                # Run ODE integration without external input
                with torch.no_grad():
                    trajectories = odeint(lambda t, x: dynamics_function(x), points_tensor, time_points)

            # Convert trajectories to numpy and reshape for PCA
            trajectories_np = trajectories.detach().cpu().numpy()
            trajectories_reshaped = trajectories_np.reshape(-1, trajectories_np.shape[-1])

            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=2, random_state=42)
            cluster_labels = kmeans.fit_predict(trajectories_np[-1])

            # Plot PCA results
            fig,ax = plt.subplots(figsize=(5, 5))
            # plt.scatter(pca_points[:, 0], pca_points[:, 1],
            #             c=['C0' if label == 0 else 'C1' for label in cluster_labels], alpha=1.0)

            
            pca_points = pca_points.reshape(num_curves, num_points, pca.n_components_)
            cluster_labels = cluster_labels.reshape(num_curves, num_points)
            # Calculate changes in cluster labels along each curve
            changes = np.diff(cluster_labels, axis=1) != 0
            for i in range(num_curves):
                # Plot first segment up to change point
                change_idx = np.where(changes[i])[0][0] if np.any(changes[i]) else num_points
                if change_idx < num_points-1:
                    ax.plot(pca_points[i,:change_idx+1,0], pca_points[i,:change_idx+1,1],
                           c='C0' if cluster_labels[i,0] == 0 else 'C1', alpha=0.5)
                
                # Plot second segment from change point if it exists
                if change_idx < num_points-1:
                    ax.plot(pca_points[i,change_idx:,0], pca_points[i,change_idx:,1],
                           c='C1' if cluster_labels[i,0] == 0 else 'C0', alpha=0.5)

                if change_idx < num_points - 1:
                    ax.plot(pca_points[i,change_idx,0], pca_points[i,change_idx,1],
                       'x', color='red', markersize=10, alpha=0.8)

            # Flatten back for plotting
            pca_points = pca_points.reshape(-1, pca.n_components_)
            cluster_labels = cluster_labels.reshape(-1)
            
            # Plot endpoints in PCA space
            endpoints = np.array([x, y])
            pca_endpoints = pca.transform(endpoints)
            ax.scatter(pca_endpoints[:, 0], pca_endpoints[:, 1], color='lightgreen', s=50, label="Endpoints",zorder=1)

            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')
            remove_frame(ax)

            # ax.title('PCA of Cubic Hermite Interpolations')
            # plt.grid(True)

            fig.savefig(Path(cfg.savepath) / cfg.experiment_details / f"hermite_cubic_interpolations_plus_clustering_scale{rand_scale}.png", dpi=300)
            fig.savefig(Path(
                cfg.savepath) / cfg.experiment_details / f"hermite_cubic_interpolations_plus_clustering_scale{rand_scale}.pdf",
                        dpi=300)
            plt.show()
            plt.close(fig)

            # print(points_tensor.shape, ext_input.shape)
            full_input_to_KEF = points_tensor
            if hasattr(cfg.dynamics,'external_input_distribution_fit'):
                full_input_to_KEF = torch.concat([points_tensor, ext_input], axis=-1)
            full_input_to_KEF = full_input_to_KEF.reshape(num_curves, num_points, -1)

            KEFvals = SL.predict(full_input_to_KEF)

            cluster_labels = kmeans.fit_predict(trajectories_np[-1])
            cluster_labels = cluster_labels.reshape(num_curves, num_points)
            changes = np.diff(cluster_labels, axis=1) != 0
            change_points = np.argmax(changes, axis=1)
            # Handle cases where there are no changes (all zeros or all ones)
            no_changes = ~np.any(changes, axis=1)
            change_points[no_changes] = num_points // 2

            change_points_alpha = alpha_range[change_points]

            q = 0.05
            absKEFvals = np.abs(KEFvals.numpy())[..., 0]
            quantiles = np.quantile(absKEFvals, q, axis=1)
            # Find indices where KEF values are below the quantile threshold
            below_quantile = absKEFvals < quantiles[:, np.newaxis]
            below_threshold = absKEFvals < absKEFvals.max() / 100

            # Set values above quantile to -1 to ensure they're not selected by argmax
            masked_KEFvals = deepcopy(absKEFvals)
            masked_KEFvals[~below_threshold] = -np.inf

            # For each curve, find the maximum KEF value that's below the quantile
            max_below_threshold_id = np.argmax(masked_KEFvals, axis=1)
            # If all values were above quantile (all -1), set to middle point
            # max_below_quantile[max_below_quantile == 0] = num_points // 2

            max_below_threshold_position = alpha_range[max_below_threshold_id]

            # Find position of minimum absolute KEF value for each curve
            max_below_threshold_id = np.argmin(absKEFvals, axis=1)
            argmin_alpha = alpha_range[max_below_threshold_id]

            fig, ax = plt.subplots(1, 1, figsize=(3.2,3.2))
            ax.scatter(change_points_alpha, argmin_alpha)
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xlabel(r'true separatrix point $\alpha$')
            ax.set_ylabel(r'$\psi=0$ point $\alpha$')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_xticks([.0, 0.5, 1.0])
            ax.set_yticks([.0, 0.5, 1.0])
            ax.spines['left'].set_bounds(0, 1)
            ax.spines['bottom'].set_bounds(0, 1)
            ax.set_xlim(-.1,1.1)
            ax.set_ylim(-.1,1.1)

            # Compute standard R^2 score between change points and argmin positions
            from sklearn.metrics import r2_score
            r2 = r2_score(change_points_alpha, argmin_alpha)
            
            # Add R^2 annotation
            ax.text(0.3, 0.7, f'$R^2={r2:.3f}$',
                   transform=ax.transAxes,
                   verticalalignment='top',
                   fontsize=12)

            ax.set_aspect('equal')

            fig.tight_layout()
            plt.show()
            fig.savefig(Path(cfg.savepath) / cfg.experiment_details / f"separatrix_poisition_along_curves{rand_scale}.png", dpi=300)
            fig.savefig(Path(cfg.savepath) / cfg.experiment_details / f"separatrix_poisition_along_curves{rand_scale}.pdf",
                dpi=300)



            
            fig, axes = plt.subplots(4, 10, figsize=(20, 8))
            axes = axes.flatten()

            for i in range(min(num_curves,len(axes.flatten()))):
                ax = axes[i]
                ax.plot(alpha_range, KEFvals[i,:,:])
                ax.axvline(x=change_points_alpha[i], color='r', linestyle='--', alpha=0.7)
                ax.set_title(f'Curve {i+1}')
                ax.grid(True)

            # Hide any unused subplots
            for i in range(num_curves, len(axes)):
                axes[i].set_visible(False)

            plt.tight_layout()
            plt.show()
            fig.savefig(Path(cfg.savepath) / cfg.experiment_details / f"hermite_cubic_interpolations_KEFvals_scale{rand_scale}.png", dpi=300)


            ########
            
            # Select 3 curves based on min, max, and median change points
            min_idx = np.argmin(change_points)
            max_idx = np.argmax(change_points)
            median_idx = np.argsort(change_points)[len(change_points)//2]
            selected_indices = [min_idx, median_idx, max_idx]
            labels = ['Curve 1', 'Curve 2', 'Curve 3']
            colors = ['blue', 'green', 'red']

            # Create single plot
            fig, ax = plt.subplots(figsize=(4, 3))

            for idx, label, color in zip(selected_indices, labels, colors):
                ax.plot(alpha_range, KEFvals[idx,:,:], color=color,lw=1)
                if idx == max_idx:
                    ax.axvline(x=change_points_alpha[idx], color=color, linestyle='--', alpha=1)
                    ax.text(change_points_alpha[idx]+0.065, 10, 'True separatrix\npoint',
                           rotation=90, va='bottom', ha='center', color=color)
                else:
                    ax.axvline(x=change_points_alpha[idx], color=color, linestyle='--', alpha=1)
            ax.axhline(0,ls='solid',c='k',alpha=0.3)
            ax.set_xlabel(r'Curve parameter $\alpha$',fontsize=13)
            ax.set_ylabel(r'KEF Value $\psi$',fontsize=13)
            remove_frame(ax,['right','top'])
            ax.set_xticks([0,0.5,1])
            ax.spines['bottom'].set_bounds(0, 1)
            ax.set_yticks([-80,-40,0,40,80])
            ax.set_yticklabels([])
            ax.spines['left'].set_bounds(-80, 80)

            # ax.set_title('KEF Values Along Selected Curves')


            fig.tight_layout()
            plt.show()
            fig.savefig(Path(cfg.savepath) / cfg.experiment_details / f"hermite_cubic_interpolations_KEFvals_selected_scale{rand_scale}.png", dpi=300)
            fig.savefig(Path(cfg.savepath) / cfg.experiment_details / f"hermite_cubic_interpolations_KEFvals_selected_scale{rand_scale}.pdf", dpi=300)

            ########





            from learn_koopman_eig import runGD_basic

            SL.models[0].eval()
            SL.prepare_models_for_gradient_descent(distribution)

            anchor_point = attractors[0:1]
            torch.manual_seed(42)
            pert = torch.randn(anchor_point.shape)
            anchor_point = anchor_point + pert

            shifted_KEF_function = compose(
                SL.functions_for_gradient_descent[0],
                lambda x: x + anchor_point
            )

            KEF_function = SL.functions_for_gradient_descent[0]

            num_points_GD = 5  # 00 #00
            noise_scale = 0.04
            hidden = attractors[:1].repeat(num_points_GD, 1)
            hidden = hidden * 0 + torch.randn(hidden.shape) * noise_scale
            hidden.shape

            print(KEF_function(attractors),KEF_function(anchor_point), shifted_KEF_function(hidden*0))

            traj, below_thr_points = runGD_basic(
                shifted_KEF_function,
                initial_conditions=hidden,
                partial_optim=partial(torch.optim.Adam, lr=0.6e-3, weight_decay=0.03),
                threshold=1e-1,
                num_steps=2000,
                save_trajectories_every=100,
            )
            new_traj, below_thr_points = runGD_basic(
                shifted_KEF_function,
                initial_conditions=hidden,
                partial_optim=partial(torch.optim.Adam, lr=0.6e-3, weight_decay=0.03),
                threshold=1e-1,
                num_steps=2000,
                save_trajectories_every=100,
            )
            traj = np.concatenate((traj,new_traj))
            traj_distances = np.linalg.norm(traj, axis=-1)
            below_thr_points = below_thr_points + anchor_point


            f = SL.functions_for_gradient_descent[0]
            best_id = np.argmin(np.linalg.norm(below_thr_points - anchor_point,axis=-1))
            new_point = below_thr_points[best_id:best_id+1] #[1:2]
            KEFat_new_point = f(new_point)
            print("KEFat_new_point",KEFat_new_point)
            with torch.no_grad():
                KEFtraj = shifted_KEF_function(torch.tensor(traj))

            fig,axs = plt.subplots(2,1, sharex=True)
            ax = axs[0]
            ax.plot(traj_distances)
            ax.set_ylabel('Distance from attractor')
            ax = axs[1]
            ax.plot(KEFtraj[...,0])
            ax.set_ylabel('KEF value')
            ax.set_xlabel('training iterations')
            plt.show()

            from separatrix_point_finder import find_separatrix_point_along_line
            # Stack anchor point and attractor[2] for endpoints
            endpoints = torch.stack([
                anchor_point[0],
                attractors[1]
            ])
            line_separatrix_point = find_separatrix_point_along_line(
                dynamics_function,
                instantiate(cfg.dynamics.static_external_input),
                endpoints,
                num_points=20,
                num_iterations=3,
                final_time=30,
            )


            separatrix_points = [new_point[0],line_separatrix_point]

            all_points_st = all_points.reshape(num_curves,num_points,all_points.shape[-1])
            changes = np.diff(cluster_labels, axis=1) != 0
            change_points = np.argmax(changes, axis=1)
            change_point_states = all_points_st[np.arange(num_curves),change_points]

            distance_to_anchor_point = lambda point: np.linalg.norm(point - anchor_point,axis=1)
            dists = [
                distance_to_anchor_point(point)
                for point in separatrix_points
            ]
            curves_change_point_dists = distance_to_anchor_point(torch.tensor(change_point_states))

            # Plot vertical lines for distances with labels and histogram
            fig, ax = plt.subplots(figsize=(4, 3))
            labels = [r'Optimal direction', 'Towards  target fixed point']
            colors = ['C8', 'C5']
            
            # Plot histogram first so it's in background
            ax.hist(curves_change_point_dists, bins=20, alpha=0.5, color='C6', label='Towards random points on separatrix', density=True)
            
            # Plot vertical lines for each distance
            for dist, label, color in zip(dists, labels, colors):
                ax.axvline(x=dist, color=color, lw=2, linestyle='--', label=label)
            # Add annotations for each vertical line
            # Add annotations for vertical lines
            for dist, label, color in zip(dists, labels, colors):
                ax.annotate(
                    label,
                    xy=(dist, ax.get_ylim()[1]),
                    xytext=(dist-0.2, (ax.get_ylim()[0]+ax.get_ylim()[1])/2),
                    rotation=90,
                    ha='right',
                    va='center',
                    color=color,
                    fontsize=13
                )
            
            # Add annotation for histogram
            ax.annotate(
                'Towards random points\non separatrix',
                xy=(curves_change_point_dists.mean(), ax.get_ylim()[1]/2),
                xytext=(curves_change_point_dists.mean()-0.8, (ax.get_ylim()[0]+ax.get_ylim()[1])/2),
                rotation=90,
                ha='right',
                va='center',
                color='C6',
                fontsize=13
            )
            ax.set_xlabel('Distance from anchor point')
            # ax.set_ylabel('Count')
            remove_frame(ax,spines_to_remove=['top','right','left'])
            ax.set_xticks(np.arange(0,13,4))
            # ax.legend(fontsize=12,framealpha=0.2)
            fig.tight_layout()
            fig.savefig(Path(cfg.savepath) / cfg.experiment_details / f"separatrix_points_distances_histogram_scale{rand_scale}.png", dpi=300)
            fig.savefig(Path(cfg.savepath) / cfg.experiment_details / f"separatrix_points_distances_histogram_scale{rand_scale}.pdf")
            plt.show()


            num_points = 1000 #0 #00
            alpha = torch.linspace(0,10,num_points)[:,None]
            og_line = torch.zeros_like(new_point) * (1-alpha) + new_point * alpha

            n_shuffles = 10 #0 #00 #000

            from odeint_utils import run_odeint_to_final
            shuffled_lines = [og_line[...,np.random.permutation(og_line.shape[-1])] for _ in range(n_shuffles)]

            ### generate hermite curves

            num_curves = 10 #0 #100  # 20  # Number of random curves to generate
            rand_scale = 4.0  # 4.1  # 10.0
            x, y = attractors.detach().cpu().numpy()  #
            m_x = -x + y  # Tangent at x
            m_y = -x + y  # Tangent at y
            all_points = []
            for _ in range(num_curves):
                m_x_perturbed = m_x * 1.4 + np.random.randn(m_x.shape[0]) * rand_scale
                m_y_perturbed = m_y * 1.4 + np.random.randn(m_x.shape[0]) * rand_scale
                points = cubic_hermite(x, y, m_x_perturbed, m_y_perturbed, num_points)
                all_points.append(points)
            all_points = np.stack(all_points)

            ####

            # all_lines =
            all_lines = torch.concat([
                torch.tensor(all_points).type_as(og_line) - anchor_point,
                torch.stack([og_line] + shuffled_lines, dim=0),
            ])
            # final_points = []
            # for line in [og_line]+shuffled_lines:
            #     final_point = run_odeint_to_final(
            #         dynamics_function,
            #         line + anchor_point,
            #         cfg.dynamics.all_attractors.T
            #     )
            #     final_points.append(final_point)
            # final_points = np.stack(final_points)
            final_points = run_odeint_to_final(
                dynamics_function,
                all_lines.reshape(-1,all_lines.shape[-1]) + anchor_point,
                cfg.dynamics.all_attractors.T
            )
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=2, random_state=0).fit(final_points.reshape(-1,final_points.shape[-1]))
            labels = kmeans.labels_.reshape(*all_lines.shape[:2])


            # plt.figure()
            # for i in range(n_shuffles):
            #     l_shuffle, = plt.plot(alpha.flatten(), labels[i], label='shuffled',c='C1',lw=0.5)
            # l_orig, = plt.plot(alpha.flatten(), labels[0], label='optimised', c='C0')
            # plt.legend([l_orig, l_shuffle],['l_orig','l_shuffle'])
            # plt.show()

            trajectory_num,position = np.where(np.abs(np.diff(labels,axis=1))==1)
            all_dists = np.linalg.norm(all_lines[trajectory_num,position],axis=-1)
            curve_dists, og_dist, perm_dists = np.split(all_dists,([num_curves,num_curves+1]),axis=0)

            # trajectories_reshaped = trajectories_np[-1].reshape(num_curves,-1,trajectories_np.shape[-1])
            # traj_change_points = np.stack([trajectories_reshaped[i,change_points_id[i]] for i in range(num_curves)])
            # euclidean_dists_curve_change_points =  np.linalg.norm(traj_change_points - anchor_point.detach().cpu().numpy(),axis=-1)
            # all_dists = np.linalg.norm(all_lines[shuffle_num,position],axis=-1)
            # optimized_dist = all_dists[0]
            # shuffled_dists = all_dists[1:]
            #
            # plt.figure()
            # plt.axvline(alpha[position[0]], color='r', linestyle='--')
            # plt.hist(alpha[position[1:]], density=True, bins=20)
            # plt.xticks(np.arange(0, 11, 2))
            # plt.show()
            print(curve_dists,og_dist,perm_dists)
            interpolating_point_dist = np.linalg.norm(instantiate(cfg.dynamics.point_on_separatrix) - anchor_point)

            plt.figure()
            plt.axvline(og_dist, color='r', linestyle='--',label=r'GD on KEF')
            plt.axvline(interpolating_point_dist, color='g', linestyle='--', label=r'where interpolating line meets separatrix')
            plt.hist(perm_dists, density=True, bins=20, label='where permutated vector line meets separatrix',alpha=0.5)
            plt.hist(curve_dists,density=True,label='where hermite curves meet separatrix',alpha=0.5)
            plt.legend()
            plt.xlabel('Distance from attractor')
            plt.show()



            # P = PCA(n_components=2)
            # P.fit(
            #     np.stack([
            #         anchor_point,traj.reshape(-1,traj.shape[-1])
            #     ])
            # )


            # GD_traj, all_below_threshold_points, all_below_threshold_masks  = SL.find_separatrix(
            #     distribution,
            #     initial_conditions = hidden,
            #     external_inputs = None,
            #     dist_needs_dim=False,
            #     return_indices = False,
            #     return_mask = True,
            #     **instantiate(cfg.separatrix_find_separatrix_kwargs)
            # )


        if False: #hasattr(cfg.dynamics,'RNN_dataset'):
            dataset = instantiate(cfg.dynamics.RNN_analysis_dataset)
            dataset.N_trials_cd = 20
            rnn = instantiate(cfg.dynamics.loaded_RNN_model)
            dist = instantiate(cfg.dynamics.IC_distribution)
            inputs, targets = dataset()
            inputs = torch.from_numpy(inputs).type(torch.float)
            targets = torch.from_numpy(targets)
            # inputs = inputs * 0
            # print('inputs.shape',inputs.shape)
            # print('batch first',rnn.batch_first)
            outputs,hidden = rnn(inputs,return_hidden=True,deterministic=False)
            hidden = hidden.detach()



            ###
            # fp_inputs = torch_inp.reshape(-1, torch_inp.shape[-1]).detach().cpu().numpy()
            # initial_conditions = hidden_traj.reshape(-1, hidden_traj.shape[-1])
            ###

            P = PCA(n_components=3)
            # non_linearity = lambda x: rnn_model.f0 / (1.0 + torch.exp(-rnn_model.beta0 * (x - rnn_model.theta0)))
            # rates = non_linearity(hidden)
            hidden_last = hidden[-3000:]
            P.fit(hidden_last.reshape(-1,hidden_last.shape[-1]).detach().cpu())
            pc_hidden = P.transform(hidden.reshape(-1,hidden.shape[-1]).detach().cpu()).reshape(*hidden.shape[:2],P.n_components)

            plt.figure()
            for i in range(pc_hidden.shape[1]):
                plt.plot(inputs[:,i,2],pc_hidden[:,i,0],lw=1,alpha=0.5)

            if cfg.run_fixed_point_finder:
                unique_fps_rates = torch.from_numpy(unique_fps.xstar)
                pc_fps = P.transform(unique_fps_rates)
                # pc_IC  = P.transform(initial_conditions)
                plt.scatter(unique_fps.inputs[unique_fps.is_stable,2],pc_fps[unique_fps.is_stable, 0], c='blue', marker='x', s=100, zorder=1001)
                plt.scatter(unique_fps.inputs[~unique_fps.is_stable,2],pc_fps[~unique_fps.is_stable, 0], c='red', marker='x', s=100, zorder=1000)
                # plt.scatter(fp_inputs[:,2],pc_IC[:,0],c='green')

            plt.savefig(Path(cfg.savepath) / "PCA_traj.png",dpi=300)
            plt.close()

            KEFvals = []
            for i in range(SL.num_models):
                mod_model = compose(
                    # lambda x: x**0.1,
                    torch.log,
                    lambda x: x + 1,
                    torch.exp,
                    # partial(torch.sum, dim=-1, keepdims=True),
                    torch.log,
                    torch.abs,
                    SL.models[i] #.to('cpu')
                )
                samples_for_normalisation = 1000
                needs_dim = True
                if hasattr(cfg.dynamics, 'dist_requires_dim'):
                    needs_dim = cfg.dynamics.dist_requires_dim

                dist_option = dist
                if hasattr(cfg.dynamics,"combined_distribution"):
                    dist_option = instantiate(cfg.dynamics.combined_distribution)

                samples = dist_option.sample(sample_shape=[samples_for_normalisation])

                norm_val = float(
                    torch.mean(torch.sum(mod_model(samples) ** 2, axis=-1)).sqrt().detach().numpy())

                # mod_model = compose(
                #     lambda x: x / norm_val,
                #     mod_model
                # )
                concat_input = torch.concat([hidden,inputs],dim=-1)
                # print('concat_input.shape',concat_input.shape,'samples.shape',samples.shape)
                KEFval = mod_model(concat_input.reshape(-1,concat_input.shape[-1])).detach().reshape(*concat_input.shape[:2],-1)
                KEFvals.append(KEFval)

            KEFvals = torch.concat(KEFvals,dim=-1).detach().cpu()

            fig,axs = plt.subplots(2,5, figsize=(10,4), sharex=True, sharey=True)
            for i in range(SL.num_models):
                ax = axs.flatten()[i]
                scatter = ax.scatter(inputs[:,:,2], pc_hidden[:,:,0], c=KEFvals[:,:,i], s=10, cmap='viridis')
                fig.colorbar(scatter, ax=ax)
            fig.tight_layout()
            fig.savefig(Path(cfg.savepath) / "PCA1_inputs_KEFvals.png",dpi=300)


            # threshold = np.quantile(inputs[:,:,2].flatten(), 0.96)
            # top_indices = np.where(inputs[:,:,2] >= threshold)
            top_indices = np.where(
                (0.9 <= inputs[:, :, 2]) & (inputs[:, :, 2] <= 0.92)
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

            print(f"Maximum distance: {max_distance}")


            # print(f"Point 1: {point1}")
            # print(f"Point 2: {point2}")

            plt.figure()
            plt.hist(point1.flatten())
            plt.hist(point2.flatten())
            plt.savefig(Path(cfg.savepath) / "fixedpoint_histograms.png",dpi=300)
            plt.close()

            # Create 100 linearly interpolated points between point1 and point2
            n_grid = 1000
            alpha = torch.linspace(-.5, 1.5, n_grid)
            interpolated_points = alpha[:, None] * point1[None, :] + (1 - alpha)[:, None] * point2[None, :]
            interpolated_inputs = (inputs[top_indices[0][max_dist_idx[0]], top_indices[1][max_dist_idx[0]]] +
                                 inputs[top_indices[0][max_dist_idx[1]], top_indices[1][max_dist_idx[1]]]) / 2

            concat_interpolated = torch.concat([interpolated_points,interpolated_inputs.repeat(n_grid,1)],dim=-1)

            KEFvals = []
            for i in range(SL.num_models):
                mod_model = compose(
                    # lambda x: x**0.1,
                    torch.log,
                    lambda x: x + 1,
                    # torch.exp,
                    # partial(torch.sum, dim=-1, keepdims=True),
                    # torch.log,
                    torch.abs,
                    SL.models[i]#.to('cpu')
                )
                samples_for_normalisation = 1000
                needs_dim = True


                samples = dist_option.sample(sample_shape=[samples_for_normalisation])

                norm_val = float(
                    torch.mean(torch.sum(mod_model(samples) ** 2, axis=-1)).sqrt().detach().numpy())

                mod_model = compose(
                    lambda x: x / norm_val,
                    mod_model
                )
                KEFval = mod_model(concat_interpolated).detach()
                KEFvals.append(KEFval)

            KEFvals = torch.stack(KEFvals, dim=0).detach().cpu()

            fig,ax = plt.subplots()
            for i in range(SL.num_models):
                ax.plot(KEFvals[i],c=f'C{i}')
            ax.set_ylabel(r'KEF value')
            ax.set_xlabel('Position along decision axis')
            plt.savefig(Path(cfg.savepath) / "KEFs_interpolated.png", dpi=300)
            plt.close()

            plt.figure()
            for i in range(pc_hidden.shape[1]):
                plt.plot(inputs[:,i,2],pc_hidden[:,i,0],lw=1,alpha=0.5)
            pc_interpolated_points = P.transform(interpolated_points)
            plt.plot(concat_interpolated[:,-1],pc_interpolated_points[:,0],lw=1,ls='dotted',c='black')
            plt.savefig(Path(cfg.savepath) / "PCA_interpolation_line.png", dpi=300)

            plt.figure()
            for i in range(pc_hidden.shape[1]):
                plt.plot(inputs[:, i, :])
            plt.savefig(Path(cfg.savepath) / "inputs.png", dpi=300)
            plt.close()


            # Run from interpolated line
            n_grid = 500
            alpha = torch.linspace(-0.4, 1.4, n_grid)
            interpolated_points = alpha[:, None] * point1[None, :] + (1 - alpha)[:, None] * point2[None, :]
            interpolated_inputs = (inputs[top_indices[0][max_dist_idx[0]], top_indices[1][max_dist_idx[0]]] +
                                   inputs[top_indices[0][max_dist_idx[1]], top_indices[1][max_dist_idx[1]]]) / 2

            # Set simulation length for interpolated points
            run_T = 1000
            samples_T = 20

            # Expand interpolated inputs over time dimension
            interpolated_inputs_expanded = interpolated_inputs.repeat(n_grid, 1).unsqueeze(0).expand(samples_T, -1, -1)

            use_odeint = True

            if use_odeint:
                dynamics_function = instantiate(cfg.dynamics.function)

                def ode_dynamics(t, x):
                    return dynamics_function(x, interpolated_inputs_expanded[0, :, :])

                times = torch.linspace(0, run_T, samples_T)
                interpolated_trajectories = odeint(ode_dynamics, interpolated_points, times)
                # interpolated_trajectories_lowtol = odeint(ode_dynamics, interpolated_points, times)
            else:
                # Run RNN from interpolated initial conditions
                interpolated_trajectories = rnn(
                    interpolated_inputs_expanded,
                    x_init=interpolated_points[None],
                    deterministic=True,
                    return_hidden=True
                )[1]  # Only take second output, ignoring hidden states


            # Reshape trajectories
            interpolated_trajectories_r = interpolated_trajectories.reshape(samples_T, n_grid, -1)
            # interpolated_trajectories_lowtol_r = interpolated_trajectories_lowtol.reshape(run_T, n_grid, -1)

            # setattr(cfg.separatrix_locator_fit_kwargs, 'num_epochs', 2000)
            # SL.fit(
            #     dynamics_function,
            #     distribution,
            #     external_input_dist=input_distribution,
            #     fixed_x_batch = interpolated_points,
            #     fixed_external_inputs = interpolated_inputs_expanded[0],
            #     **instantiate(cfg.separatrix_locator_fit_kwargs)
            # )


            #### Computing PDE error
            from learn_koopman_eig import shuffle_normaliser

            # # Define three sets of points
            # point_sets = [
            #     instantiate(cfg.dynamics.IC_distribution_full).sample(sample_shape=(interpolated_points.shape[0],)),
            #     instantiate(cfg.dynamics.IC_distribution_task_relevant).sample(sample_shape=(interpolated_points.shape[0],)),
            #     instantiate(cfg.dynamics.IC_distribution_task_relevant_PC1).sample(sample_shape=(interpolated_points.shape[0],)),
            #     instantiate(cfg.dynamics.IC_interpolation_line_1).sample(sample_shape=(interpolated_points.shape[0],)),
            #     interpolated_points,
            #     interpolated_points + torch.normal(mean=0.0, std=0.4, size=interpolated_points.shape),
            # ]
            #
            # #dist_PC1 = instantiate(cfg.dynamics.IC_distribution_task_relevant_PC1)
            #
            # colors = ['g', 'b', 'y',  'purple', 'r', 'orange']  # Colors for each set
            # labels = ['Isotropic','Task data', 'Task data PC1', 'interpolated_points', 'line', 'line+noise0.4',]
            #
            # from mpl_toolkits.mplot3d import Axes3D
            #
            # plt.close()
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # for i, (points, label) in enumerate(zip(point_sets, labels)):
            #     if i<2:
            #         continue
            #     # Transform points using PCA
            #     transformed_points = P.transform(points.detach().cpu().numpy())
            #     # Plot PC1, PC2, and PC3
            #     ax.scatter(transformed_points[:, 0], transformed_points[:, 1], transformed_points[:, 2], label=label, alpha=0.4, c=colors[i])
            # ax.set_xlabel('PC1')
            # ax.set_ylabel('PC2')
            # ax.set_zlabel('PC3')
            # ax.legend()
            # plt.show()
            #
            # vector_1 = point_sets[2][1] - point_sets[2][0]
            # vector_2 = point_sets[4][1] - point_sets[4][0]  # Adjusted index to match the provided point_sets
            # from scipy.stats import pearsonr
            # pearson_corr, _ = pearsonr(vector_1.detach().cpu(), vector_2.detach().cpu())
            # print(f"Pearson's correlation of the two vectors: {pearson_corr}")
            #
            #
            # plt.close()
            # plt.figure()
            # for i, (points, label) in enumerate(zip(point_sets,labels)):
            #     if i<3:
            #         continue
            #     if i>4:
            #         continue
            #     points.requires_grad_(True)
            #     inputs = torch.concat([points, interpolated_inputs_expanded[0]], axis=-1)
            #     phi_x = SL.predict(inputs, no_grad=False)
            #
            #     phi_x_prime = torch.autograd.grad(
            #         outputs=phi_x,
            #         inputs=points,
            #         grad_outputs=torch.ones_like(phi_x),
            #         create_graph=True
            #     )[0]
            #
            #     # Compute F(x_batch)
            #     F_x = dynamics_function(points, interpolated_inputs_expanded[0])
            #
            #     # Main loss term: ||phi'(x) F(x) - phi(x)||^2
            #     dot_prod = (phi_x_prime * F_x).sum(axis=-1, keepdim=True)
            #
            #     # Use shuffle_normaliser to compute the normalized loss
            #     normalised_loss = shuffle_normaliser(dot_prod, phi_x)
            #     print(f"Normalised loss for set {i+1}:", normalised_loss)
            #
            #     # # Evaluate the log likelihood at the points
            #     # log_likelihoods = dist.log_prob(points)
            #     # print(f"Log likelihoods at points for set {i+1}:", log_likelihoods)
            #
            #     # Evaluate the log likelihood at zero
            #     # zero_point = torch.zeros_like(points)
            #     # log_likelihoods_zero = dist.log_prob(zero_point)
            #     # print(f"Log likelihoods at zero for set {i+1}:", log_likelihoods_zero)
            #
            #     # Plot PDE errors for each set
            #     plt.scatter(
            #         dot_prod.detach().cpu().repeat(1, phi_x.shape[-1]), phi_x.detach().cpu(), color=colors[i], label=label, alpha=0.4
            #     )
            #
            # plt.xlabel(r'$\nabla \psi(x) \cdot f(x)$')
            # plt.ylabel(r'$\lambda \psi(x)$')
            # plt.legend()
            # # plt.aspect('equal')
            # plt.tight_layout()
            # plt.show()
            # plt.savefig(Path(cfg.savepath) / "PDE_scatter.png", dpi=300)


            print('transforming interpolated points')

            # Transform interpolated trajectories using PCA
            interpolated_trajectories_pca = P.transform(interpolated_trajectories_r.reshape(-1, interpolated_trajectories_r.shape[-1]).detach().cpu().numpy())
            interpolated_trajectories_pca = interpolated_trajectories_pca.reshape(*interpolated_trajectories_r.shape[:-1], -1)
            # interpolated_trajectories_lowtol_pca = P.transform(interpolated_trajectories_lowtol_r.reshape(-1, interpolated_trajectories_lowtol_r.shape[-1]).detach().cpu().numpy())
            # interpolated_trajectories_lowtol_pca = interpolated_trajectories_lowtol_pca.reshape(*interpolated_trajectories_lowtol_r.shape[:-1], -1)

            # Create figure for PCA trajectories over time
            fig, ax = plt.subplots(figsize=(10, 6))
            # Create colormap based on alpha values
            norm = plt.Normalize(alpha.min(), alpha.max())
            cmap = plt.cm.viridis
            mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            for i in range(n_grid):
                ax.plot(interpolated_trajectories_pca[:, i, 0],
                        color=cmap(norm(alpha[i])),
                        alpha=0.5, lw=1)
            # Adjust layout and add colorbar
            fig.subplots_adjust(right=0.85)
            cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
            fig.colorbar(mappable, cax=cbar_ax, label='Alpha')
            ax.set_xlabel('time')
            ax.set_ylabel('PC1')
            ax.set_title('PCA Trajectories from Interpolated Initial Conditions')
            plt.show()
            fig.savefig(Path(cfg.savepath) / "interpolated_trajectories_pca.png", dpi=300)
            plt.close(fig)

            # Concatenate points and inputs for KEF evaluation
            concat_traj = torch.cat([interpolated_trajectories_r, interpolated_inputs_expanded], dim=-1)

            print('evaluating KEFs')

            # Evaluate KEFs
            KEFvals_traj = []
            for i in range(SL.num_models):
                mod_model = compose(
                    # torch.log,
                    # lambda x: x + 1,
                    # torch.abs,
                    SL.models[i]
                )
                samples_for_normalisation = 1000
                samples = dist_option.sample(sample_shape=[samples_for_normalisation])
                norm_val = float(
                    torch.mean(torch.sum(mod_model(samples) ** 2, axis=-1)).sqrt().detach().numpy())
                # mod_model = compose(
                #     lambda x: x / norm_val,
                #     mod_model
                # )
                KEFval = mod_model(concat_traj).detach()
                KEFvals_traj.append(KEFval)

            KEFvals_traj = torch.stack(KEFvals_traj, dim=0).detach().cpu()

            # Plot KEFs
            fig, axs = plt.subplots(2, 5, figsize=(10, 4))
            axs = axs.ravel()
            mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            for i in range(SL.num_models):
                for t in range(n_grid):
                    axs[i].plot(KEFvals_traj[i, :, t],
                              color=cmap(norm(alpha[t])),
                              alpha=0.5, lw=1)
                axs[i].set_title(f'KEF {i+1}')
                axs[i].set_xlabel('Time')
                axs[i].set_ylabel('KEF value')
                axs[i].set_xlim(0,20)
            # Adjust layout and add colorbar
            fig.subplots_adjust(right=0.85)
            cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
            # fig.tight_layout()
            fig.colorbar(mappable, cax=cbar_ax, label='Alpha')
            fig.savefig(Path(cfg.savepath) / "interpolated_trajectories_KEF.png", dpi=300)
            plt.close(fig)


            ### KEFs vs PCA
            # Create a figure with two subplots stacked vertically
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 6), sharex=True)

            # Plot KEFs in top subplot
            for i in range(SL.num_models):
                ax1.plot(alpha, np.abs(KEFvals_traj[i, 0]), label=f'KEF {i+1}', c=f'C{i}')
            ax1.set_xlabel('Alpha')
            ax1.set_ylabel('KEF value')
            # ax1.legend()

            # Plot final PCA timepoint in bottom subplot
            pca_final = interpolated_trajectories_pca[-1,:,0]  # Get final timepoint
            # pca_final_lowtol = interpolated_trajectories_lowtol_pca[-1,:,0]  # Get final timepoint with lower tolerance
            ax2.plot(alpha, pca_final, label='Standard tolerance')
            # ax2.plot(alpha, pca_final_lowtol, label='Low tolerance', ls='--')
            ax2.set_xlabel('Alpha')
            ax2.set_ylabel('Final PC1')
            ax2.legend()

            # plt.tight_layout()
            fig.savefig(Path(cfg.savepath) / "KEF_pca_vs_alpha.png", dpi=300)
            plt.show()
            plt.close(fig)




            # Sample every 100 timesteps
            n_trials = 40
            sample_every = 100
            sampled_hidden = hidden[::sample_every,:n_trials].detach().clone()
            sampled_inputs = inputs[::sample_every,:n_trials].detach().clone()

            # Add multivariate Gaussian noise with PCA covariance statistics
            pca_cov = torch.from_numpy(P.get_covariance()).float()
            mvn = torch.distributions.MultivariateNormal(
                loc=torch.zeros(sampled_hidden.shape[-1]),
                covariance_matrix=pca_cov
            )
            noise = mvn.sample(sampled_hidden.shape[:-1])
            perturbed_hidden = sampled_hidden + noise * 0.5

            # Reshape sampled tensors to 2D (combining first two dimensions)
            sampled_hidden_2d = sampled_hidden.reshape(-1, sampled_hidden.shape[-1])
            sampled_inputs_2d = sampled_inputs.reshape(-1, sampled_inputs.shape[-1])
            perturbed_hidden_2d = perturbed_hidden.reshape(-1, perturbed_hidden.shape[-1])

            # Set simulation length
            run_T = 100

            # Expand inputs by repeating along new middle dimension
            sampled_inputs_2d = sampled_inputs_2d.unsqueeze(0).expand(run_T, -1, -1)

            # Run RNN on perturbed initial conditions
            perturbed_trajectories = rnn(
                sampled_inputs_2d, #.to(rnn.device),
                x_init=perturbed_hidden_2d[None], #.to(rnn.device),
                deterministic=True,
                return_hidden=True,
            )[1]  # Only take second output, ignoring hidden states

            perturbed_trajectories_r = perturbed_trajectories.reshape(run_T, *perturbed_hidden.shape)
            # Expand sampled_inputs to match perturbed_trajectories_r shape
            sampled_inputs_expanded = sampled_inputs.unsqueeze(0).expand(perturbed_trajectories_r.shape[0], -1, -1, -1)
            # Concatenate perturbed trajectories and sampled inputs along last dimension
            perturbed_trajectories_r_inp = torch.cat([perturbed_trajectories_r, sampled_inputs_expanded], dim=-1)

            # Transform perturbed trajectories using PCA
            # P.transform(hidden)
            sampled_hidden_pca = P.transform(sampled_hidden.reshape(-1, sampled_hidden.shape[-1]).detach().cpu().numpy())
            sampled_hidden_pca = sampled_hidden_pca.reshape(*sampled_hidden.shape[:-1], -1)
            perturbed_hidden_pca = P.transform(perturbed_hidden.reshape(-1, perturbed_hidden.shape[-1]).detach().cpu().numpy())
            perturbed_hidden_pca = perturbed_hidden_pca.reshape(*perturbed_hidden.shape[:-1], -1)
            perturbed_trajectories_pca = P.transform(perturbed_trajectories_r.reshape(-1, perturbed_trajectories_r.shape[-1]).detach().cpu().numpy())
            perturbed_trajectories_pca = perturbed_trajectories_pca.reshape(*perturbed_trajectories_r.shape[:-1], -1)

            all_KEF_vals = []
            for i in range(SL.num_models):
                KEFvals = SL.models[i](perturbed_trajectories_r_inp.reshape(-1, perturbed_trajectories_r_inp.shape[-1]))
                KEFvals = KEFvals.reshape(*perturbed_trajectories_r_inp.shape[:-1],-1)
                all_KEF_vals.append(KEFvals)
            all_KEF_vals = torch.stack(all_KEF_vals, dim=0).detach().cpu()
            print(all_KEF_vals.shape)

            # Create subplots with 5 rows and 10 columns
            fig, axs = plt.subplots(5, 10, figsize=(20, 10), sharey='row',sharex='col')

            # Plot PCA trajectories in first row
            for j in range(10):  # For each trajectory
                axs[0,j].plot(perturbed_trajectories_pca[...,5*j,:,0])
                if j == 0:  # Only leftmost column gets y labels
                    axs[0,j].set_ylabel('PCA 1')

            # Plot each KEF value in remaining rows
            for i in range(4):  # For each model
                for j in range(10):  # For each trajectory
                    axs[i+1,j].plot(all_KEF_vals[i,:,5*j,:,0])
                    if i == 3:  # Only bottom row gets x labels
                        axs[i+1,j].set_xlabel('time')
                    if j == 0:  # Only leftmost column gets y labels
                        axs[i+1,j].set_ylabel(f'KEF {i+1}')

            plt.tight_layout()
            plt.savefig(Path(cfg.savepath) / "KEF_evolution.png", dpi=300)
            plt.close()


            # Reshape perturbed trajectories to match original shape
            plt.figure()
            neuron_id = 2
            for i in range(hidden.shape[1]):
                plt.plot(hidden[:, i, :].mean(-1).detach().cpu().numpy())
            plt.savefig(Path(cfg.savepath) / "hidden_traj.png",dpi=300)
            plt.close()
            print(
                'hidden[:5,0,0]:',hidden[:5,0,0]
            )

            # inputs = inputs[:2]
            # hidden = hidden[:2]
            #
            # select_every = 1
            # external_inputs_select = inputs.reshape(-1,inputs.shape[-1]).detach().clone()[::select_every]
            # hidden_select = hidden.reshape(-1, hidden.shape[-1]).detach().clone()[::select_every]
            # GD_traj, all_below_threshold_points, all_below_threshold_masks  = SL.find_separatrix(
            #     distribution,
            #     initial_conditions = hidden_select,
            #     external_inputs = external_inputs_select,
            #     external_input_dist = input_distribution,
            #     dist_needs_dim=cfg.dynamics.dist_requires_dim if hasattr(cfg.dynamics,
            #                                                              "dist_requires_dim") else True,
            #     return_indices = False,
            #     return_mask = True,
            #     **instantiate(cfg.separatrix_find_separatrix_kwargs)
            # )
            #
            #
            # KEFvals = []
            # delta_dists = []
            # delta_hiddens = []
            # for i in range(SL.num_models):
            #     mod_model = compose(
            #         torch.log,
            #         lambda x: x + 1,
            #         torch.exp,
            #         partial(torch.sum, dim=-1, keepdims=True),
            #         torch.log,
            #         torch.abs,
            #         SL.models[i]
            #     )
            #     samples_for_normalisation = 1000
            #     needs_dim = True
            #     if hasattr(cfg.dynamics, 'dist_requires_dim'):
            #         needs_dim = cfg.dynamics.dist_requires_dim
            #
            #     samples = dist.sample(
            #         sample_shape=[samples_for_normalisation] + ([cfg.dynamics.dim] if needs_dim else []))
            #     input_samples = input_distribution.sample(sample_shape=[samples_for_normalisation])
            #     samples = torch.cat((samples,input_samples),dim=-1)
            #     norm_val = float(
            #         torch.mean(torch.sum(mod_model(samples) ** 2, axis=-1)).sqrt().detach().numpy())
            #
            #     mod_model = compose(
            #         lambda x: x / norm_val,
            #         mod_model
            #     )
            #     input_to_KEF = hidden
            #     input_to_KEF = torch.cat([hidden,inputs],dim=-1)
            #     KEFval = mod_model(input_to_KEF).detach()
            #     KEFvals.append(KEFval)
            #
            #     below_threshold_points = all_below_threshold_points[i]
            #     below_threshold_mask = all_below_threshold_masks[i]
            #     hidden_reshaped = hidden.clone().detach().reshape(-1, hidden.shape[-1]).detach()
            #     hidden_reshaped[~below_threshold_mask] = torch.nan
            #     delta_hidden = torch.zeros_like(hidden_reshaped)
            #     delta_hidden[below_threshold_mask] = below_threshold_points[...,:hidden.shape[-1]] - hidden_reshaped[below_threshold_mask]
            #     delta_hidden[~below_threshold_mask] = torch.nan
            #     delta_hidden = delta_hidden.reshape(*hidden.shape)
            #     delta_hiddens.append(delta_hidden)
            #     delta_dist = torch.nanmean(
            #         (delta_hidden)**2,
            #         axis = -1
            #     )
            #     delta_dists.append(delta_dist)
            #
            #     # hidden_onlyvalid = hidden_reshaped.reshape(*hidden.shape)
            # print('len(delta_dists)',len(delta_dists))
            #
            # ### perturbation
            # scale = 1.0 #3.0
            # pert_rnn = instantiate(cfg.dynamics.perturbable_RNN_model)
            # delta_dists_st = torch.stack(delta_dists, axis=-1)
            # min_ids = np.argmin(np.nanmin(np.array(delta_dists_st), axis=-1), axis=0)
            # pert_inputs = torch.zeros((*delta_dists_st.shape[:2], hidden.shape[-1]))
            # random_pert_inputs = pert_inputs.clone()
            # for i in range(len(min_ids)):
            #     pert_vector = delta_hiddens[0][min_ids[i], i, :] * scale
            #     pert_inputs[min_ids[i]:min_ids[i]+3, i, :] = pert_vector[None]
            #     random_pert_inputs[min_ids[i]:min_ids[i]+3, i, :] = pert_vector[np.random.permutation(len(pert_vector))][None]
            # concat_inputs = torch.concat((inputs, pert_inputs), dim=-1)
            # random_concat_inputs = torch.concat((inputs, random_pert_inputs), dim=-1)
            # pert_outputs, pert_hidden = pert_rnn(concat_inputs, return_hidden=True)
            # random_pert_outputs, random_pert_hidden = pert_rnn(random_concat_inputs, return_hidden=True)
            #
            # KEFvals = torch.concatenate(KEFvals,axis=-1).detach().cpu().numpy()
            # inputs = inputs.detach().cpu().numpy()
            # targets = targets.detach().cpu().numpy()
            # outputs = outputs.detach().cpu().numpy()
            # pert_outputs = pert_outputs.detach().cpu().numpy()
            # random_pert_outputs = random_pert_outputs.detach().cpu().numpy()
            #
            #
            # fig, axes = plt.subplots(5, 10, sharex=True, sharey='row', figsize=(15, 12))
            #
            # for trial_num in range(axes.shape[1]):
            #     axs = axes[:, trial_num]
            #
            #     # Column Titles (Above First Row)
            #     axs[0].set_title(f"Trial-{trial_num}")
            #
            #     # First Row: Inputs
            #     ax = axs[0]
            #     ax.plot(inputs[:, trial_num])
            #     ax.spines['top'].set_visible(False)
            #     ax.spines['right'].set_visible(False)
            #     ax.spines['left'].set_bounds(-1, 1)
            #
            #     ax = axs[1]
            #     ax.plot(np.linalg.norm(pert_inputs[:,trial_num], axis=-1))
            #     ax.spines['top'].set_visible(False)
            #     ax.spines['right'].set_visible(False)
            #     ax.spines['left'].set_bounds(0, 1)
            #     ax.set_ylim(-0.1,1.1)
            #
            #
            #     # Second Row: Outputs/Targets
            #     # ax = axs[2]
            #     # ax.plot(targets[:, trial_num])
            #     # ax.plot(outputs[:, trial_num], ls='solid',label='No pert', alpha=0.7)
            #     # ax.plot(pert_outputs[:, trial_num], ls='dashed', label='Calc pert', alpha=0.7)
            #     # ax.plot(random_pert_outputs[:, trial_num], ls='dashed', label='Random pert', alpha=0.7)
            #     # ax.spines['top'].set_visible(False)
            #     # ax.spines['right'].set_visible(False)
            #     # ax.spines['left'].set_bounds(-1,1)
            #     # # ax.set_ylim(-0.1, 1.1)
            #
            #     # Third Row: KEF values
            #     ax = axs[3]
            #     ax.plot(KEFvals[:, trial_num])
            #     ax.spines['top'].set_visible(False)
            #     ax.spines['right'].set_visible(False)
            #     ax.set_yscale('log')
            #
            #     ## Fourth Row: dist to separatrix
            #     ax = axs[4]
            #     ax.plot(torch.stack(delta_dists,axis=-1)[:,trial_num],marker='o',markersize=1,alpha=0.5)
            #     # ax.plot(dist.log_prob(hidden[:, trial_num]).detach().cpu().numpy())
            #     # ax.set_ylabel('Log prob(hidden)')
            #     ax.spines['top'].set_visible(False)
            #     ax.spines['right'].set_visible(False)
            #     ax.set_yscale('log')
            #
            #
            #
            # # Set y-axis labels only for the first column
            # ylabel_texts = ['inputs', 'Norm of Pert input', 'outputs/targets', 'KEF values', 'Dist to Separatrix']
            # for row, label in enumerate(ylabel_texts):
            #     axes[row, 0].set_ylabel(label)
            #
            # axes[2,1].legend(fontsize=8)
            # # for ax in axes.flatten():
            # #     ax.set_xlim(230,300)
            #
            # fig.tight_layout()
            # fig.savefig(Path(cfg.savepath) / "RNN_task_KEFvals_sweep.png", dpi=300)
            # plt.close(fig)



            # P = PCA(n_components=3)
            # pc_hidden = P.fit_transform(hidden.reshape(-1,hidden.shape[-1]).detach().cpu().numpy())
            # pc_hidden = pc_hidden.reshape(*hidden.shape[:-1],P.n_components)
            #
            #
            # # fig,ax = plt.subplots()
            # fig = plt.figure(figsize=(6, 5))
            # ax = fig.add_subplot(111, projection='3d')
            #
            # n_lines = pc_hidden.shape[1]
            # import matplotlib.cm as cm
            # colors = cm.Purples(np.linspace(0.3, 0.9, n_lines))
            #
            # # fig, ax = plt.subplots()
            # for i in range(n_lines):
            #     ax.plot(*pc_hidden[100:, i, :].T, color=colors[i])
            #
            # for i in range(SL.num_models):
            #     below_threshold_points = all_below_threshold_points[i].detach().cpu().numpy()
            #     below_threshold_points = below_threshold_points[~np.isnan(below_threshold_points).any(axis=-1)]
            #     if len(below_threshold_points) == 0:
            #         continue
            #     pc_below_threshold_points = P.transform(below_threshold_points)
            #     ax.scatter(*pc_below_threshold_points[:,:].T,c=f'C{i}',s=10)
            #
            #
            # pc_unique_fps = P.transform(unique_fps.xstar)
            # ax.scatter(*pc_unique_fps[unique_fps.is_stable, :].T, c='blue',marker='x',s=100,zorder=1001)
            # ax.scatter(*pc_unique_fps[~unique_fps.is_stable, :].T, c='red',marker='x',s=100,zorder=1000) #
            #
            # fig.tight_layout()
            # fig.savefig(Path(cfg.savepath) / "trajectory_PCA.png", dpi=300)
            # # plt.close(fig)
            #
            # ##### Function to rotate the plot ######
            # def rotate(angle):
            #     ax.view_init(elev=30, azim=angle)
            #
            # # Create animation
            # num_frames = 360  # Number of frames for a full rotation
            # rotation_animation = animation.FuncAnimation(fig, rotate, frames=num_frames, interval=1000 / 30)
            #
            # # Save the animation to a file
            # rotation_animation.save(Path(cfg.savepath) / 'PCA_3d_rotation.mp4', writer='ffmpeg', fps=30, dpi=100)
            # plt.close(fig)


def main(cfg):
    omegaconf_resolvers()
    cfg.savepath = os.path.join(project_path, cfg.savepath)


    OmegaConf.resolve(cfg.model)

    print(OmegaConf.to_yaml(cfg))

    F = instantiate(cfg.dynamics.function)
    dist = instantiate(cfg.dynamics.IC_distribution)
    model = instantiate(cfg.model)

    print(model)
    print(dict(model.named_parameters()))

    path = Path(cfg.savepath)
    path.mkdir(parents=True,exist_ok=True)

    if cfg.load_KEF_model:
        model = torch.load( path / (cfg.model.name+'_KEFmodel.torch'))
        #.load_state_dict(torch.load(os.path.join(cfg.savepath,'KEFmodel.torch'),weights_only=True))

    model.to(device)
    dist_kwargs = {}
    if hasattr(cfg.dynamics, 'dist_requires_dim'):
        dist_kwargs = {'dist_requires_dim': cfg.dynamics.dist_requires_dim}


    if hasattr(cfg,'train_func_teacher'):
        if hasattr(cfg.dynamics, 'analytical_eigenfunction'):
            train_func_teacher = instantiate(cfg.train_func_teacher)
            model.train()
            train_func_teacher(
                model,
                instantiate(cfg.dynamics.analytical_eigenfunction),
                dist,
                **dist_kwargs
            )

    if hasattr(cfg,'train_func'):
        train_func = instantiate(cfg.train_func)
        model.train()
        param_specific_hyperparams = []
        if hasattr(cfg.model, 'param_specific_hyperparams'):
            param_specific_hyperparams = instantiate(cfg.model.param_specific_hyperparams)
        train_func(
            model,
            F,
            dist,
            device=device,
            param_specific_hyperparams=param_specific_hyperparams,
            **dist_kwargs
        )
        model.eval()
    model.to('cpu')

    if hasattr(cfg,'train_func_trajectories'):
        train_func_trajectories = instantiate(cfg.train_func_trajectories)
        needs_dim = True
        if hasattr(cfg.dynamics,'dist_requires_dim'):
            needs_dim = cfg.dynamics.dist_requires_dim

        initial_conditions = dist.sample(sample_shape=[cfg.train_trajectories]+([cfg.dynamics.dim] if needs_dim else []))
        print(initial_conditions.shape)
        times = torch.linspace(0, cfg.train_trajectory_duration, cfg.train_points_per_trajectory)

        trajectories = odeint(lambda t, x: F(x), initial_conditions, times)
        trajectories = trajectories.swapaxes(0,1)
        train_func_trajectories(trajectories,model,times)

    if cfg.save_KEF_model:
        torch.save(model, path / (cfg.model.name+'_KEFmodel.torch'))

    test_func = instantiate(cfg.test_func)
    # with torch.no_grad():
    test_losses = torch.stack([test_func(model,F,dist,**dist_kwargs) for _ in range(20)])
    test_losses_mean = torch.mean(test_losses,axis=0).detach().cpu().numpy()
    test_losses_std = torch.std(test_losses,axis=0).detach().cpu().numpy()
    if cfg.save_results:
        results = {
            'test_losses_mean': list(test_losses_mean),
            'test_losses_std': list(test_losses_std),
            'test_loss_type': cfg.test_func.normaliser._target_,
            'model_name'    : cfg.model.name,
            **cfg.hyperparams_to_record_in_results
        }
        pd.DataFrame(results).to_csv(path / (cfg.model.name+'_results.csv'))

    GD_on_KEF_trajectories = None
    KEFvalues_GDtraj = None
    below_threshold_points = None
    KEFvalues_below_threshold_points = None

    model_to_GD_on = compose(
        # lambda x: x ** 0.1,
        # lambda x: x ** 2,
        torch.log,
        lambda x: x + 1,
        torch.exp,
        partial(torch.sum, dim=-1, keepdims=True),
        torch.log,
        torch.abs,
        model
    )

    samples_for_normalisation = 1000
    needs_dim = True
    if hasattr(cfg.dynamics, 'dist_requires_dim'):
        needs_dim = cfg.dynamics.dist_requires_dim

    samples = dist.sample(sample_shape=[samples_for_normalisation] + ([cfg.dynamics.dim] if needs_dim else []))
    norm_val = float(torch.mean(torch.sum(model_to_GD_on(samples)**2,axis=-1)).sqrt().detach().numpy())

    model_to_GD_on = compose(
        lambda x: x/norm_val,
        model_to_GD_on
    )

    if cfg.plot_KEF_of_traj:
        needs_dim = True
        if hasattr(cfg.dynamics,'dist_requires_dim'):
            needs_dim = cfg.dynamics.dist_requires_dim
        initial_conditions = dist.sample(sample_shape=[100]+([cfg.dynamics.dim] if needs_dim else []))
        times = torch.linspace(0, 5, 50)
        trajectories = odeint(lambda t, x: F(x), initial_conditions, times)
        fig, ax = plt.subplots()
        model_eval_phi_t = compose(
            # torch.log,
            # lambda x: x + 1,
            # torch.exp,
            partial(torch.sum, dim=-1, keepdims=True),
            torch.log,
            torch.abs,
            model
        )
        phi_vals = model_eval_phi_t(
            trajectories.reshape(-1,trajectories.shape[-1])
        ).reshape(*trajectories.shape[:2],-1).detach().cpu().numpy()
        print(phi_vals.shape)
        for i in range(trajectories.shape[1]):
            ax.plot(times, phi_vals[:, i, 0])
        # ax.plot([0, 5], [0, 5], ls='dashed', color='black')
        ax.set_ylabel(r'$\log \phi(t)$')
        ax.set_xlabel(r'$t$')
        plt.savefig( path / 'phi_t.png', dpi=300)

    if cfg.runGD:
        print('Running gradient descent on KEF landscape.')
        GD_on_KEF = instantiate(cfg.GD_on_KEF)

        GD_on_KEF_trajectories,below_threshold_points = GD_on_KEF(
            model_to_GD_on,
            dist,
            dist_needs_dim = (cfg.dynamics.dist_requires_dim if hasattr(cfg.dynamics,'dist_requires_dim') else True),
        )
        KEFvalues_GDtraj = model_to_GD_on(GD_on_KEF_trajectories.reshape(-1,GD_on_KEF_trajectories.shape[-1])).detach().cpu().numpy().reshape(*GD_on_KEF_trajectories.shape[:-1],-1)
        # KEFvalues_below_threshold_points = model_to_GD_on(below_threshold_points).detach().cpu().numpy()
        print('Gradient descent on KEF landscape complete.')
        print(f'Found {below_threshold_points.shape[0]} points below threshold.')

    print("Test loss:",test_losses_mean)
    if hasattr(cfg.dynamics,'analytical_eigenfunction'):
        analytical_eigenfunction = instantiate(cfg.dynamics.analytical_eigenfunction)
        test_loss_analytical = torch.mean(torch.tensor([test_func(analytical_eigenfunction, F, dist) for _ in range(20)]))
        print("Test loss analytical:", test_loss_analytical)


    if hasattr(cfg.dynamics,'analytical_eigenfunction'):
        num_trials = 10
        initial_conditions = dist.sample(sample_shape=(num_trials, cfg.dynamics.dim))
        times = torch.linspace(0, 10, 1000)

        trajectories = odeint(lambda t,x: F(x), initial_conditions, times)
        # trajectories = trajectories.detach().cpu().numpy()

        analytical_eigenfunction = instantiate(cfg.dynamics.analytical_eigenfunction)
        fig, ax = plt.subplots()
        phi_vals = analytical_eigenfunction(trajectories).detach().cpu() #.numpy()
        for i in range(trajectories.shape[1]):
            ax.plot(times, np.log(phi_vals[:, i, 0]))
        ax.plot([0, 5], [0, 5], ls='dashed', color='black')
        ax.set_ylabel(r'$\phi(t)$')
        ax.set_xlabel(r'$t$')
        plt.savefig(path / 'analytical_phi_t.png', dpi=300)
        plt.close(fig)


        if cfg.runGD_analytical:
            fig, axs = plt.subplots(2, 2, figsize=(8, 8),sharex=True)

            GD_on_KEF = instantiate(cfg.GD_on_KEF)

            mu_vals = [-0.5,0,0.5,1]
            for i,mu in enumerate(mu_vals):
                ax=axs.flatten()[i]
                cfg.dynamics.analytical_eigenfunction.mu = mu
                analytical_eigenfunction = instantiate(cfg.dynamics.analytical_eigenfunction)
                trajectories = GD_on_KEF(analytical_eigenfunction, dist)
                ax.plot(trajectories[...,0], trajectories[...,1], color='grey', alpha=0.3)
                ax.scatter(trajectories[0, :, 0], trajectories[0, :, 1], color='green', alpha=0.3)
                ax.scatter(trajectories[-1, :, 0], trajectories[-1, :, 1], color='red')
                ax.set_title(rf'$\mu={mu}$')

            x_limits = (-2, 2)  # Limits for x-axis
            y_limits = (-2, 2)  # Limits for y-axis
            if hasattr(cfg.dynamics, 'lims'):
                x_limits = cfg.dynamics.lims.x
                y_limits = cfg.dynamics.lims.y
            for ax in axs.flatten():
                ax.set_xlim(*x_limits)
                ax.set_ylim(*y_limits)
            fig.tight_layout()
            fig.savefig(path / 'GD_on_KEF_trajectories.png', dpi=300)


    if not cfg.run_analysis:
        return

    if cfg.dynamics.dim == 1:
        x = torch.linspace(-15, 15, 1000,dtype=torch.float32)
        x.requires_grad_(True)
        phi_val = model(x[:, None])
        F_val = F(x[:, None])

        F_val = F_val.detach().cpu().numpy()[...,0]


        phi_x_prime = torch.autograd.grad(
            outputs=phi_val[:,:].sum(axis=-1),
            inputs=x,
            grad_outputs=torch.ones_like(phi_val[:,0]),
            create_graph= True
        )[0]
        phi_val = phi_val.detach().cpu().numpy()[...,0]
        x = x.detach().cpu().numpy()
        phi_x_prime = phi_x_prime.detach().cpu().numpy()

        fig,axs = plt.subplots(4 + int(below_threshold_points is not None),1,figsize=(4,7),sharex=True)
        ax = axs[0]
        ax.plot(x, 0 * x, c='grey', lw=1)
        ax.plot(x, F_val, label='F')

        ax = axs[1]
        ax.plot(x, np.abs(phi_val)/np.sqrt((phi_val**2).mean()), label=f'$\phi$')
        if hasattr(cfg.dynamics, 'analytical_eigenfunction'):
            analytical_eigenfunction = instantiate(cfg.dynamics.analytical_eigenfunction)
            ana_phi_val = analytical_eigenfunction(x)
            ax.plot(x,np.abs(ana_phi_val)/np.sqrt(ana_phi_val**2).mean(),ls='dashed',color='black',alpha=0.5)

        ax = axs[2]
        print(phi_x_prime.shape,F_val.shape,phi_val.shape)
        diff = (phi_x_prime * F_val) - 1 * phi_val
        diff /= diff.std()
        ax.plot(
            x,
            # (phi_x_prime*F_val.flatten())[...,None]-1*phi_val,
            np.abs(diff),
            label=r'$\nabla \phi-\lambda \phi$',
            lw=1
        )

        ax = axs[3]
        # print(phi_x_prime.shape,F_val.shape,phi_val.shape)
        dot_prods = (phi_x_prime * F_val) # - 1 * phi_val

        ax.plot(
            x,
            np.abs(dot_prods),
            label=r'$\nabla \phi \cdot F$',
            lw=1,
            c='red'
        )
        ax.plot(
            x,
            np.abs(phi_val),
            label=r'$\phi$',
            lw=1,
            c='blue'
        )
        # ax.set_yscale('log')
        ax.legend()


        if below_threshold_points is not None:
            ax = axs[4]
            ax.hist(below_threshold_points,density=True,bins=100)

        # plt.legend()
        for ax in axs.flatten():
            ax.set_xlim(-5, 5)
            ax.set_ylim(0, 5)
        axs[0].set_ylim(-2,2)
        fig.tight_layout()
        fig.savefig(path / 'F_and_phi.png',dpi=300)
        plt.close(fig)


        if model.__class__.__name__ == "RBFLayer":
            fig,ax = plt.subplots()
            D = pd.DataFrame({
                'kernels_centers': model.get_kernels_centers.detach().cpu().numpy().flatten(),
                'weights': torch.abs(model.get_weights.detach().cpu()).numpy().flatten(),
                'log shapes': torch.log(model.get_shapes.detach()).cpu().numpy().flatten(),
                'shapes': (model.get_shapes.detach()).cpu().numpy().flatten(),
            })
            sns.scatterplot(x='kernels_centers',y='log shapes',size='weights',data=D,ax=ax)
            fig.savefig(path / 'RBF_data.png',dpi=300)
            # plt.show()


    elif cfg.dynamics.dim == 2:
        if KEFvalues_GDtraj is not None:
            fig,ax = plt.subplots()
            t = np.arange(KEFvalues_GDtraj.shape[0])
            ax.plot(t,KEFvalues_GDtraj[:,:,0])
            fig.tight_layout()
            fig.savefig(path / 'KEFvalues_GDtraj.png')



        # Configurable parameters
        heatmap_resolution = 500  # Resolution for the heatmap
        quiver_resolution = 25  # Resolution for the quiver plot
        laplace_resolution = 50

        x_limits = (-2, 2)  # Limits for x-axis
        y_limits = (-2, 2)  # Limits for y-axis
        if hasattr(cfg.dynamics,'lims'):
            x_limits = cfg.dynamics.lims.x
            y_limits = cfg.dynamics.lims.y

        num_trials = 100
        initial_conditions = torch.concat([
            torch.rand(size=(num_trials, 1)) * (x_limits[1] - x_limits[0]) + x_limits[0],
            torch.rand(size=(num_trials, 1)) * (y_limits[1] - y_limits[0]) + y_limits[0]
        ],axis=-1)
        times = torch.linspace(0, 10, 100)
        trajectories = None
        if hasattr(cfg.dynamics,'run_traj'):
            if cfg.dynamics.run_traj:
                trajectories = odeint(lambda t,x: F(x), initial_conditions, times)
                trajectories = trajectories.detach().cpu().numpy()

        # Define grid for heatmap (higher resolution)
        x_heatmap = torch.linspace(x_limits[0], x_limits[1], heatmap_resolution)
        y_heatmap = torch.linspace(y_limits[0], y_limits[1], heatmap_resolution)
        X_heatmap, Y_heatmap = torch.meshgrid(x_heatmap, y_heatmap, indexing='ij')
        heatmap_grid = torch.stack([X_heatmap.flatten(), Y_heatmap.flatten()], dim=-1)

        # Define grid for laplace (higher resolution)
        x_laplace = torch.linspace(x_limits[0], x_limits[1], laplace_resolution)
        y_laplace = torch.linspace(y_limits[0], y_limits[1], laplace_resolution)
        X_laplace, Y_laplace = torch.meshgrid(x_laplace, y_laplace, indexing='ij')
        laplace_grid = torch.stack([X_laplace.flatten(), Y_laplace.flatten()], dim=-1)

        # Define grid for quiver plot (lower resolution)
        x_quiver = torch.linspace(x_limits[0], x_limits[1], quiver_resolution)
        y_quiver = torch.linspace(y_limits[0], y_limits[1], quiver_resolution)
        X_quiver, Y_quiver = torch.meshgrid(x_quiver, y_quiver, indexing='ij')
        quiver_grid = torch.stack([X_quiver.flatten(), Y_quiver.flatten()], dim=-1)

        # Evaluate trajectories from IC and compute Laplace integral
        tau = 0.1
        # times = torch.linspace(0, 5, 100)
        # laplace_trajectories = odeint(lambda t,x: F(x), laplace_grid, times,rtol=1e-10, atol=1e-10)
        # observable = lambda x: x[...,1]
        center = torch.tensor([0.0,-6.0])
        center = torch.tensor([0.0, 3.0])
        observable = lambda x: (torch.log((x - center[None,None,:])**2)).mean(axis=-1)
        # laplace_integrals = (observable(laplace_trajectories) * torch.exp(-tau * times[:,None])).mean(axis=0)
        # laplace_integrals = laplace_integrals.detach().cpu().numpy().reshape(laplace_resolution, laplace_resolution)

        # Compute F values for the quiver grid
        F_val_quiver = F(quiver_grid).detach().cpu().numpy()
        Fx_quiver = F_val_quiver[:, 0].reshape(quiver_resolution, quiver_resolution)
        Fy_quiver = F_val_quiver[:, 1].reshape(quiver_resolution, quiver_resolution)

        # Compute F values for the heatmap grid (for kinetic energy)
        F_val_heatmap = F(heatmap_grid).detach().cpu().numpy()
        Fx_heatmap = F_val_heatmap[:, 0].reshape(heatmap_resolution, heatmap_resolution)
        Fy_heatmap = F_val_heatmap[:, 1].reshape(heatmap_resolution, heatmap_resolution)
        kinetic_energy = Fx_heatmap ** 2 + Fy_heatmap ** 2

        # Compute phi values for the heatmap grid
        # phi_val = torch.prod( model(heatmap_grid),dim=-1).detach().cpu().numpy()
        phi_val = model_to_GD_on(heatmap_grid).detach().cpu().numpy()
        phi_val = phi_val.reshape(heatmap_resolution, heatmap_resolution)

        # heatmap_grid.requires_grad_(True)
        # phi_val_2 = model(heatmap_grid).detach().cpu().numpy()
        # phi_val_2_prime = torch.autograd.grad(
        #     outputs=phi_val_2.sum(axis=-1),
        #     inputs=heatmap_grid,
        #     grad_outputs=torch.ones_like(phi_val_2.sum(axis=-1)),
        #     create_graph=False  # True
        # )[0]
        # phi_val_2_prime


        # Set up the figure with two side-by-side subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        # Left subplot: Kinetic energy of F with quiver plot
        im1 = axes[0].imshow(
            np.log(kinetic_energy).T, extent=[x_limits[0], x_limits[1], y_limits[0], y_limits[1]],
            origin='lower', aspect='auto', cmap='plasma'
        )
        axes[0].quiver(
            X_quiver, Y_quiver, Fx_quiver, Fy_quiver, color='white', scale=50, width=0.002, headwidth=3,
            pivot='middle'
        )
        if trajectories is not None:
            axes[0].plot(trajectories[..., 0], trajectories[..., 1],c='grey', lw=1)
        axes[0].set_title('Kinetic Energy and Vector Field of $F(x, y)$')
        axes[0].set_xlabel('$x$')
        axes[0].set_ylabel('$y$')
        if below_threshold_points is not None:
            # print('below_threshold_points',below_threshold_points)
            axes[0].scatter(below_threshold_points[ :, 0],
                            below_threshold_points[ :, 1], c='red', zorder=1000)
        fig.colorbar(im1, ax=axes[0], label='Kinetic Energy $||F(x, y)||^2$')

        # Right subplot: Heatmap of phi with contour for phi(x, y) = 0
        # im2 = axes[1].imshow(
        #     phi_val.T, extent=[x_limits[0], x_limits[1], y_limits[0], y_limits[1]],
        #     origin='lower', aspect='auto', cmap='viridis'
        # )
        # contour = axes[1].contour(
        #     X_heatmap, Y_heatmap, phi_val, levels=[0,1,2,3], colors='red', linewidths=2
        # )
        contour = axes[1].contourf(
            X_heatmap, Y_heatmap, phi_val,
        )
        # if GD_on_KEF_trajectories is not None:
        #     axes[1].plot(GD_on_KEF_trajectories[..., 0],
        #                  GD_on_KEF_trajectories[..., 1], c='grey', lw=0.5,alpha=0.3)
        #     axes[1].scatter(GD_on_KEF_trajectories[-1, :, 0],
        #                GD_on_KEF_trajectories[-1, :, 1], c='red',zorder=1000)
        if below_threshold_points is not None:
            # print('below_threshold_points',below_threshold_points)
            axes[1].scatter(below_threshold_points[ :, 0],
                            below_threshold_points[ :, 1], c='red', zorder=1000)
            # print('Plotting below_threshold_points:',below_threshold_points)
        # axes[1].clabel(contour, inline=True, fontsize=10)
        axes[1].set_title('Contour plot of $\log \phi(x, y)$')
        axes[1].set_xlabel('$x$')
        axes[1].set_ylabel('$y$')


        fig.colorbar(contour, ax=axes[1], label='$\phi(x, y)$')

        contour1 = axes[2].contourf(
            X_heatmap, Y_heatmap, np.abs(phi_val) ** 0.1,
        )
        # axes[2].clabel(contour1, inline=True, fontsize=10)
        axes[2].set_title('Contour plot of $\phi(x, y)$')
        axes[2].set_xlabel('$x$')
        axes[2].set_ylabel('$y$')

        fig.colorbar(contour1, ax=axes[2], label='$\phi(x, y)^{0.1}$')

        if hasattr(cfg.dynamics,"analytical_eigenfunction"):
            analytical_phi_func = instantiate(cfg.dynamics.analytical_eigenfunction)
            phi_val = analytical_phi_func(heatmap_grid).detach().cpu().numpy()
            phi_val = phi_val.reshape(heatmap_resolution, heatmap_resolution)
            print('phi_val range',phi_val.min(), phi_val.max(), phi_val.mean())
            phi_val = np.clip(phi_val, 0, 1.0)
            label = r'$\phi^{ana}(x, y)$'
        else:
            phi_val = np.abs(phi_val)**0.05
            label = r'$\phi(x, y)^{0.05}$'



        contour2 = axes[3].contourf(
            X_heatmap, Y_heatmap, (np.abs(phi_val)),
        )
        # axes[3].clabel(contour2, inline=True, fontsize=10)
        axes[3].set_title('Contour plot of $\phi(x, y)$')
        axes[3].set_xlabel('$x$')
        axes[3].set_ylabel('$y$')

        for ax in axes:
            ax.set_xlim(*x_limits)
            ax.set_ylim(*y_limits)
        fig.colorbar(contour2, ax=axes[3], label=label)


        # plot laplace integrals
        # contour = axes[2].contourf(
        #     X_laplace, Y_laplace, laplace_integrals, linewidths=2
        # )
        # axes[2].set_title('Contour plot of Laplace integral $f^*_1(x, y)$')
        # axes[2].set_xlabel('$x$')
        # axes[2].set_ylabel('$y$')
        # for ax in axes:
        #     ax.set_xlim(*x_limits)
        #     ax.set_ylim(*y_limits)
        # fig.colorbar(contour, ax=axes[2], label='$f^*_1(x, y)$')


        # Save the figure with both subplots
        fig.tight_layout()
        fig.savefig(path / 'F_and_phi_subplots.png',dpi=300)
        fig.savefig(path / 'F_and_phi_subplots.pdf')
        plt.close(fig)

        ### Plotting all
        phi_vals = model(heatmap_grid).detach().cpu().numpy()

        if phi_vals.shape[1]>=10:
            fig,axs = plt.subplots(2,5,figsize=(10, 8),sharey=True,sharex=True)
            for i,ax in enumerate(axs.flatten()):
                phi_val = phi_vals[..., i]
                phi_val = phi_val.reshape(heatmap_resolution, heatmap_resolution)
                contour = ax.contourf(
                    X_heatmap, Y_heatmap, np.log(np.abs(phi_val)),
                )
            fig.tight_layout()
            fig.savefig(path / 'all_phi.png',dpi=300)

        x = torch.linspace(-1,1,100)
        inp = torch.stack([x,torch.zeros_like(x)], dim=-1)
        phi_val = model(inp).detach().cpu().numpy()

        plt.figure()
        plt.plot(x,phi_val, label=f'$\phi$')
        plt.xlabel(r'$x$')
        plt.ylabel(r'$\phi(x,0)$')
        plt.tight_layout()
        plt.savefig(path / 'phi(x,0).png',dpi=300)










    elif cfg.dynamics.dim > 2:
        trajectories = None
        KEFvaltraj = None
        if hasattr(cfg.dynamics, 'run_traj'):
            if cfg.dynamics.run_traj:
                num_trials = 50
                times = torch.linspace(0, 5, 100)
                needs_dim = True
                if hasattr(cfg.dynamics, 'dist_requires_dim'):
                    needs_dim = cfg.dynamics.dist_requires_dim

                initial_conditions = dist.sample(
                    sample_shape=[num_trials] + ([cfg.dynamics.dim] if needs_dim else []))

                trajectories = odeint(lambda t, x: F(x), initial_conditions, times)
                KEFvaltraj = compose(torch.abs, model)(trajectories).detach().cpu().numpy()
                trajectories = trajectories.detach().cpu().numpy()

        if trajectories is not None:
            fig,ax = plt.subplots()
            ax.plot(times, KEFvaltraj[...,0], c='grey', lw=1)
            fig.savefig(path / "KEF_of_trajectories.png",dpi=300)

        rnn_model = instantiate(cfg.dynamics.loaded_RNN_model)
        # cfg.dynamics.RNN_dataset.batch_size = 5000
        cfg.dynamics.RNN_dataset.n_trials = 1000
        dataset = instantiate(cfg.dynamics.RNN_dataset)
        inp, targ = dataset()

        torch_inp = torch.from_numpy(inp).type(torch.float) #.to(device)
        outputs,hidden_traj = rnn_model(torch_inp,return_hidden=True)
        outputs,hidden_traj = outputs.detach().cpu().numpy(), hidden_traj.detach().cpu().numpy()


        # print(model)
        fpf_hps = {
            'max_iters': 1000, #10000
            'n_iters_per_print_update': 1000,
            'lr_init': .1,
            'outlier_distance_scale': 10.0,
            'verbose': True,
            'super_verbose': True,
            # 'tol_q':1e-6,
            # 'tol_q': 1e-15,
            # 'tol_dq': 1e-15,
        }
        FPF = FixedPointFinderTorch(
            rnn_model.rnn,
            **fpf_hps
        )
        num_trials = 1000
        # initial_conditions = dist.sample(sample_shape=(num_trials,)).detach().cpu().numpy()
        inputs = np.zeros((1, cfg.dynamics.RNN_model.act_size))
        # inputs[...,0] = 1
        initial_conditions = hidden_traj.reshape(-1,hidden_traj.shape[-1])
        initial_conditions = initial_conditions[np.random.choice(initial_conditions.shape[0],size=num_trials,replace=False)]
        initial_conditions += np.random.normal(size=initial_conditions.shape) * 0.05
        # print('initial_conditions', initial_conditions.shape)
        unique_fps,all_fps = FPF.find_fixed_points(
            initial_conditions,
            inputs
        )
        # print(all_fps.shape)
        fixed_point_data = pd.DataFrame({
            'stability': unique_fps.is_stable,
            'q': unique_fps.qstar,
            'KEF': model_to_GD_on(torch.from_numpy(unique_fps.xstar).type(torch.float)).detach().cpu().numpy()[...,0],
        })
        fixed_point_data.to_csv(path / 'fixed_point_data.csv',index=False)


        P = PCA(n_components=3)
        pc_traj = P.fit_transform(hidden_traj.reshape(-1, hidden_traj.shape[-1])).reshape(*hidden_traj.shape[:-1],P.n_components)
        # pc_initial_conditions = P.fit_transform(initial_conditions)
        pc_initial_conditions = P.transform(initial_conditions)
        pc_fps = P.transform(all_fps.xstar)
        pc_unique_fps = P.transform(unique_fps.xstar)
        # where_best = (unique_fps.qstar < np.median(unique_fps.qstar))
        where_best = True
        # pc_traj = P.transform(hidden_traj.reshape(-1,hidden_traj.shape[-1])).reshape(*hidden_traj.shape[:-1],P.n_components)





        # fig,axs = plt.subplots(figsize=(4,3))
        # ax = axs
        # # ax.scatter(pc_initial_conditions[:,0],pc_initial_conditions[:,1],size=10)
        # # Plot and capture scatter plot artists
        #
        # scatter1 = ax.scatter(pc_unique_fps[unique_fps.is_stable & where_best, 0], pc_unique_fps[unique_fps.is_stable & where_best, 1], s=5,
        #                        c='C0')
        # scatter2 = ax.scatter(pc_unique_fps[(~unique_fps.is_stable) & where_best, 0], pc_unique_fps[(~unique_fps.is_stable) & where_best, 1], s=5,
        #                       c='C1')
        #
        # ax.plot(pc_traj[:, :100, 0], pc_traj[:, :100, 1], lw=0.5, c='grey', alpha=0.1)
        #
        # where_decide_1 = np.argmax(outputs, axis=-1) == 1
        # where_decide_2 = np.argmax(outputs, axis=-1) == 2
        #
        #
        # scatter3 = ax.scatter(pc_traj[..., 0][where_decide_1][:100], pc_traj[..., 1][where_decide_1][:100], c='C3',s=10)
        # scatter4 = ax.scatter(pc_traj[..., 0][where_decide_2][:100], pc_traj[..., 1][where_decide_2][:100], c='C4',s=10)
        #
        # if GD_on_KEF_trajectories is not None:
        #     GD_on_KEF = instantiate(cfg.GD_on_KEF)
        #     GD_on_KEF_trajectories = GD_on_KEF(
        #         compose(torch.abs, model),
        #         dist,
        #         initial_conditions = torch.from_numpy(initial_conditions).type(torch.float),
        #     )
        #
        #
        #     pc_GD_on_KEF_trajectories = P.transform(GD_on_KEF_trajectories.reshape(-1,GD_on_KEF_trajectories.shape[-1])).reshape(*GD_on_KEF_trajectories.shape[:-1],P.n_components)
        #     ax.plot(pc_GD_on_KEF_trajectories[:, :, 0], pc_GD_on_KEF_trajectories[:, :, 1], c='C5',lw=0.5,alpha=0.5)
        #     scatter5 = ax.scatter(pc_GD_on_KEF_trajectories[-1,:,0],pc_GD_on_KEF_trajectories[-1,:,1],c='C5',s=10)

        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')

        # Plot and capture scatter plot artists in 3D
        scatter1 = ax.scatter(
            pc_unique_fps[unique_fps.is_stable & where_best, 0],
            pc_unique_fps[unique_fps.is_stable & where_best, 1],
            pc_unique_fps[unique_fps.is_stable & where_best, 2],
            s=5, c='C0'
        )
        scatter2 = ax.scatter(
            pc_unique_fps[(~unique_fps.is_stable) & where_best, 0],
            pc_unique_fps[(~unique_fps.is_stable) & where_best, 1],
            pc_unique_fps[(~unique_fps.is_stable) & where_best, 2],
            s=5, c='C1'
        )

        ax.plot(
            pc_traj[:, :10, 0],
            pc_traj[:, :10, 1],
            pc_traj[:, :10, 2],
            lw=0.5, c='grey', alpha=0.5
        )

        # where_decide_1 = np.argmax(outputs, axis=-1) == 1
        # where_decide_2 = np.argmax(outputs, axis=-1) == 2

        # scatter3 = ax.scatter(
        #     pc_traj[..., 0][where_decide_1][:100],
        #     pc_traj[..., 1][where_decide_1][:100],
        #     pc_traj[..., 2][where_decide_1][:100],
        #     c='C3', s=10
        # )
        # scatter4 = ax.scatter(
        #     pc_traj[..., 0][where_decide_2][:100],
        #     pc_traj[..., 1][where_decide_2][:100],
        #     pc_traj[..., 2][where_decide_2][:100],
        #     c='C4', s=10
        # )




        # Handle GD_on_KEF_trajectories if not None
        if GD_on_KEF_trajectories is not None:
            # GD_on_KEF = instantiate(cfg.GD_on_KEF)
            # GD_on_KEF_trajectories = GD_on_KEF(
            #     compose(torch.abs, model),
            #     dist,
            #     initial_conditions=torch.from_numpy(initial_conditions).type(torch.float),
            # )
            final_KEFvals = compose(torch.log,torch.abs, model)(GD_on_KEF_trajectories[-1]).detach()
            select_best = final_KEFvals.flatten()<torch.quantile(final_KEFvals,.05)
            print(
                'Max KEF val', final_KEFvals.max(),
                'Min KEF val', final_KEFvals.min(),
                'Median KEF val', torch.quantile(final_KEFvals,.05),
            )

            pc_GD_on_KEF_trajectories = P.transform(
                GD_on_KEF_trajectories.reshape(-1, GD_on_KEF_trajectories.shape[-1])).reshape(
                *GD_on_KEF_trajectories.shape[:-1], P.n_components
            )

            if trajectories is not None:
                pc_trajectories = P.transform(
                    trajectories.reshape(-1, trajectories.shape[-1])).reshape(
                    *trajectories.shape[:-1], P.n_components
                )
                for i in np.arange(pc_trajectories.shape[1]):
                    traj_l,=ax.plot(
                        pc_trajectories[:, i, 0],
                        pc_trajectories[:, i, 1],
                        pc_trajectories[:, i, 2],
                        c='C6', lw=0.5, alpha=0.5
                    )

            # for i in np.where(select_best)[0]:
            #     ax.plot(
            #         pc_GD_on_KEF_trajectories[[0,-1], i, 0],
            #         pc_GD_on_KEF_trajectories[[0,-1], i, 1],
            #         pc_GD_on_KEF_trajectories[[0,-1], i, 2],
            #         c='C5', lw=0.5, alpha=0.5
            #     )

            # scatter5 = ax.scatter(
            #     pc_GD_on_KEF_trajectories[-1, select_best, 0],
            #     pc_GD_on_KEF_trajectories[-1, select_best, 1],
            #     pc_GD_on_KEF_trajectories[-1, select_best, 2],
            #     c='C5', s=10,alpha=0.5
            # )
            if below_threshold_points is not None:
                pc_below_threshold_points = P.transform(below_threshold_points)
                scatter5 = ax.scatter(
                    pc_below_threshold_points[:, 0],
                    pc_below_threshold_points[:, 1],
                    pc_below_threshold_points[:, 2],
                    c='red', s=10,alpha=0.5
                )
            # scatter5 = ax.scatter(
            #     pc_GD_on_KEF_trajectories[-1, select_best, 0],
            #     pc_GD_on_KEF_trajectories[-1, select_best, 1],
            #     pc_GD_on_KEF_trajectories[-1, select_best, 2],
            #     c='red', s=10, alpha=0.5
            # )


        # Add legend using scatter plot artists
        ax.legend(
            [scatter1, scatter2], #, scatter5],#, scatter3, scatter4],
            ['Stable Fixed Points', 'Unstable Fixed Points','KEF minima'], #'Report Decision 1', 'Report Decision 2'
            loc='best',
            fontsize='small'
        )

        # Add legend to the plot
        # ax.legend(handles=legend_handles, loc='best', fontsize='small')
        fig.tight_layout()

        fig.savefig(path / 'PCA_3d.png',dpi=300)




        ##### Function to rotate the plot ######
        def rotate(angle):
            ax.view_init(elev=30, azim=angle)

        # Create animation
        num_frames = 360  # Number of frames for a full rotation
        rotation_animation = animation.FuncAnimation(fig, rotate, frames=num_frames, interval=1000/30)

        # Save the animation to a file
        rotation_animation.save(path / 'PCA_3d_rotation.mp4', writer='ffmpeg', fps=30, dpi=100)
        plt.close(fig)



        # fig,ax = plt.subplots(figsize=(4,3))
        # ax.hist(unique_fps.qstar, bins=np.logspace(-7,-3,50))
        # ax.axvline(np.median(unique_fps.qstar), color='black',linestyle='--',lw=1,label='median')
        # ax.legend()
        # ax.set_xlabel(r'$q$')
        # ax.set_xscale('log')
        # fig.tight_layout()
        # fig.savefig(path / 'qstar_hist.png',dpi=200)

        # fig,ax = plt.subplots(figsize=(4,3))
        # bins = np.linspace(-3.5,2.5,101)
        # ax.hist(pc_traj[..., 0][where_decide_1|where_decide_2], bins=bins, color='C0')  # , bins=np.logspace(-7,-3,50))
        # # ax.hist(pc_traj[..., 0][where_decide_1], bins=bins, label='Report decision 1', color='C3',alpha=0.5) #, bins=np.logspace(-7,-3,50))
        # # ax.hist(pc_traj[..., 0][where_decide_2], bins=bins, label='Report decision 2', color='C4',alpha=0.5)  # , bins=np.logspace(-7,-3,50))
        # # ax.axvline(np.median(unique_fps.qstar), color='black',linestyle='--',lw=1,label='median')
        # # ax.legend()
        # ax.set_xlabel(r'PC1 decision points')
        # # ax.set_xscale('log')
        # fig.tight_layout()
        # fig.savefig(path / 'PC1_decision_hist.png',dpi=200)




        # # A = F.functions[0].keywords['A']
        # # initial_conditions = 15 * torch.randn(size=(num_trials, A.shape[-1]))[:,:2] @ A.T[:2]
        # initial_conditions = dist.sample(sample_shape=(num_trials, cfg.dynamics.dim))
        # times = torch.linspace(0, 2, 500)
        # trajectories = odeint(lambda t, x: F(x), initial_conditions, times).detach()
        # # print(torch.mean(trajectories**2,dim=(-1))[-1])
        # # print(trajectories[-1,0,:])
        #
        # # Reshape the data for PCA (combine timesteps and trials into one axis)
        # data = trajectories.reshape(-1, trajectories.shape[-1]).numpy()  # Shape: [6000, 20]
        #
        # # Perform PCA to reduce the dimensionality to 2 for 2D visualization
        # pca = PCA(n_components=2)
        # # data_pca = pca.fit_transform(data)  # Shape: [6000, 2]
        # pca.fit(initial_conditions.detach().cpu().numpy())
        # data_pca = pca.transform(data)
        #
        # # Reshape back to [500, 100, 2] for plotting
        # data_pca = data_pca.reshape(times.shape[0], num_trials, 2)
        #
        # # Define the range for the PCA plane
        # pc1_min, pc1_max = np.min(data_pca[:, :, 0]), np.max(data_pca[:, :, 0])
        # pc2_min, pc2_max = np.min(data_pca[:, :, 1]), np.max(data_pca[:, :, 1])
        #
        # # Calculate the range for each principal component
        # pc1_range = pc1_max - pc1_min
        # pc2_range = pc2_max - pc2_min
        #
        # # Extend the range by 5%
        # pc1_min = pc1_min - 0.05 * pc1_range
        # pc1_max = pc1_max + 0.05 * pc1_range
        # pc2_min = pc2_min - 0.05 * pc2_range
        # pc2_max = pc2_max + 0.05 * pc2_range
        #
        # # Create a grid in the PCA space
        # resolution = 100
        # pc1 = np.linspace(pc1_min, pc1_max, resolution)
        # pc2 = np.linspace(pc2_min, pc2_max, resolution)
        # PC1, PC2 = np.meshgrid(pc1, pc2)
        #
        # # Flatten the grid for evaluation
        # pc_points = np.stack([PC1.ravel(), PC2.ravel()], axis=1)  # Shape: [resolution^2, 2]
        #
        # # Map PCA points back to the original N-dimensional space
        # original_space_points = pca.inverse_transform(pc_points)  # No need for zero-padding
        #
        # # Ensure the model evaluates only the grid points
        # with torch.no_grad():
        #     model_output = model(torch.tensor(original_space_points, dtype=torch.float32)).numpy()
        #
        # # Ensure the model output size matches the grid
        # assert model_output.size == PC1.size, f"Expected {PC1.size}, but got {model_output.size}"
        #
        # # Reshape model output to match the grid
        # model_output = model_output.reshape(PC1.shape)
        #
        # # print(model_output.min())
        #
        # # Plot the trajectories in the first two principal components
        # plt.figure(figsize=(10, 8))
        #
        # levels = np.arange(0,1.5,0.1) #np.linspace(model_output.min(),model_output.max(),4)
        # # Plot the contour of the model
        # zero_contour = plt.contour(
        #     PC1, PC2, model_output, levels=[0], colors='red', linewidths=2
        # )
        # plt.clabel(zero_contour, inline=True, fontsize=10)
        # contour = plt.contourf(PC1, PC2, np.abs(model_output), levels=levels, cmap='viridis', alpha=0.7)
        # plt.colorbar(contour, label=r"$|\phi|$")
        #
        # for trial in range(data_pca.shape[1]):  # Iterate over trials
        #     plt.plot(
        #         data_pca[:, trial, 0],  # PC1
        #         data_pca[:, trial, 1],  # PC2
        #         alpha=0.7,
        #         c='grey',
        #         lw=1
        #     )
        #     plt.scatter(
        #         data_pca[0, trial, 0],  # PC1
        #         data_pca[0, trial, 1],  # PC2
        #         c='green'
        #     )
        #     plt.scatter(
        #         data_pca[-1, trial, 0],  # PC1
        #         data_pca[-1, trial, 1],  # PC2
        #         c='red'
        #     )
        #
        # # Set labels and title
        # plt.xlabel("PC1")
        # plt.ylabel("PC2")
        # plt.title("Trajectories and Model Contour in PCA-reduced Space (2D)")
        #
        # # Save the plot
        # plt.savefig(path / 'trajectories_with_contour.png', dpi=300)
        # plt.savefig(path / 'trajectories_with_contour.pdf')


if __name__ == '__main__':
    decorated_main()

