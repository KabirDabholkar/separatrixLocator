import torch
from torchdiffeq import odeint
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def find_separatrix_point_along_line(dynamics_function, external_input, attractors, num_points=5, num_iterations=5, time_points=None, return_all_points=False, final_time=5000):
    """
    Find a point on the separatrix along the line between two attractors by iteratively refining the search space.
    
    Args:
        dynamics_function: Function that defines the system dynamics
        external_input: External input tensor
        attractors: Tuple of two attractor points
        num_points: Number of points to sample along the line between attractors
        num_iterations: Number of refinement iterations
        time_points: Time points for trajectory integration. If None, defaults to [0, 5000]
        return_all_points: If True, returns all points along the line. If False (default), returns only the mean point.
        
    Returns:
        If return_all_points is False (default):
            mean_point: The mean of the final refined points along the line
            trajectories: Trajectories from the final points
            labels: Cluster labels for the final points
        If return_all_points is True:
            current_points: All final refined points along the line
            trajectories: Trajectories from the final points
            labels: Cluster labels for the final points
    """
    if time_points is None:
        time_points = torch.linspace(0, final_time, 2)
        
    attractor1, attractor2 = attractors
    current_points = attractor1 + torch.linspace(0, 1, num_points).unsqueeze(-1) * (attractor2 - attractor1)
    current_t_values = torch.linspace(0, 1, num_points)
    
    for iteration in range(num_iterations):
        # Run trajectories for all current points in batch
        with torch.no_grad():
            if external_input is None:
                trajectories = odeint(lambda t, x: dynamics_function(x), current_points, time_points).detach().cpu()
            else:
                trajectories = odeint(lambda t, x: dynamics_function(x, external_input[None].repeat(num_points,1)), current_points, time_points).detach().cpu()
        
        # Get final points and perform k-means clustering
        final_points = trajectories[-1]
        kmeans = KMeans(n_clusters=2, random_state=0).fit(final_points)
        labels = kmeans.labels_
        
        # Find the index where the trajectory switches from one cluster to another
        switch_idx = None
        for i in range(len(current_points)-1):
            if labels[i] != labels[i+1]:
                switch_idx = i
                break
        
        if switch_idx is None:
            print(f"Could not find switch point in iteration {iteration}")
            break
            
        # Create new grid between the switching points
        new_t_values = torch.linspace(current_t_values[switch_idx], 
                                    current_t_values[switch_idx+1], 
                                    num_points)
        current_points = attractor1 + new_t_values.unsqueeze(-1) * (attractor2 - attractor1)
        current_t_values = new_t_values
        
        print(f"Iteration {iteration}: Found switch between points {switch_idx} and {switch_idx+1}")
    
    if return_all_points:
        return current_points, trajectories, labels
    else:
        return torch.mean(current_points, dim=0)

def plot_separatrix_trajectories(trajectories, labels, points, attractors, num_points):
    """
    Plot the trajectories and points along the line between attractors.
    
    Args:
        trajectories: List of trajectories
        labels: Cluster labels for the points
        points: Points along the line between attractors
        attractors: Tuple of two attractor points
        num_points: Number of points used
    """
    plt.figure(figsize=(5,5))
    colors = plt.cm.viridis(torch.linspace(0, 1, num_points))  # Color gradient for trajectories
    
    # Plot trajectories from find_separatrix_point_along_line
    for i, (traj, label) in enumerate(zip(trajectories, labels)):
        plt.plot(traj[:, 0], traj[:, 1], '--', color='gray', alpha=0.3)
        plt.scatter(traj[0, 0], traj[0, 1], c='C' + str(label), marker='o', s=50)
    
    # Plot points along the line
    for i, point in enumerate(points):
        plt.scatter(point[0], point[1], c=colors[i], marker='o', s=50)
    
    attractor1, attractor2 = attractors
    plt.scatter(attractor1[0], attractor1[1], c='r', label='Attractor 1', s=100)
    plt.scatter(attractor2[0], attractor2[1], c='g', label='Attractor 2', s=100)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Trajectories from Points along Line between Attractors')
    plt.legend()
    plt.show()

def find_saddle_point(dynamics_function, point_on_separatrix, T=1, steps=100, return_all=False):
    """
    Find a saddle point by following a trajectory from a point on the separatrix and finding the point with minimum kinetic energy.
    
    Args:
        dynamics_function: Function that defines the system dynamics
        point_on_separatrix: Starting point on the separatrix
        T: Integration time (default: 1)
        steps: Number of integration steps (default: 100)
        
    Returns:
        saddle_point: The point with minimum kinetic energy along the trajectory
        eigenvalues: Eigenvalues of the Jacobian at the saddle point
        trajectory: The full trajectory
        ke_traj: Kinetic energy along the trajectory
    """
    # Define kinetic energy function
    def kinetic_energy(x):
        return torch.sum(dynamics_function(x)**2) / 2
    
    # Run trajectory starting from point on separatrix
    time_points = torch.linspace(0, T, steps)
    with torch.no_grad():
        trajectory = odeint(lambda t,x: dynamics_function(x), point_on_separatrix, time_points)
    
    # Calculate kinetic energy along trajectory
    ke_traj = torch.tensor([kinetic_energy(x) for x in trajectory])
    
    # Find point with minimum kinetic energy along trajectory
    min_ke_idx = torch.argmin(ke_traj)
    saddle_point = trajectory[min_ke_idx]

    # Refine saddle point using Adam optimizer to minimize kinetic energy
    x = saddle_point.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=0.005)
    
    for i in range(500):
        optimizer.zero_grad()
        ke = kinetic_energy(x)
        ke.backward()
        optimizer.step()
        if i % 100 == 0:
            print('iteration',i,'kinetic energy',ke)
        if ke < 1e-10:  # Early stopping if kinetic energy is very small
            break
            
    saddle_point = x.detach()

    if return_all:
        # Compute Jacobian at saddle point using autograd
        x = saddle_point.clone().detach().requires_grad_(True)
        jacobian = torch.autograd.functional.jacobian(dynamics_function, x)
        eigenvalues = torch.linalg.eigvals(jacobian)
        return saddle_point, eigenvalues, trajectory, ke_traj 
    else:
        return saddle_point