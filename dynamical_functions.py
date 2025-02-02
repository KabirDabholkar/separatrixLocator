import torch
import numpy as np

def radial_monostable(z):
    x, y = z[...,0], z[...,1]
    r = torch.sqrt(x**2 + y**2)
    dxdt = x * (1 - r) - y
    dydt = y * (1 - r) + x
    return torch.stack([dxdt, dydt],dim=-1)

def radial_bistable(z):
    x, y = z[...,0], z[...,1]
    r = torch.sqrt(x**2 + y**2)
    dxdt = -x * (r - 1) * (r - 2) - y
    dydt = -y * (r - 1) * (r - 2) + x
    return torch.stack([dxdt, dydt],dim=-1)


def radial_to_cartesian(radial_dynamics):
    """
    Convert a dynamical system in radial coordinates to Cartesian coordinates.

    Parameters:
    - radial_dynamics: function that takes a state (r, θ) and returns
      the dynamics (dr/dt, dθ/dt) in radial coordinates.

    Returns:
    - A function that takes a state (x, y) and returns the dynamics
      (dx/dt, dy/dt) in Cartesian coordinates.
    """

    def cartesian_dynamics(z):
        x, y = z[..., 0], z[..., 1]
        # Compute r and θ from Cartesian coordinates
        r = torch.sqrt(x ** 2 + y ** 2)
        theta = torch.arctan2(y, x)

        # Get dynamics in radial coordinates
        r_theta_dt = radial_dynamics(torch.stack([r, theta],axis=-1))
        dr_dt, dtheta_dt = r_theta_dt[..., 0], r_theta_dt[..., 1]

        # Convert to Cartesian derivatives
        dx_dt = dr_dt * torch.cos(theta) - r * dtheta_dt * torch.sin(theta)
        dy_dt = dr_dt * torch.sin(theta) + r * dtheta_dt * torch.cos(theta)

        return torch.stack([dx_dt, dy_dt], axis=-1)

    return cartesian_dynamics

def radial_bistable(x):
    r,theta = x[...,0:1],x[...,1:2]
    drdt = (r-2) - (r-2)**3
    dthetadt = -torch.sin(theta)
    return torch.concat([drdt, dthetadt],dim=-1)

def analytical_phi(z,mu = 1.5):
    # r, theta = x[..., 0:1], x[..., 1:2]
    x, y = z[..., 0:1], z[..., 1:2]
    # Compute r and θ from Cartesian coordinates
    r = torch.sqrt(x ** 2 + y ** 2)
    theta = torch.arctan2(y, x)
    # theta_func = torch.abs(torch.sin(theta)**(-1) - torch.tan(theta)**(-1))**(lam-1)
    theta_func = torch.abs(torch.tan(theta/2)) ** (mu - 1)
    # r_func = torch.abs((2-r)/(1-r)/r) ** (mu)
    # r_func = torch.abs((2 - r) / torch.sqrt(torch.abs(r-2**2-4*r+3))) ** (mu)
    r_func = torch.abs((r-2) / torch.sqrt(torch.abs((r-2) ** 2 - 1))) ** (mu)
    return theta_func*r_func

def hopfield(z,A):
    return ( -z + torch.tanh(z) @ (A @ A.T) ) * 5

def init_hopfield(N,R,seed=0,binary=True,normalise=True,scaling=4):
    torch.manual_seed(seed)
    a = torch.randn((N,R))
    if binary:
        a = (a > 1) * 2 - 1
        a = (a).to(torch.float)
    if normalise:
        a /= torch.linalg.norm(a,axis=0,keepdims=True)
    if scaling is not None:
        a *= scaling
    return a

def init_hopfield_ring(N,M,seed=0,binary=True,normalise=True,scaling=4):
    torch.manual_seed(seed)
    theta = torch.arange(0,M) * torch.pi/M
    a = torch.randn((N,2))
    a /= torch.linalg.norm(a, axis=0, keepdims=True)
    a = a @ torch.stack([torch.cos(theta),torch.sin(theta)],axis=0)
    print(theta)

    if binary:
        a = (a > 1) * 2 - 1
        a = (a).to(torch.float)
    if normalise:
        a /= torch.linalg.norm(a,axis=0,keepdims=True)
    if scaling is not None:
        a *= scaling
    return a


if __name__ == '__main__':
    from functools import partial
    A = init_hopfield(10,3,seed=0)
    print(
        partial(hopfield,A=init_hopfield(10,3,seed=0))(torch.ones(10,1)).shape
    )


